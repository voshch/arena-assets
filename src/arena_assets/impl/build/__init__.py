from __future__ import annotations

import abc
import os
from enum import Enum
from pathlib import Path
import shutil
import typing
from collections.abc import Callable, Iterable, Iterator

import yaml

from arena_assets.impl import (
    ANNOTATION_NAME,
    DATABASE_NAME,
    Annotation,
    AssetType,
    converter,
    serialization_asset_context,
)
from arena_assets.utils.Database import Database
from arena_assets.utils.logging import get_logger, get_manager

logger = get_logger("build")


class OverwriteMode(str, Enum):
    SKIP = "skip"
    OVERWRITE = "overwrite"
    ANNOTATIONS_ONLY = "annotations"

    @classmethod
    def _missing_(cls, value):
        """
        Legacy integer API shim. Accepts 0 (SKIP), 1 (OVERWRITE), -1 (ANNOTATIONS_ONLY).
        """
        legacy = {
            "0": cls.SKIP,
            "1": cls.OVERWRITE,
            "-1": cls.ANNOTATIONS_ONLY,
        }
        return legacy.get(str(value))


class OptionRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(func):
            self._registry[name] = func
            return func

        return decorator

    def __get__(self, instance, _):
        class BoundRegistry:
            def __init__(self, registry, instance):
                self._registry = registry
                self._instance = instance

            def __call__(self, name, *args, **kwargs):
                if name in self._registry:
                    func = self._registry[name]
                    return func.__get__(self._instance, type(self._instance)).__call__(*args, **kwargs)

        return BoundRegistry(self._registry, instance)


AnnotationT = typing.TypeVar("AnnotationT", bound=Annotation)


class DatabaseBuilder(abc.ABC, typing.Generic[AnnotationT]):
    __registry: typing.ClassVar[dict[AssetType, typing.Callable[[AssetType], typing.Type[DatabaseBuilder]]]] = {}
    _type: typing.ClassVar[AssetType]

    @classmethod
    def register(cls, type_: AssetType, /) -> typing.Callable[[typing.Callable[[AssetType], typing.Type[DatabaseBuilder]]], None]:
        def wrapper(
            builder_cls: typing.Callable[[AssetType], typing.Type[DatabaseBuilder]],
        ) -> None:
            cls.__registry[type_] = builder_cls

        return wrapper

    @classmethod
    def get_registered(cls) -> Iterable[AssetType]:
        return cls.__registry.keys()

    @classmethod
    def get_all_options(cls) -> dict[AssetType, list[str]]:
        options = {}
        for type_, builder_fn in cls.__registry.items():
            builder_cls = builder_fn(type_)
            options[type_] = list(builder_cls.enable._registry.keys())
        return options

    @classmethod
    def Builder(cls, type_: AssetType) -> typing.Type[DatabaseBuilder]:
        if type_ not in cls.__registry:
            raise ValueError(f"No builder registered for type: {type_}")
        builder_cls = cls.__registry[type_](type_)
        builder_cls._type = type_
        return builder_cls

    _annotation_cls: typing.ClassVar[typing.Type[Annotation]]

    @classmethod
    def read_annotation_file(cls, dir_path: Path) -> AnnotationT | Exception:
        try:
            with open(dir_path / ANNOTATION_NAME, "r") as yaml_file:
                return typing.cast(
                    AnnotationT,
                    converter.structure(
                        {
                            **yaml.safe_load(yaml_file),
                            "name": dir_path.name,
                            "path": dir_path,
                        },
                        cls._annotation_cls,
                    ),
                )
        except Exception as e:
            return e

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        overwrite: OverwriteMode = OverwriteMode.SKIP,
        **kwargs,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self._overwrite = overwrite

        self._pre: list[Callable] = []
        self._pipeline: list[Callable[[AnnotationT], typing.Any]] = []
        self._post: list[Callable] = []

        self._db = Database(path=self.output_path / DATABASE_NAME)

        def save_annotation(annotation: AnnotationT) -> None:
            annotation_path = self.output_path / annotation.path / ANNOTATION_NAME
            os.makedirs(annotation_path.parent, exist_ok=True)
            with open(annotation_path, "w") as f:
                with serialization_asset_context(str(annotation.path)):
                    yaml.safe_dump(converter.unstructure(annotation), f)

        self._pipeline.append(save_annotation)

    enable: OptionRegistry

    def build(self):
        logger.info("Starting database build...")
        manager = get_manager()

        status_bar = manager.status_bar(
            status_format="Building {build_type} database from {input_path}: {stage}{fill}",
            input_path=str(self.input_path),
            build_type=self._type.name,
            stage="Initializing",
        )

        status_bar.update(stage="Discovery")

        skipped_count = 0

        def skip_existing(annotation: AnnotationT) -> bool:
            dest_path = self._dest_path(annotation)

            if (dest_path / ANNOTATION_NAME).exists() and self._overwrite == OverwriteMode.SKIP:
                logger.info("Skipping existing entity at %s", annotation.path)
                nonlocal skipped_count
                skipped_count += 1
                return False
            return True

        queue = list(filter(skip_existing, self.discover()))

        if not queue:
            logger.info("Skipped %s build: No entities found.", self._type.name)
            status_bar.update(stage="Done (skipped)")
        else:
            if self._pre:
                status_bar.update(stage="Pre-processing")
                progress = manager.counter(
                    total=len(self._pre),
                    desc="Pre-processing",
                    unit="steps",
                    format="{desc}{desc_pad}{count:d} {unit}{unit_pad}{elapsed}]{fill}",
                    leave=False,
                )
                with progress:
                    logger.info("Running pre-processing...")
                    for fn in progress(self._pre):
                        fn()

            status_bar.update(stage="Processing")
            progress = manager.counter(total=len(queue), desc="Entities", unit="entities", leave=False)

            with progress:
                success = progress.add_subcounter("green")
                failure = progress.add_subcounter("red")
                skipped = progress.add_subcounter("blue")
                progress.update(skipped_count)
                skipped.update_from(progress, skipped_count)
                del skipped_count

                for annotation in queue:
                    progress.update()
                    status_bar.update(stage=f"Processing {annotation.path}")

                    target_path = self._target_path(annotation)
                    dest_path = self._dest_path(annotation)

                    logger.info("Processing entity at %s", annotation.path)
                    os.makedirs(dest_path, exist_ok=True)

                    if self._overwrite != OverwriteMode.ANNOTATIONS_ONLY:
                        annotation = self.process_entity(annotation, dest_path)
                        if annotation is None:
                            shutil.rmtree(dest_path)
                            failure.update_from(progress, 1)
                            continue

                    annotation.name = annotation.name or target_path.name
                    annotation.path = str(target_path)
                    for fn in self._pipeline:
                        try:
                            fn(annotation)
                        except Exception:
                            logger.exception(
                                "Failed to process pipeline function %s for annotation %s",
                                fn,
                                annotation,
                            )
                            shutil.rmtree(dest_path)
                            failure.update_from(progress, 1)
                            break
                    else:
                        success.update_from(progress, 1)

            if self._post:
                status_bar.update(stage="Post-processing")
                progress = manager.counter(
                    total=len(self._post),
                    desc="Post-processing",
                    unit="steps",
                    format="{desc}{desc_pad}{count:d} {unit}{unit_pad}{elapsed}]{fill}",
                    leave=False,
                )
                with progress:
                    logger.info("Running export...")
                    for fn in progress(self._post):
                        fn()

            # actually looked these up by hand, LLMs won't replace me just yet
            status_bar.update(stage=f"Done ✓{success.count} ✗{failure.count} ➤{skipped.count}")

        status_bar.close()
        logger.info("Database build complete.")

    _DISCOVER_PATH: typing.ClassVar[str] = ""

    def discover(
        self,
        *,
        base_path: str | None = None,
        filter_: typing.Callable[[str], bool] | None = None,
    ) -> Iterator[AnnotationT]:

        if filter_ is None:

            def filter__(path):
                del path
                return True

            filter_ = filter__

        if base_path is None:
            base_path = self._DISCOVER_PATH

        for root, dirs, files in os.walk(
            self.input_path / base_path,
        ):
            if not filter_(root):
                continue
            dirs.sort()
            files.sort()
            logger.info("Scanning directory: %s", root)
            if ANNOTATION_NAME in files:
                # don't recurse into subdirs
                dirs.clear()
                annotation = self.read_annotation_file(Path(root))
                if isinstance(annotation, Exception):
                    logger.warning("Failed to read annotation file in %s: %s", root, annotation)
                elif annotation is not None:
                    yield annotation

    @abc.abstractmethod
    def process_entity(self, annotation: AnnotationT, dest: Path) -> AnnotationT | None: ...

    def _target_path(self, annotation: AnnotationT) -> Path:
        return Path(annotation.path).relative_to(self.input_path)

    def _dest_path(self, annotation: AnnotationT) -> Path:
        return self.output_path / self._target_path(annotation)


@DatabaseBuilder.register(AssetType.MATERIAL)
def lazy_material(type_: AssetType) -> typing.Type[DatabaseBuilder]:
    from .MaterialDatabaseBuilder import MaterialDatabaseBuilder

    return MaterialDatabaseBuilder


@DatabaseBuilder.register(AssetType.OBJECT)
def lazy_object(type_: AssetType) -> typing.Type[DatabaseBuilder]:
    from .ObjectDatabaseBuilder import ObjectDatabaseBuilder

    return ObjectDatabaseBuilder


@DatabaseBuilder.register(AssetType.HUMAN)
def lazy_human(type_: AssetType) -> typing.Type[DatabaseBuilder]:
    from .HumanDatabaseBuilder import HumanDatabaseBuilder

    return HumanDatabaseBuilder


@DatabaseBuilder.register(AssetType.SOUND)
def lazy_sound(type_: AssetType) -> typing.Type[DatabaseBuilder]:
    from .SoundDatabaseBuilder import SoundDatabaseBuilder

    return SoundDatabaseBuilder
