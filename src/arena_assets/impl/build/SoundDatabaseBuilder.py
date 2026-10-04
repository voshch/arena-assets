from __future__ import annotations

import os
import shutil
import typing
from collections.abc import Iterator
from pathlib import Path

import attrs
import yaml

from arena_assets.impl import ANNOTATION_NAME, Annotation, AssetType, convert_list_str
from arena_assets.utils.logging import get_logger

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.sound")

MANIFEST_VERSION = 2


def _tag_list(value: object, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise ValueError(f"{where}: tags must be a list of strings")
    return value


@attrs.define
class SoundAnnotation(Annotation):
    kind: str = ""
    loop: bool = False
    level_db: float = 0.0
    variants: list[str] = attrs.field(factory=list, converter=convert_list_str)

    @property
    def as_metadata(self) -> dict:
        return {
            **super().as_metadata,
            "kind": self.kind,
            "loop": self.loop,
            "level_db": self.level_db,
            "variants": ",".join(self.variants),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        return cls(
            name=metadata["name"],
            path=metadata["path"],
            desc=metadata.get("desc", ""),
            tags=tags.split(",") if (tags := metadata.get("tags")) else [],
            asa=int(metadata.get("asa", 0)),
            kind=metadata["kind"],
            loop=bool(metadata["loop"]),
            level_db=float(metadata["level_db"]),
            variants=variants.split(",") if (variants := metadata["variants"]) else [],
        )

    @classmethod
    def from_manifest(cls, directory: Path) -> typing.Self:
        """Annotation derived from the `<name>/<name>.yaml` manifest, ValueError on a malformed one."""
        manifest = directory / f"{directory.name}.yaml"
        with open(manifest) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{manifest}: manifest must be a mapping")
        if data.get("version") != MANIFEST_VERSION:
            raise ValueError(f"{manifest}: manifest version must be {MANIFEST_VERSION}, got {data.get('version')!r}")
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{manifest}: 'kind' must be a non-empty string")
        level_db = data.get("level_db")
        if isinstance(level_db, bool) or not isinstance(level_db, int | float):
            raise ValueError(f"{manifest}: 'level_db' must be a number")
        desc = data.get("desc", "")
        if not isinstance(desc, str):
            raise ValueError(f"{manifest}: 'desc' must be a string")
        variants = data.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{manifest}: 'variants' must be a non-empty list")
        if not all(isinstance(variant, dict) and variant.get("id") for variant in variants):
            raise ValueError(f"{manifest}: every variant must be a mapping with an id")
        tags = [
            *_tag_list(data.get("tags"), str(manifest)),
            *(tag for variant in variants for tag in _tag_list(variant.get("tags"), f"{manifest} variant {variant['id']!r}")),
            kind,
        ]
        return cls(
            name=directory.name,
            path=str(directory),
            desc=desc,
            tags=list(dict.fromkeys(tags)),
            kind=kind,
            loop=bool(data.get("loop", False)),
            level_db=float(level_db),
            variants=[str(variant["id"]) for variant in variants],
        )


class SoundDatabaseBuilder(DatabaseBuilder[SoundAnnotation]):
    _annotation_cls = SoundAnnotation
    _DISCOVER_PATH = "Sound"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def store_db(annotation: SoundAnnotation) -> None:
            self._db.store(AssetType.SOUND.value, annotation)

        self._pipeline.append(store_db)

    def discover(
        self,
        *,
        base_path: str | None = None,
        filter_: typing.Callable[[str], bool] | None = None,
    ) -> Iterator[SoundAnnotation]:
        if base_path is None:
            base_path = self._DISCOVER_PATH

        for root, dirs, files in os.walk(self.input_path / base_path):
            if filter_ is not None and not filter_(root):
                continue
            dirs.sort()
            logger.info("Scanning directory: %s", root)
            if f"{os.path.basename(root)}.yaml" in files:
                dirs.clear()
                yield SoundAnnotation.from_manifest(Path(root))

    def process_entity(self, annotation, dest):
        try:
            shutil.copytree(
                annotation.path,
                dest,
                ignore=shutil.ignore_patterns(ANNOTATION_NAME),
                dirs_exist_ok=True,
            )

            return annotation

        except Exception as e:
            logger.error("Unexpected error processing entity %s: %s", annotation.path, e)
            return None

    enable = OptionRegistry()
