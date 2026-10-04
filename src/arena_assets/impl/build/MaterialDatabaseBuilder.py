from __future__ import annotations

import json
import os
import shutil

import attrs

from arena_assets.impl import ANNOTATION_NAME, Annotation, AssetType
from arena_assets.utils.logging import get_logger

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.material")


@attrs.define
class MaterialAnnotation(Annotation):
    pass


class MaterialDatabaseBuilder(DatabaseBuilder[MaterialAnnotation]):
    _annotation_cls = MaterialAnnotation
    _DISCOVER_PATH = "Material"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def store_db(annotation: MaterialAnnotation) -> None:
            self._db.store(AssetType.MATERIAL.value, annotation)

        self._pipeline.append(store_db)

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
            logger.error(
                "Unexpected error processing entity %s: %s", annotation.path, e
            )
            return None

    enable = OptionRegistry()

    @enable.register("procthor")
    def procthor(self):
        logger.info("Enabling Procthor export option.")
        self._procthor = {}
        self._pipeline.append(
            lambda annotation: self._procthor.update(
                {annotation.path.replace(os.sep, "_"): {"todo": None}}
            )
        )

        def export():
            with open(self.output_path / "arena-materials-database.json", "w") as f:
                json.dump(self._procthor, f, indent=4)

        self._post.append(export)
