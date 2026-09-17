from __future__ import annotations

import math
import shutil
from pathlib import Path

import attrs

from arena_models.impl import ANNOTATION_NAME, AssetType, SpatialAnnotation
from arena_models.utils.logging import get_logger

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.human")


@attrs.define
class HumanAnnotation(SpatialAnnotation):
    pass


class HumanDatabaseBuilder(DatabaseBuilder[HumanAnnotation]):
    _annotation_cls = HumanAnnotation
    _DISCOVER_PATH = "Human"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def store_db(annotation: HumanAnnotation) -> None:
            self._db.store(AssetType.HUMAN.value, annotation)

        self._pipeline.append(store_db)

        self._previews_enabled = False

    def process_entity(self, annotation, dest):
        try:
            shutil.copytree(
                annotation.path,
                dest,
                ignore=shutil.ignore_patterns(ANNOTATION_NAME),
                dirs_exist_ok=True,
            )

            if self._previews_enabled:
                self._render_preview(annotation, dest)

            return annotation

        except Exception as e:
            logger.error("Unexpected error processing entity %s: %s", annotation.path, e)
            return None

    def _render_preview(self, annotation: HumanAnnotation, dest: Path) -> None:
        from arena_models.utils.ModelConverter.converter import ModelConverter

        with ModelConverter() as model_converter:
            model_converter.load(str(dest / "meshes" / f"{annotation.name}.dae"))
            model_converter.bind_uv_maps()
            model_converter.render_perspective(
                str(dest / f"{annotation.name}_thumb.png"),
                resolution=(512, 512),
                theta=math.radians(annotation.face.angle + 30),
                elevation=math.pi / 2,
                ambient=0.5,
            )
        for line in model_converter.stderr.splitlines():
            logger.warning("ModelConverter: %s", line)

    enable = OptionRegistry()

    @enable.register("previews")
    def previews(self):
        logger.info("Enabling preview generation option.")
        self._previews_enabled = True
