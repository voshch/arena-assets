from __future__ import annotations

import json
import math
import shutil
import typing
from pathlib import Path

import attrs
import numpy as np
from cattrs.gen import make_dict_unstructure_fn, override

from arena_assets.impl import ANNOTATION_NAME, AssetType, Face, SpatialAnnotation, converter
from arena_assets.utils.human import Legs, PelvisClearance, legs, pelvis_clearance, read_skin
from arena_assets.utils.logging import get_logger
from arena_assets.utils.ModelConverter import BlenderWorker

from . import DatabaseBuilder, OptionRegistry, OverwriteMode

logger = get_logger("build.human")


_FORWARD = {Face.POS_X: (1.0, 0.0), Face.NEG_X: (-1.0, 0.0), Face.POS_Y: (0.0, 1.0), Face.NEG_Y: (0.0, -1.0)}


@attrs.define
class HumanAnnotation(SpatialAnnotation):
    pelvis_clearance: PelvisClearance | None = None
    legs: Legs | None = None

    @property
    def as_metadata(self) -> dict:
        metadata = super().as_metadata
        if self.pelvis_clearance is not None:
            metadata["pelvis_clearance"] = json.dumps(attrs.asdict(self.pelvis_clearance))
        if self.legs is not None:
            metadata["legs"] = json.dumps(attrs.asdict(self.legs))
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        annotation = super().from_metadata(metadata)
        if clearance := metadata.get("pelvis_clearance"):
            annotation.pelvis_clearance = PelvisClearance(**json.loads(clearance))
        if measured := metadata.get("legs"):
            annotation.legs = Legs(**json.loads(measured))
        return annotation


converter.register_unstructure_hook(PelvisClearance, make_dict_unstructure_fn(PelvisClearance, converter, seated=override(omit_if_default=True)))
converter.register_unstructure_hook(
    HumanAnnotation,
    make_dict_unstructure_fn(HumanAnnotation, converter, pelvis_clearance=override(omit_if_default=True), legs=override(omit_if_default=True)),
)


def measure_body(annotation: HumanAnnotation, dae: Path) -> None:
    """Annotate the body's pelvis clearance and legs, left None (with a warning) when its model cannot be measured."""
    annotation.pelvis_clearance = annotation.legs = None
    forward = _FORWARD.get(annotation.face)
    if forward is None:
        logger.warning("%s faces %s, not along an axis: no body measurements", annotation.path, annotation.face.value)
        return
    try:
        skin = read_skin(dae)
        annotation.pelvis_clearance, annotation.legs = pelvis_clearance(skin, np.array(forward)), legs(skin)
    except (OSError, KeyError, ValueError) as e:
        logger.warning("%s: no body measurements: %s", annotation.path, e)


class HumanDatabaseBuilder(DatabaseBuilder[HumanAnnotation]):
    _annotation_cls = HumanAnnotation
    _DISCOVER_PATH = "Human"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def store_db(annotation: HumanAnnotation) -> None:
            self._db.store(AssetType.HUMAN.value, annotation)

        self._pipeline.append(store_db)

        def measure_built(annotation: HumanAnnotation) -> None:
            measure_body(annotation, self.output_path / annotation.path / "meshes" / f"{annotation.name}.dae")

        if self._overwrite == OverwriteMode.ANNOTATIONS_ONLY:
            self._pipeline.insert(0, measure_built)

        self._previews_enabled = False
        self._blender = BlenderWorker()

    def process_entity(self, annotation, dest):
        try:
            shutil.copytree(
                annotation.path,
                dest,
                ignore=shutil.ignore_patterns(ANNOTATION_NAME),
                dirs_exist_ok=True,
            )
            measure_body(annotation, dest / "meshes" / f"{annotation.name}.dae")

            if self._previews_enabled:
                self._render_preview(annotation, dest)

            return annotation

        except Exception as e:
            logger.error("Unexpected error processing entity %s: %s", annotation.path, e)
            return None

    def _render_preview(self, annotation: HumanAnnotation, dest: Path) -> None:
        result = self._blender.run(
            [
                ("load", str(dest / "meshes" / f"{annotation.name}.dae")),
                ("bind_uv_maps",),
                ("render_perspective", str(dest / f"{annotation.name}_thumb.png"), {"resolution": (512, 512), "theta": math.radians(annotation.face.angle + 30), "elevation": math.pi / 2, "ambient": 0.5}),
            ]
        )
        for line in result.stderr.splitlines():
            logger.warning("ModelConverter: %s", line)

    enable = OptionRegistry()

    @enable.register("previews")
    def previews(self):
        logger.info("Enabling preview generation option.")
        self._previews_enabled = True
        if self._overwrite != OverwriteMode.ANNOTATIONS_ONLY:
            self._pre.append(self._blender.start)
        self._post.append(self._blender.close)
