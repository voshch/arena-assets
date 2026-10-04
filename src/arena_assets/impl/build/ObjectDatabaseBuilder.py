from __future__ import annotations

import json
import math
import os
import shutil
import typing
from pathlib import Path

import attrs

from arena_assets import semantic
from arena_assets.impl import AssetType, SpatialAnnotation
from arena_assets.utils.logging import get_logger
from arena_assets.utils.ModelConverter import ModelFormat
from arena_assets.utils.ModelConverter.UsdBaker import UsdBaker

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.object")


@attrs.define
class ObjectAnnotation(SpatialAnnotation):
    @property
    def as_text(self):
        sections = [self.name_text, semantic.words(self.tags), self.desc, self.note]
        return ". ".join(section for section in sections if section)

    @property
    def as_procthor(self) -> dict:
        hoi = self.facets.get("hoi", [])
        return {
            "assetId": self.path,
            "bounding_box": {
                "x": self.bounding_box.max_x - self.bounding_box.min_x,
                "y": self.bounding_box.max_y - self.bounding_box.min_y,
                "z": self.bounding_box.max_z - self.bounding_box.min_z,
            },
            "objectType": self.desc,
            "tags": self.tags,
            "primaryProperty": hoi[0] if hoi else "",
            "secondaryProperties": hoi[1:],
            "materials": [[material, material] for material in self.facets.get("material", [])],
            "note": self.note,
        }

    @property
    def as_gpt_meta(self) -> dict:
        return {
            "name": self.name,
            "desc": self.desc,
            "face": self.face.angle,
            "bounding_box": {
                "x": self.bounding_box.max_x - self.bounding_box.min_x,
                "y": self.bounding_box.max_y - self.bounding_box.min_y,
                "z": self.bounding_box.max_z - self.bounding_box.min_z,
            },
            "note": self.note,
        }


class ObjectDatabaseBuilder(DatabaseBuilder[ObjectAnnotation]):
    _annotation_cls = ObjectAnnotation
    _DISCOVER_PATH = "Object"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._formats: set[ModelFormat] = {
            ModelFormat.USDZ,
            ModelFormat.FBX,
            ModelFormat.OBJ,
        }
        self._usd_baker: typing.Optional[UsdBaker] = None

        def store_db(annotation: ObjectAnnotation) -> None:
            self._db.store(AssetType.OBJECT.value, annotation)

        self._pipeline.append(store_db)

        self._previews_enabled = False

    @classmethod
    def find_mainfile(cls, annotation: ObjectAnnotation) -> str | None:
        from arena_assets.utils.ModelConverter.converter import ModelConverter

        return next(
            (
                os.path.join(
                    os.path.relpath(walk[0], annotation.path),
                    f,
                )
                for walk in os.walk(annotation.path)
                for f in walk[2]
                if f.endswith(ModelConverter.exts())
            ),
            None,
        )

    @classmethod
    def is_usd(cls, filename: str) -> bool:
        usd_extensions = {
            ModelFormat.USD,
            ModelFormat.USDA,
            ModelFormat.USDC,
            ModelFormat.USDZ,
        }
        return any(
            os.path.splitext(filename.lower())[-1].lstrip(".") == ext.value
            for ext in usd_extensions
        )

    def process_entity(self, annotation, dest):
        from arena_assets.utils.ModelConverter.converter import ModelConverter

        baked_usd_path: Path | None = None
        try:
            main_file = self.find_mainfile(annotation)
            if main_file is None:
                logger.warning("No supported model file found in %s", annotation.path)
                return None

            input_file = Path(annotation.path) / main_file

            if self._usd_baker is not None and self.is_usd(main_file):
                baked_usd_path = dest / "build" / f"{annotation.name}.fbx"
                baked_usd_path.parent.mkdir(exist_ok=True)

                self._usd_baker.convert(
                    str(input_file.relative_to(self.input_path)),
                    str(baked_usd_path.relative_to(self.output_path)),
                )
                logger.info("Baked USD model %s to %s", input_file, baked_usd_path)
                input_file = baked_usd_path

            model_paths = (
                dest / f"{annotation.name}.{ext.value}" for ext in self._formats
            )

            with ModelConverter() as model_converter:
                model_converter.load(str(input_file))
                model_converter.rectify()
                bounding_box = model_converter.bounding_box().round(4)

                if self._previews_enabled:
                    # model_converter.render_perspective(str(dest / f"{annotation.name}.png"), resolution=None, theta=annotation.face.angle + 30)
                    model_converter.render_perspective(
                        str(dest / f"{annotation.name}_thumb.png"),
                        resolution=(512, 512),
                        theta=math.radians(annotation.face.angle + 30),
                    )
                    model_converter.render_topdown(
                        str(dest / f"{annotation.name}_topdown.png"),
                        resolution=(512, 512),
                    )
                for model_path in model_paths:
                    model_path.parent.mkdir(exist_ok=True)
                    model_converter.save(str(model_path))

            logger.info("Processed model %s", annotation.path)
            for line in model_converter.stdout.splitlines():
                if line.startswith("Warning:"):
                    logger.warning("ModelConverter: %s", line)
                else:
                    logger.info("ModelConverter: %s", line)
            for line in model_converter.stderr.splitlines():
                logger.warning("ModelConverter: %s", line)

            annotation.bounding_box = bounding_box

            return annotation

        except Exception as e:
            logger.error(
                "Unexpected error processing entity %s: %s", annotation.path, repr(e)
            )
            return None

        finally:
            pass
            if baked_usd_path is not None:
                try:
                    baked_usd_path.unlink()
                    shutil.rmtree(baked_usd_path.parent)
                    logger.debug("Removed temporary baked file %s", baked_usd_path)
                except Exception as e:
                    logger.warning(
                        "Failed to remove temporary baked file %s: %s",
                        baked_usd_path,
                        e,
                    )

    enable = OptionRegistry()

    @enable.register("formats")
    def sdf(self, formats: str | None = None):
        self._formats.clear()
        if formats is None:
            formats = "*"
        if formats == "*":
            self._formats = set(ModelFormat.__members__.values())
            return
        for fmt in formats.split(","):
            try:
                self._formats.add(ModelFormat(fmt.lower()))
                logger.info("Enabling %s export option.", fmt)
            except ValueError:
                logger.error("Unknown model format specified: %s", fmt)

    @enable.register("procthor")
    def procthor(self):
        logger.info("Enabling Procthor export option.")
        self._procthor = {}
        self._pipeline.append(
            lambda annotation: self._procthor.update(
                {annotation.path.replace(os.sep, "_"): annotation.as_procthor}
            )
        )

        def export():
            with open(os.path.join(self.output_path, "asset-database.json"), "w") as f:
                json.dump(self._procthor, f, indent=4)

        self._post.append(export)

    @enable.register("bake-mdl")
    def bake_mdl(self, isaacsim_path: str | None = None):
        if isaacsim_path is not None:
            from arena_assets.utils.ModelConverter.UsdBaker.LocalUsdBaker import (
                LocalUsdBaker,
            )

            self._usd_baker = LocalUsdBaker(
                input_dir=self.input_path,
                output_dir=self.output_path,
                isaacsim_path=Path(isaacsim_path),
            )

        else:
            from arena_assets.utils.ModelConverter.UsdBaker.DockerUsdBaker import (
                DockerUsdBaker,
            )

            self._usd_baker = DockerUsdBaker(
                input_dir=Path(self.input_path),
                output_dir=Path(self.output_path),
            )

        self._pre.append(self._usd_baker.start)

        def cleanup_baker():
            if self._usd_baker is not None:
                self._usd_baker.cleanup()

        self._post.append(cleanup_baker)

    @enable.register("previews")
    def previews(self):
        logger.info("Enabling preview generation option.")
        self._previews_enabled = True
