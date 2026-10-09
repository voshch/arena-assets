from __future__ import annotations

import enum
import json
import math
import os
import shutil
import typing
from collections.abc import Iterable
from pathlib import Path

import attrs
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from arena_assets import semantic
from arena_assets.impl import AssetType, SpatialAnnotation, converter
from arena_assets.utils.logging import get_logger
from arena_assets.utils.ModelConverter import ModelFormat
from arena_assets.utils.ModelConverter.lights import DEFAULT_CCT_K, SourceLight, SourceMaterial, Vector3, color_temperature, nearest_material
from arena_assets.utils.ModelConverter.UsdBaker import UsdBaker

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.object")

LUMENS_PER_WATT = 683.0
MAX_CONE_DEG = 179.99


class LightFixture(enum.StrEnum):
    BULB = "bulb"
    SPOT = "spot"
    DOWNLIGHT = "downlight"
    PANEL = "panel"
    TUBE = "tube"


def _vector(value: Iterable[float]) -> Vector3:
    x, y, z = (float(component) for component in value)
    return (x, y, z)


def _positive(instance: object, attribute: attrs.Attribute, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}")


def _nonzero(instance: object, attribute: attrs.Attribute, value: Vector3) -> None:
    if not any(value):
        raise ValueError(f"{attribute.name} must be nonzero, got {value}")


def _cone(instance: object, attribute: attrs.Attribute, value: float) -> None:
    if not 0.0 < value < 180.0:
        raise ValueError(f"{attribute.name} must be in (0, 180), got {value}")


@attrs.define(kw_only=True)
class ObjectLight:
    """Light fixture carried by an object, in the frame of the built model."""

    name: str = attrs.field(default="light", validator=attrs.validators.matches_re(r"[a-z0-9_]+"))
    fixture: LightFixture = attrs.field(converter=LightFixture)
    offset: Vector3 = attrs.field(converter=_vector)
    direction: Vector3 = attrs.field(default=(0.0, 0.0, -1.0), converter=_vector, validator=_nonzero)
    lumens: float = attrs.field(converter=float, validator=_positive)
    cct_K: float = attrs.field(default=DEFAULT_CCT_K, converter=float, validator=_positive)
    cone_deg: float | None = attrs.field(default=None, converter=attrs.converters.optional(float), validator=attrs.validators.optional(_cone))
    glow: str | None = attrs.field(default=None, converter=attrs.converters.optional(str))


def _structure_lights(value: Iterable[typing.Any] | None) -> list[ObjectLight]:
    return [converter.structure(entry, ObjectLight) for entry in value or ()]


def _unique_names(instance: object, attribute: attrs.Attribute, value: list[ObjectLight]) -> None:
    names = [light.name for light in value]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"{attribute.name} names must be unique, duplicated: {', '.join(duplicates)}")


def _round(value: float, ndigits: int) -> float:
    return round(value, ndigits) + 0.0


def _fixture(source: SourceLight) -> LightFixture | None:
    match source.type:
        case "POINT":
            return LightFixture.BULB
        case "SPOT":
            return LightFixture.SPOT
        case "AREA":
            return LightFixture.PANEL if source.shape in {"SQUARE", "RECTANGLE"} else LightFixture.DOWNLIGHT
    return None


def object_lights(sources: Iterable[SourceLight], materials: Iterable[SourceMaterial] = ()) -> list[ObjectLight]:
    """Object lights for the lights of a source model, named light, light_1, ... in source name order, each glowing the nearest material."""
    materials = list(materials)
    mapped: list[tuple[LightFixture, SourceLight, float]] = []
    for source in sorted(sources, key=lambda source: source.name):
        fixture = _fixture(source)
        if fixture is None:
            logger.warning("Skipping %s light %s, an object cannot carry it", source.type, source.name)
            continue
        lumens = _round(source.power * LUMENS_PER_WATT, 1)
        if lumens <= 0.0:
            logger.warning("Skipping light %s with power %s W", source.name, source.power)
            continue
        mapped.append((fixture, source, lumens))

    lights = []
    for index, (fixture, source, lumens) in enumerate(mapped):
        norm = math.hypot(*source.direction)
        lights.append(
            ObjectLight(
                name="light" if index == 0 else f"light_{index}",
                fixture=fixture,
                offset=tuple(_round(component, 4) for component in source.location),
                direction=tuple(_round(component / norm, 4) for component in source.direction),
                lumens=lumens,
                cct_K=color_temperature(source.color),
                cone_deg=min(_round(math.degrees(source.spot_size), 2), MAX_CONE_DEG) if fixture is LightFixture.SPOT else None,
                glow=nearest_material(source.location, materials),
            )
        )
    return lights


@attrs.define
class ObjectAnnotation(SpatialAnnotation):
    lights: list[ObjectLight] = attrs.field(factory=list, converter=_structure_lights, validator=_unique_names)

    @property
    def as_metadata(self) -> dict:
        if not self.lights:
            return super().as_metadata
        return {**super().as_metadata, "lights": json.dumps(converter.unstructure(self.lights))}

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        annotation = super().from_metadata(metadata)
        if lights := metadata.get("lights"):
            annotation.lights = json.loads(lights)
        return annotation

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


_structure_light_fields = make_dict_structure_fn(ObjectLight, converter, _cattrs_detailed_validation=False)


def _structure_light(value: typing.Any, _: type) -> ObjectLight:
    if isinstance(value, ObjectLight):
        return value
    try:
        return _structure_light_fields(value, ObjectLight)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid light {value!r}: {e!r}") from e


converter.register_structure_hook(ObjectLight, _structure_light)
converter.register_unstructure_hook(ObjectLight, make_dict_unstructure_fn(ObjectLight, converter, cone_deg=override(omit_if_default=True), glow=override(omit_if_default=True)))
converter.register_structure_hook(ObjectAnnotation, make_dict_structure_fn(ObjectAnnotation, converter, _cattrs_detailed_validation=False))
converter.register_unstructure_hook(ObjectAnnotation, make_dict_unstructure_fn(ObjectAnnotation, converter, lights=override(omit_if_default=True)))


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
                source_lights = model_converter.lights()
                model_converter.remove_lights()
                extracted = object_lights(source_lights, model_converter.emissive_materials())
                model_converter.prepare_glow(light.glow for light in annotation.lights or extracted if light.glow is not None)

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

            if annotation.lights:
                logger.info("Keeping %d annotated lights of %s, skipped extracting %d source lights", len(annotation.lights), annotation.path, len(extracted))
            else:
                annotation.lights = extracted

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
