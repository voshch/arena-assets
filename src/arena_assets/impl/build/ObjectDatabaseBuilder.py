from __future__ import annotations

import enum
import json
import math
import os
import shutil
import tempfile
import typing
from collections.abc import Iterable
from pathlib import Path

import attrs
import numpy as np
import yaml
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from arena_assets import semantic
from arena_assets.impl import ANNOTATION_NAME, AssetType, Face, SpatialAnnotation, collider, converter
from arena_assets.impl import articulation as link_bundles
from arena_assets.impl.affordance import Affordances, structure_affordances, unstructure_affordances
from arena_assets.impl.articulation import Articulation, ArticulationError, structure_articulation, unstructure_articulation
from arena_assets.utils import ModelConverter
from arena_assets.utils.affordance import SEAT_FALLBACK, SEAT_SIZE, largest_seat, surface_z
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.logging import get_logger
from arena_assets.utils.mesh_io import read_mesh
from arena_assets.utils.texture import MAX_TEXTURE_PX
from arena_assets.utils.ModelConverter import BlenderError, BlenderWorker, Result
from arena_assets.utils.ModelConverter.lights import DEFAULT_CCT_K, Quaternion, SourceLight, SourceMaterial, Vector3, color_temperature, nearest_material
from arena_assets.utils.ModelConverter.UsdBaker import UsdBaker

from . import DatabaseBuilder, OptionRegistry, OverwriteMode

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.affordance import AffordanceKind, AffordanceSlot, Box, Footprint, Slot
    from arena_simulation_setup.utils.articulation import Box as LinkBox

logger = get_logger("build.object")

LUMENS_PER_WATT = 683.0
MAX_CONE_DEG = 179.99


class LightFixture(enum.StrEnum):
    BULB = "bulb"
    SPOT = "spot"
    DOWNLIGHT = "downlight"
    PANEL = "panel"
    TUBE = "tube"


LightPose = tuple[Vector3, Vector3 | Quaternion]


def _pose(value: Iterable[Iterable[float]]) -> LightPose:
    position, orientation = (tuple(float(component) for component in part) for part in value)
    if len(position) != 3 or len(orientation) not in (3, 4):
        raise ValueError(f"pose must be [[x, y, z], [roll, pitch, yaw]] or [[x, y, z], [w, x, y, z]], got {value}")
    if len(orientation) == 4 and not math.isclose(math.hypot(*orientation), 1.0, abs_tol=1e-3):
        raise ValueError(f"pose orientation {orientation} is no unit quaternion")
    return (position, orientation)


def _positive(instance: object, attribute: attrs.Attribute, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}")


def _cone(instance: object, attribute: attrs.Attribute, value: float) -> None:
    if not 0.0 < value < 180.0:
        raise ValueError(f"{attribute.name} must be in (0, 180), got {value}")


@attrs.define(kw_only=True)
class ObjectLight:
    """Light fixture carried by an object, its pose in the frame of the built model, emitting along the local -z axis of that pose."""

    name: str = attrs.field(default="light", validator=attrs.validators.matches_re(r"[a-z0-9_]+"))
    fixture: LightFixture = attrs.field(converter=LightFixture)
    pose: LightPose = attrs.field(converter=_pose)
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
        lights.append(
            ObjectLight(
                name="light" if index == 0 else f"light_{index}",
                fixture=fixture,
                pose=(
                    tuple(_round(component, 4) for component in source.location),
                    (1.0, 0.0, 0.0, 0.0) if fixture is LightFixture.BULB else tuple(_round(component, 6) for component in source.rotation),
                ),
                lumens=lumens,
                cct_K=color_temperature(source.color),
                cone_deg=min(_round(math.degrees(source.spot_size), 2), MAX_CONE_DEG) if fixture is LightFixture.SPOT else None,
                glow=nearest_material(source.location, materials),
            )
        )
    return lights


def _seat_box(footprint: Footprint, floor: float, z: float) -> Box:
    return (footprint[0], footprint[1], (floor, z))


def place_slot(slot: Slot, points: np.ndarray, triangles: np.ndarray, bounding_box: BoundingBox, face: Face, *, recompute: bool, label: str) -> Slot:
    """The slot with a z range from the object floor to the seat surface, an empty slot placed on the largest seat facing the face axis."""
    from arena_simulation_setup.utils.affordance import Slot, has_z
    from arena_simulation_setup.utils.geometry import Facing

    fallback = round(min(SEAT_FALLBACK, bounding_box.max_z), 3)
    if slot.bbox is None:
        seat = largest_seat(points, triangles)
        if seat is None:
            logger.warning("%s has no seat surface, placing it at the bbox center %.3f m high", label, fallback)
            seat = ((bounding_box.min_x + bounding_box.max_x) / 2, (bounding_box.min_y + bounding_box.max_y) / 2, fallback)
        x, y, z = seat
        half = SEAT_SIZE / 2
        footprint = ((round(max(x - half, bounding_box.min_x), 3), round(min(x + half, bounding_box.max_x), 3)), (round(max(y - half, bounding_box.min_y), 3), round(min(y + half, bounding_box.max_y), 3)))
        return Slot(bbox=_seat_box(footprint, bounding_box.min_z, z), facing=slot.facing if slot.facing is not None else Facing.parse(face.value))
    (x_min, x_max), (y_min, y_max), (_, z_max) = slot.bbox
    if x_min < bounding_box.min_x or x_max > bounding_box.max_x or y_min < bounding_box.min_y or y_max > bounding_box.max_y:
        logger.warning("%s bbox %s reaches past the object bbox %s", label, slot.bbox, list(bounding_box))
    if has_z(slot.bbox) and not recompute:
        if z_max > bounding_box.max_z:
            logger.warning("%s sits at %.3f m, above the object top at %.3f m", label, z_max, bounding_box.max_z)
        return slot
    z = surface_z(points, triangles, slot.bbox[:2])
    if z is None:
        logger.warning("%s has no seat surface under its bbox, using %.3f m", label, fallback)
        z = fallback
    return Slot(bbox=_seat_box(slot.bbox[:2], bounding_box.min_z, z), facing=slot.facing)


def place_slots(annotation: ObjectAnnotation, points: np.ndarray, triangles: np.ndarray, bounding_box: BoundingBox, *, recompute: bool) -> Affordances:
    return {kind: [place_slot(slot, points, triangles, bounding_box, annotation.face, recompute=recompute, label=f"{annotation.path} {kind} slot {index}") for index, slot in enumerate(slots)] for kind, slots in annotation.affordances.items()}


BUILT_MODEL_FORMATS = ("usdz", "usd", "usdc", "usda", "fbx", "glb", "gltf", "obj", "sdf")


def _mesh(value: list) -> tuple[np.ndarray, np.ndarray]:
    """World-space vertices and triangle indices from the worker's triangles reply."""
    return np.array(value[0], dtype=np.float64).reshape(-1, 3), np.array(value[1], dtype=np.int64).reshape(-1, 3)


def built_model(dest: Path, name: str) -> Path | None:
    """The first model a build left in dest, by format preference, an SDF export standing in with the visual mesh in its directory."""
    for ext in BUILT_MODEL_FORMATS:
        path = dest / f"{name}.sdf" / f"{name}.dae" if ext == "sdf" else dest / f"{name}.{ext}"
        if path.is_file():
            return path
    return None


def carry_slots(slots: list[Slot], built: list[Slot]) -> list[Slot]:
    """Slots given the z ranges of an earlier build, matched by x and y ranges, the slot without a bbox taking the one built slot no source bbox matches."""
    from arena_simulation_setup.utils.affordance import Slot, has_z

    known = {(slot.bbox[0], slot.bbox[1]): slot.bbox for slot in built if slot.bbox is not None and has_z(slot.bbox)}
    sourced = {(slot.bbox[0], slot.bbox[1]) for slot in slots if slot.bbox is not None}
    leftover = [slot for slot in built if slot.bbox is not None and has_z(slot.bbox) and (slot.bbox[0], slot.bbox[1]) not in sourced]
    carried = []
    for slot in slots:
        if slot.bbox is None:
            placed = leftover[0] if len(leftover) == 1 else None
            same_facing = placed is not None and placed.facing is not None and (slot.facing is None or slot.facing.serialize() == placed.facing.serialize())
            carried.append(placed if placed is not None and same_facing else slot)
        elif not has_z(slot.bbox) and (slot.bbox[0], slot.bbox[1]) in known:
            carried.append(Slot(bbox=known[(slot.bbox[0], slot.bbox[1])], facing=slot.facing))
        else:
            carried.append(slot)
    return carried


@attrs.define
class ObjectAnnotation(SpatialAnnotation):
    lights: list[ObjectLight] = attrs.field(factory=list, converter=_structure_lights, validator=_unique_names)
    collider: str = ""
    affordances: Affordances = attrs.field(factory=dict, converter=structure_affordances)
    articulation: Articulation | None = attrs.field(default=None, converter=structure_articulation)

    def query_affordance(self, kind: AffordanceKind | str, slot: int = 0) -> AffordanceSlot:
        from arena_simulation_setup.utils.affordance import affordance_kind

        try:
            key = affordance_kind(kind)
        except ValueError as e:
            raise KeyError(str(e)) from None
        if key not in self.affordances:
            raise KeyError(f"{self.name} has no {key} affordance, it has: {', '.join(self.affordances) or 'none'}")
        slots = self.affordances[key]
        if not 0 <= slot < len(slots):
            raise KeyError(f"{self.name} has {len(slots)} {key} slots, got slot {slot}")
        return slots[slot].placed(key, slot)

    def affordance_slots(self) -> list[AffordanceSlot]:
        """Placed slots of every kind, the input of AffordanceClaims."""
        from arena_simulation_setup.utils.affordance import placed_slots

        return placed_slots(self.affordances)

    @property
    def as_metadata(self) -> dict:
        metadata = super().as_metadata
        if self.lights:
            metadata["lights"] = json.dumps(converter.unstructure(self.lights))
        if self.collider:
            metadata["collider"] = self.collider
        if self.affordances:
            metadata["affordances"] = json.dumps(unstructure_affordances(self.affordances))
        metadata["articulated"] = self.articulation is not None
        if self.articulation is not None:
            metadata["articulation"] = json.dumps(unstructure_articulation(self.articulation))
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        annotation = super().from_metadata(metadata)
        if lights := metadata.get("lights"):
            annotation.lights = json.loads(lights)
        annotation.collider = metadata.get("collider", "")
        if affordances := metadata.get("affordances"):
            try:
                annotation.affordances = json.loads(affordances)
            except ModuleNotFoundError as e:
                if e.name != "arena_simulation_setup":
                    raise
                logger.debug("%s keeps no affordances, reading them needs arena_simulation_setup", annotation.path)
        if articulation := metadata.get("articulation"):
            try:
                annotation.articulation = json.loads(articulation)
            except ModuleNotFoundError as e:
                if e.name != "arena_simulation_setup":
                    raise
                logger.debug("%s keeps no articulation, reading it needs arena_simulation_setup", annotation.path)
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
converter.register_unstructure_hook(
    ObjectAnnotation,
    make_dict_unstructure_fn(ObjectAnnotation, converter, lights=override(omit_if_default=True), collider=override(omit_if_default=True), affordances=override(omit_if_default=True, unstruct_hook=unstructure_affordances), articulation=override(omit_if_default=True, unstruct_hook=unstructure_articulation)),
)


class ObjectDatabaseBuilder(DatabaseBuilder[ObjectAnnotation]):
    _annotation_cls = ObjectAnnotation
    _DISCOVER_PATH = "Object"

    def __init__(self, *args, blender: BlenderWorker | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._formats: set[str] = {"usdz", "fbx", "obj"}
        self._usd_baker: UsdBaker | None = None
        self._blender = blender if blender is not None else BlenderWorker()
        if self._overwrite != OverwriteMode.ANNOTATIONS_ONLY:
            self._pre.append(self._blender.start)
            self._pre.append(self._resolve_formats)
        self._post.append(self._blender.close)

        def store_db(annotation: ObjectAnnotation) -> None:
            self._db.store(AssetType.OBJECT.value, annotation)

        self._pipeline.append(store_db)

        def carry_affordances(annotation: ObjectAnnotation) -> None:
            if not annotation.affordances:
                return
            try:
                place_on_built(annotation)
            except Exception:
                logger.warning("%s keeps its affordance slots unplaced, placing them on the built model failed", annotation.path, exc_info=True)

        def place_on_built(annotation: ObjectAnnotation) -> None:
            from arena_simulation_setup.utils.affordance import has_z

            dest = self.output_path / annotation.path
            built = dest / ANNOTATION_NAME
            if built.is_file() and not self._recompute_affordances:
                previous = structure_affordances((yaml.safe_load(built.read_text()) or {}).get("affordances"))
                for kind, slots in annotation.affordances.items():
                    annotation.affordances[kind] = carry_slots(slots, previous.get(kind, []))
            if not self._recompute_affordances and all(slot.bbox is not None and has_z(slot.bbox) for slots in annotation.affordances.values() for slot in slots):
                return
            model = built_model(dest, annotation.name)
            if model is None:
                logger.warning("%s has no built model to place its affordance slots on, run a model build", annotation.path)
                return
            values = self._blender.run([("load", str(model)), ("rectify",), ("bounding_box",), ("triangles",)]).values
            bounding_box = BoundingBox(values[2]).round(4)
            points, triangles = _mesh(values[3])
            if bounding_box.volume <= 0:
                logger.warning("%s loaded without meshes, its affordance slots stay unplaced", model)
                return
            annotation.affordances = place_slots(annotation, points, triangles, bounding_box, recompute=self._recompute_affordances)
            logger.info("Placed the affordance slots of %s on %s", annotation.path, model)

        def carry_articulation(annotation: ObjectAnnotation) -> None:
            built = self.output_path / annotation.path / ANNOTATION_NAME
            try:
                previous = structure_articulation((yaml.safe_load(built.read_text()) or {}).get("articulation")) if built.is_file() else None
            except ValueError as e:
                logger.warning("%s drops the articulation of its last build, which does not parse (%s), run a model build", annotation.path, e)
                previous = None
            if annotation.articulation is not None and previous is None:
                logger.warning("%s has an articulation its last build lacks, run a model build to export its links", annotation.path)
            annotation.articulation = previous

        if self._overwrite == OverwriteMode.ANNOTATIONS_ONLY:
            self._pipeline.insert(0, carry_affordances)
            self._pipeline.insert(0, carry_articulation)

        self._previews_enabled = False
        self._max_texture = MAX_TEXTURE_PX
        self._recompute_affordances = False

    @classmethod
    def find_mainfile(cls, annotation: ObjectAnnotation) -> str | None:
        return next(
            (
                os.path.join(
                    os.path.relpath(walk[0], annotation.path),
                    f,
                )
                for walk in os.walk(annotation.path)
                for f in walk[2]
                if f.endswith(ModelConverter.exts()) and os.path.normpath(os.path.join(os.path.relpath(walk[0], annotation.path), f)) != os.path.normpath(annotation.collider or ".")
            ),
            None,
        )

    @classmethod
    def is_usd(cls, filename: str) -> bool:
        return os.path.splitext(filename.lower())[-1].lstrip(".") in {"usd", "usda", "usdc", "usdz"}

    def process_entity(self, annotation, dest):
        baked_usd_path: Path | None = None
        try:
            main_file = self.find_mainfile(annotation)
            if main_file is None:
                logger.warning("No supported model file found in %s", annotation.path)
                return None

            input_file = Path(annotation.path) / main_file
            link_bundles.clear(dest, annotation.name)

            if self._usd_baker is not None and self.is_usd(main_file):
                baked_usd_path = dest / "build" / f"{annotation.name}.fbx"
                baked_usd_path.parent.mkdir(exist_ok=True)

                self._usd_baker.convert(
                    str(input_file.relative_to(self.input_path)),
                    str(baked_usd_path.relative_to(self.output_path)),
                )
                logger.info("Baked USD model %s to %s", input_file, baked_usd_path)
                input_file = baked_usd_path

            textures = Path(tempfile.mkdtemp(prefix=".textures-", dir=dest))
            try:
                load_options = [{"merge_parent_xform": False}] if annotation.articulation is not None and self.is_usd(input_file.name) else []
                scan = self._blender.run([("load", str(input_file), *load_options), ("rectify",), ("bounding_box",), ("lights",), ("remove_lights",), ("emissive_materials",), *([("triangles",)] if annotation.affordances else [])], keep=True)
                try:
                    rectified = np.array(scan.values[1])
                    articulation = self._built_articulation(annotation, rectified)
                    bounding_box = BoundingBox(scan.values[2]).round(4)
                    extracted = object_lights([SourceLight(**{key: tuple(value) if isinstance(value, list) else value for key, value in light.items()}) for light in scan.values[3]], [SourceMaterial(name=material["name"], center=tuple(material["center"])) for material in scan.values[5]])
                    steps: list[tuple] = [("prepare_glow", [light.glow for light in annotation.lights or extracted if light.glow is not None])]
                    if annotation.affordances:
                        points, triangles = _mesh(scan.values[6])
                        annotation.affordances = place_slots(annotation, points, triangles, bounding_box, recompute=self._recompute_affordances)
                    if self._previews_enabled:
                        steps.append(("render_perspective", str(dest / f"{annotation.name}_thumb.png"), {"resolution": (512, 512), "theta": math.radians(annotation.face.angle + 30)}))
                        steps.append(("render_topdown", str(dest / f"{annotation.name}_topdown.png"), {"resolution": (512, 512)}))
                    if self._max_texture > 0:
                        steps.append(("limit_textures", self._max_texture, str(textures)))
                    steps.extend(("save", str(dest / f"{annotation.name}.{ext}")) for ext in sorted(self._formats))
                except Exception:
                    self._blender.run([])
                    raise
                result = self._blender.run(steps, keep=articulation is not None)
                link_boxes, link_runs = self._export_links(articulation, dest) if articulation is not None else ({}, [])
            finally:
                shutil.rmtree(textures, ignore_errors=True)
            if self._max_texture > 0:
                for limit in result.values[[step[0] for step in steps].index("limit_textures")]:
                    logger.info("Texture %s of %s: %dx%d to %dx%d", limit["name"], annotation.path, limit["width"], limit["height"], limit["new_width"], limit["new_height"])

            logger.info("Processed model %s", annotation.path)
            runs = [scan, result, *link_runs]
            for line in "".join(run.stdout for run in runs).splitlines():
                if line.startswith("Warning:"):
                    logger.warning("ModelConverter: %s", line)
                else:
                    logger.info("ModelConverter: %s", line)
            for line in "".join(run.stderr for run in runs).splitlines():
                logger.warning("ModelConverter: %s", line)

            annotation.bounding_box = bounding_box

            for usd in collider.usd_models(dest, annotation.name):
                collider.collide_visuals(usd)

            if annotation.collider:
                points, triangles = read_mesh(Path(annotation.path) / annotation.collider)
                baked = collider.bake(dest, annotation.name, points @ rectified[:3, :3].T + rectified[:3, 3], triangles)
                annotation.collider = f"{annotation.name}{collider.SUFFIX}"
                logger.info("Baked %d collider triangles of %s into %s", len(triangles), annotation.path, ", ".join(path.name for path in baked))

            if annotation.lights:
                logger.info("Keeping %d annotated lights of %s, skipped extracting %d source lights", len(annotation.lights), annotation.path, len(extracted))
            else:
                annotation.lights = extracted

            if articulation is not None:
                written = link_bundles.finish(dest, annotation.name, self._target_path(annotation).as_posix(), articulation, link_boxes)
                try:
                    annotation.articulation = attrs.evolve(articulation, swept=articulation.sweep(link_boxes))
                except ValueError as e:
                    raise ArticulationError(str(e)) from None
                logger.info("Exported %d links of %s with %s", len(link_boxes), annotation.path, ", ".join(path.name for path in written))

            return annotation

        except ArticulationError as e:
            logger.error("Cannot build %s: %s", annotation.path, e)
            return None

        except Exception as e:
            logger.error("Unexpected error processing entity %s: %s", annotation.path, repr(e))
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

    @staticmethod
    def _built_articulation(annotation: ObjectAnnotation, rectified: np.ndarray) -> Articulation | None:
        """The source articulation carried into the built frame."""
        if annotation.articulation is None:
            return None
        try:
            return annotation.articulation.transformed(rectified)
        except ValueError as e:
            raise ArticulationError(str(e)) from None

    def _export_links(self, articulation: Articulation, dest: Path) -> tuple[dict[str, LinkBox], list[Result]]:
        """Split the kept scene into the articulation's links and save each link with meshes as its own bundle, return the link boxes and the worker runs."""
        try:
            split = self._blender.run([("split_links", articulation.links, articulation.root), ("link_bounding_boxes",)], keep=True)
        except BlenderError as e:
            raise ArticulationError(f"its articulation does not fit the model: {str(e).splitlines()[0]}") from None
        try:
            boxes: dict[str, LinkBox] = {record["link"]: BoundingBox(record["bounding_box"]).round(4) for record in split.values[1]}
            for link in boxes:
                link_bundles.link_dir(dest, link).mkdir(parents=True, exist_ok=True)
        except Exception:
            self._blender.run([])
            raise
        saved = self._blender.run([("save_link", link, str(link_bundles.link_dir(dest, link) / f"{link}.{ext}")) for link in boxes for ext in link_bundles.link_formats(self._formats)])
        return boxes, [split, saved]

    enable = OptionRegistry()

    @enable.register("formats")
    def sdf(self, formats: str | None = None):
        self._formats = {fmt.lower() for fmt in (formats or "*").split(",")}
        logger.info("Enabling %s export option.", ",".join(sorted(self._formats)))

    def _resolve_formats(self) -> None:
        known = set(ModelConverter.exts())
        if "*" in self._formats:
            self._formats = set(known)
        for fmt in sorted(self._formats - known):
            logger.error("Unknown model format specified: %s", fmt)
        self._formats &= known

    @enable.register("procthor")
    def procthor(self):
        logger.info("Enabling Procthor export option.")
        self._procthor = {}
        self._pipeline.append(lambda annotation: self._procthor.update({annotation.path.replace(os.sep, "_"): annotation.as_procthor}))

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

    @enable.register("recompute-affordances")
    def recompute_affordances(self):
        self._recompute_affordances = True

    @enable.register("max-texture")
    def max_texture(self, pixels: str = str(MAX_TEXTURE_PX)):
        self._max_texture = int(pixels)
        logger.info("Fitting textures into %d px, 0 keeps them as they are.", self._max_texture)

    @enable.register("previews")
    def previews(self):
        logger.info("Enabling preview generation option.")
        self._previews_enabled = True
