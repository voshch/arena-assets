"""Implementation modules for arena_assets CLI commands."""

import contextlib
import contextvars
import enum
import json
import re
import typing

import attrs
import cattrs.preconf.pyyaml

from arena_assets import semantic
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.logging import get_logger

logger = get_logger("arena_assets.impl")

converter = cattrs.preconf.pyyaml.make_converter(prefer_attrib_converters=True)

_active_serialization_asset: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_serialization_asset",
    default=None,
)


@contextlib.contextmanager
def serialization_asset_context(asset: str):
    token = _active_serialization_asset.set(asset)
    try:
        yield
    finally:
        _active_serialization_asset.reset(token)


converter.register_structure_hook(BoundingBox, lambda d, t: d if isinstance(d, BoundingBox) else BoundingBox(d))


def _unstructure_bounding_box(box: BoundingBox) -> list:
    if (volume := box.volume) < 1e-9:
        asset = _active_serialization_asset.get()
        if volume < 0:
            logger.warning(
                "Serializing Negative-volume (%d) BoundingBox for %s: %s",
                volume,
                asset or "<unknown-asset>",
                box,
            )
        else:
            logger.warning(
                "Serializing Zero-volume (%d) BoundingBox for %s: %s",
                volume,
                asset or "<unknown-asset>",
                box,
            )
    return list(box)


converter.register_unstructure_hook(BoundingBox, _unstructure_bounding_box)


class AssetType(enum.Enum):
    OBJECT = "object"
    MATERIAL = "material"
    HUMAN = "human"
    SOUND = "sound"


DATABASE_NAME = ".db"
ANNOTATION_NAME = "annotation.yaml"


def convert_list_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


@attrs.define
class Annotation:
    name: str
    path: str
    desc: str = ""
    tags: list[str] = attrs.field(factory=list, converter=convert_list_str)
    asa: int = 0

    @property
    def as_text(self) -> str:
        sections = [self.name_text, semantic.words(self.tags), self.desc]
        return ". ".join(section for section in sections if section)

    @property
    def name_text(self) -> str:
        """Name normalized for embedding: words split, trailing instance counters dropped."""
        text = re.sub(r"[_\-]+", " ", self.name)
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        return re.sub(r"(?:\s+\d+)+$", "", text)

    @property
    def facets(self) -> dict[str, list[str]]:
        """Tags grouped by ASA facet key, see arena_assets.semantic."""
        return semantic.parse_tags(self.tags)

    @property
    def as_metadata(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "desc": self.desc,
            "tags": ",".join(self.tags),
            "asa": self.asa,
            **semantic.metadata_columns(self.tags),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        return cls(
            name=metadata["name"],
            path=metadata["path"],
            desc=metadata.get("desc", ""),
            tags=tags.split(",") if (tags := metadata.get("tags")) else [],
            asa=int(metadata.get("asa", 0)),
        )


class Face(enum.StrEnum):
    POS_X = "+x"
    NEG_X = "-x"
    POS_Y = "+y"
    NEG_Y = "-y"
    XY = "xy"

    @property
    def angle(self) -> float:
        return {
            Face.NEG_X: 0.0,
            Face.NEG_Y: +90.0,
            Face.POS_X: -180.0,
            Face.XY: -135.0,
            Face.POS_Y: -90.0,
        }[self]


@attrs.define
class SpatialAnnotation(Annotation):
    bounding_box: BoundingBox = attrs.field(factory=BoundingBox.empty)
    face: Face = attrs.field(default=Face.NEG_Y)
    note: str = attrs.field(default="")

    @classmethod
    def _parse_face(cls, value: str | float | Face | None) -> Face:
        if value is None or value == "":
            return Face.NEG_Y

        if isinstance(value, Face):
            return value

        # Preferred format: enum value strings like "-y", "+x", "xy"
        if isinstance(value, str):
            try:
                return Face(value)
            except ValueError:
                pass

            # Backward-compatible: some databases stored numeric angles as strings
            try:
                value = float(value)
            except ValueError:
                return Face.NEG_Y

        # Backward-compatible: numeric angle values (legacy)
        if isinstance(value, (int, float)):
            angle_to_face = {
                0.0: Face.NEG_X,
                90.0: Face.NEG_Y,
                -180.0: Face.POS_X,
                -135.0: Face.XY,
                -90.0: Face.POS_Y,
            }
            return angle_to_face.get(float(value), Face.NEG_Y)

        return Face.NEG_Y

    @property
    def as_metadata(self) -> dict:
        return {
            **super().as_metadata,
            "face": self.face.value,
            "bounding_box": json.dumps(list(self.bounding_box)),
            "width": self.bounding_box.max_x - self.bounding_box.min_x,
            "depth": self.bounding_box.max_y - self.bounding_box.min_y,
            "height": self.bounding_box.max_z - self.bounding_box.min_z,
            "volume": self.bounding_box.volume,
            "note": self.note,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> typing.Self:
        return cls(
            name=metadata.get("name", ""),
            path=metadata.get("path", ""),
            desc=metadata.get("desc", ""),
            tags=tags.split(",") if (tags := metadata.get("tags")) else [],
            asa=int(metadata.get("asa", 0)),
            bounding_box=BoundingBox(json.loads(bounding_box)) if (bounding_box := metadata.get("bounding_box")) else BoundingBox.empty(),
            face=cls._parse_face(metadata.get("face")),
            note=metadata.get("note", ""),
        )
