from __future__ import annotations

import math
import typing

import attrs

Vector3 = tuple[float, float, float]

DEFAULT_CCT_K = 6500.0
MIN_CCT_K = 1000.0
MAX_CCT_K = 40000.0

_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_MCCAMY_EPICENTER = (0.3320, 0.1858)


@attrs.frozen(kw_only=True)
class SourceLight:
    """Light found in a source model scene, in world coordinates of the rectified model."""

    name: str
    type: str
    shape: str = ""
    spot_size: float = 0.0
    power: float
    color: Vector3
    location: Vector3
    direction: Vector3


@attrs.frozen(kw_only=True)
class SourceMaterial:
    """Emissive material found in a source model scene, with the world-space center of its faces."""

    name: str
    center: Vector3


def nearest_material(location: Vector3, materials: typing.Iterable[SourceMaterial]) -> str | None:
    """Name of the material centered nearest to a location, the first by name on a tie."""
    nearest = min(materials, key=lambda material: (math.dist(location, material.center), material.name), default=None)
    return None if nearest is None else nearest.name


def color_temperature(color: typing.Sequence[float]) -> float:
    """Correlated color temperature in K of a linear sRGB color, by McCamy's approximation."""
    x_, y_, z_ = (sum(weight * channel for weight, channel in zip(row, color, strict=True)) for row in _SRGB_TO_XYZ)
    total = x_ + y_ + z_
    if total <= 0.0:
        return DEFAULT_CCT_K
    x, y = x_ / total, y_ / total
    if y == _MCCAMY_EPICENTER[1]:
        return DEFAULT_CCT_K
    n = (x - _MCCAMY_EPICENTER[0]) / (_MCCAMY_EPICENTER[1] - y)
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
    return round(min(max(cct, MIN_CCT_K), MAX_CCT_K), -1)
