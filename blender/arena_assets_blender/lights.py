from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceLight:
    """Light found in a source model scene, in world coordinates of the rectified model."""

    name: str
    type: str
    shape: str = ""
    spot_size: float = 0.0
    power: float
    color: Vector3
    location: Vector3
    rotation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    """Unit quaternion (w, x, y, z) of the light, which emits along its local -z axis."""
    radiance: float | None = None
    """Emitted luminance of a UsdLux light in its stage's units, None without UsdLux photometry."""
    area: float = 0.0
    """World-space emitting area in m^2."""
    face_area: float = 0.0
    """World-space emitting area seen along the emission axis in m^2."""
    cct_K: float | None = None
    link: str = ""
    """Articulation link the light sits on, empty for an object without articulation."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class UsdLight:
    """UsdLux light prim of a source stage, sizes in its local units."""

    names: tuple[str, ...]
    """Prim path components, root first."""
    kind: str
    width: float = 0.0
    height: float = 0.0
    radius: float = 0.0
    intensity: float
    exposure: float = 0.0
    normalize: bool = False
    cct_K: float | None = None
    cone_deg: float | None = None
    """Full shaping cone, None for an open hemisphere."""


def _in_order(sequence: Iterable[str], items: Iterable[str]) -> bool:
    remaining = iter(sequence)
    return all(item in remaining for item in items)


def matching_prim(lineage: Sequence[str], prims: Iterable[UsdLight]) -> UsdLight | None:
    """The prim a light object was imported from by its lineage (names nearest first): same name, the lineage among its ancestors in order, the fewest ancestors left out first."""
    candidates = [prim for prim in prims if prim.names[-1] == lineage[0] and _in_order(reversed(prim.names[:-1]), lineage[1:])]
    return min(candidates, key=lambda prim: len(prim.names), default=None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceMaterial:
    """Emissive material found in a source model scene, with the world-space center of its faces."""

    name: str
    center: Vector3
