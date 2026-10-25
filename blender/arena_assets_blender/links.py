from __future__ import annotations

import dataclasses
import re

DUPLICATE_SUFFIX = re.compile(r"\.\d{3,}$")

Side = tuple[float, float]


def base_name(name: str) -> str:
    """Object name without Blender's duplicate suffix such as .001."""
    return DUPLICATE_SUFFIX.sub("", name)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinkObjects:
    """Mesh objects an articulation link takes, by object name."""

    link: str
    objects: list[str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinkBox:
    """World-space bounding box of an articulation link's meshes."""

    link: str
    bounding_box: tuple[Side, Side, Side]
