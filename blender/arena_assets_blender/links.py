from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping

DUPLICATE_SUFFIX = re.compile(r"\.\d{3,}$")

Side = tuple[float, float]


def base_name(name: str) -> str:
    """Object name without Blender's duplicate suffix such as .001."""
    return DUPLICATE_SUFFIX.sub("", name)


def node_links(links: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """Link of every articulation node, by node name without a duplicate suffix."""
    return {base_name(node): link for link, nodes in links.items() for node in nodes}


def link_of(lineage: Iterable[str], owners: Mapping[str, str], root: str) -> str:
    """Link naming an object or its nearest listed ancestor, root for the rest."""
    return next((owners[base] for name in lineage if (base := base_name(name)) in owners), root)


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
