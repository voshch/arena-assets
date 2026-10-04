"""ASA facet taxonomy: load asa.yaml and parse the key::path::value tag grammar."""

from __future__ import annotations

import re
from pathlib import Path

import attrs
import yaml

_SEGMENT = re.compile(r"^[a-z0-9_]+$")

_DEFAULT_SPEC_PATH = Path(__file__).parent / "asa.yaml"

_CLASS_DIR_NAMES = {"Object": "object", "Material": "material", "Human": "human"}


@attrs.define
class Facet:
    key: str
    applies: list[str]
    cardinality: str
    values: list[str] | dict[str, list[str]]
    recommend: bool = False
    open: bool = False
    embed: bool = False

    def applies_to(self, asset_class: str) -> bool:
        return asset_class in self.applies

    def values_for(self, asset_class: str) -> list[str]:
        if isinstance(self.values, dict):
            return self.values.get(asset_class, [])
        return self.values


@attrs.define
class Spec:
    version: int
    facets: dict[str, Facet]


def load_spec(path: Path | str | None = None) -> Spec:
    """Load the ASA spec. Defaults to the packaged asa.yaml."""
    spec_path = Path(path) if path is not None else _DEFAULT_SPEC_PATH
    with open(spec_path) as f:
        raw = yaml.safe_load(f)

    facets = {}
    for key, body in raw.get("facets", {}).items():
        facets[key] = Facet(
            key=key,
            applies=list(body.get("applies", [])),
            cardinality=body.get("cardinality", "multi"),
            values=body.get("values", []),
            recommend=bool(body.get("recommend", False)),
            open=bool(body.get("open", False)),
            embed=bool(body.get("embed", False)),
        )

    return Spec(version=raw.get("asa", 1), facets=facets)


def class_from_dirname(name: str) -> str | None:
    """Map an on-disk class directory name (Object/Material/Human) to its asset class."""
    return _CLASS_DIR_NAMES.get(name)


def parse_tag(tag: str) -> tuple[str, str] | None:
    """Split a tag into (facet_key, value). Returns None if the tag has no '::'."""
    if "::" not in tag:
        return None
    segments = tag.split("::")
    return "::".join(segments[:-1]), segments[-1]


def parse_tags(tags: list[str]) -> dict[str, list[str]]:
    """Group tags by facet key, preserving tag order within each facet."""
    grouped: dict[str, list[str]] = {}
    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        key, value = parsed
        grouped.setdefault(key, []).append(value)
    return grouped


def is_grammatical(tag: str) -> bool:
    """Check the key::path::value grammar: lowercase [a-z0-9_]+ segments, at least 2, none empty."""
    segments = tag.split("::")
    if len(segments) < 2:
        return False
    return all(_SEGMENT.match(segment) for segment in segments)
