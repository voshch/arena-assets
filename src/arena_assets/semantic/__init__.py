"""ASA (Arena Semantic Annotations): tag grammar, facets, and their text/metadata projections.

Kept import-light (stdlib + attrs + yaml only) so `arena_assets.impl` can depend on it
unconditionally, without pulling in cattrs or chromadb.
"""

from __future__ import annotations

from arena_assets.semantic.taxonomy import Facet, Spec, load_spec, parse_tag, parse_tags

CURRENT_ASA = 1

_RESERVED_METADATA_COLUMNS = {
    "name",
    "path",
    "desc",
    "tags",
    "note",
    "face",
    "bounding_box",
    "width",
    "depth",
    "height",
    "volume",
    "asa",
}

_spec = load_spec()


def words(tags: list[str]) -> str:
    """Project tags to embeddable text: known embed-false facets are dropped, everything else keeps its value."""
    seen: list[str] = []
    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is None:
            word = tag
        else:
            key, value = parsed
            facet = _spec.facets.get(key)
            if facet is not None and not facet.embed:
                continue
            word = value
        if word not in seen:
            seen.append(word)
    return " ".join(seen)


def metadata_columns(tags: list[str]) -> dict[str, str]:
    """Project tags to one metadata column per facet (key::path -> key.path), comma-joined values."""
    columns: dict[str, list[str]] = {}
    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        key, value = parsed
        column = key.replace("::", ".")
        if column in _RESERVED_METADATA_COLUMNS:
            continue
        columns.setdefault(column, []).append(value)
    return {column: ",".join(values) for column, values in columns.items()}


__all__ = [
    "CURRENT_ASA",
    "Facet",
    "Spec",
    "load_spec",
    "metadata_columns",
    "parse_tag",
    "parse_tags",
    "words",
]
