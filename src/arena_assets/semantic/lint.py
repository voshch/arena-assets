"""Lint engine for ASA annotation.yaml files. Works on raw parsed YAML, not on Annotation classes."""

from __future__ import annotations

import difflib
from pathlib import Path

import attrs
import yaml

from arena_assets.semantic.taxonomy import Spec, class_from_dirname, is_grammatical, parse_tag


@attrs.define
class Finding:
    file: Path
    severity: str
    code: str
    message: str
    fixable: bool = False


def _coerce_tags(raw: dict) -> list[str]:
    tags = raw.get("tags")
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(tag) for tag in tags]
    return [str(tags)]


def _infer_asset_class(raw: dict, file: Path) -> str | None:
    path_field = raw.get("path")
    if path_field:
        first_segment = str(path_field).replace("\\", "/").split("/")[0]
        asset_class = class_from_dirname(first_segment)
        if asset_class is not None:
            return asset_class
    return class_from_dirname(file.parent.parent.name)


def lint_file(path: Path, spec: Spec) -> list[Finding]:
    """Lint a single annotation.yaml file against the ASA spec."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    findings: list[Finding] = []
    asset_class = _infer_asset_class(raw, path)
    tags = _coerce_tags(raw)

    seen_facets: dict[str, list[str]] = {}

    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is None:
            findings.append(Finding(path, "info", "bare-tag", f"bare tag {tag!r}", fixable=True))
            continue

        if not is_grammatical(tag):
            findings.append(Finding(path, "error", "malformed-tag", f"malformed tag grammar: {tag!r}"))
            continue

        key, value = parsed
        seen_facets.setdefault(key, []).append(value)

        facet = spec.facets.get(key)
        if facet is None:
            findings.append(Finding(path, "warning", "unknown-facet", f"unknown facet {key!r} in tag {tag!r}"))
            continue

        if asset_class is not None and not facet.applies_to(asset_class):
            findings.append(Finding(path, "warning", "facet-not-applicable", f"facet {key!r} does not apply to {asset_class} (tag {tag!r})"))

        allowed = facet.values_for(asset_class) if asset_class is not None else (facet.values if isinstance(facet.values, list) else [])
        if allowed and value not in allowed:
            if facet.open:
                suggestion = difflib.get_close_matches(value, allowed, n=1)
                hint = f", did you mean {suggestion[0]!r}?" if suggestion else ""
                findings.append(Finding(path, "info", "novel-value", f"novel value {value!r} for facet {key!r}{hint}"))
            else:
                findings.append(Finding(path, "warning", "closed-vocab-violation", f"value {value!r} not in closed vocabulary for facet {key!r}: {allowed}"))

    for key, values in seen_facets.items():
        facet = spec.facets.get(key)
        if facet is not None and facet.cardinality == "single" and len(values) > 1:
            findings.append(Finding(path, "warning", "cardinality-violation", f"facet {key!r} is single-valued but has {len(values)} values: {values}"))

    if asset_class is not None:
        for key, facet in spec.facets.items():
            if facet.recommend and facet.applies_to(asset_class) and key not in seen_facets:
                findings.append(Finding(path, "info", "missing-recommended-facet", f"missing recommended facet {key!r} for {asset_class}", fixable=False))

    if "asa" not in raw:
        findings.append(Finding(path, "info", "missing-asa-stamp", "missing asa version stamp", fixable=True))

    return findings


def lint_tree(root: Path, spec: Spec) -> list[Finding]:
    """Lint every annotation.yaml file under root."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("annotation.yaml")):
        findings.extend(lint_file(path, spec))
    return findings
