"""Mechanical annotation.yaml rewrites for `arena-assets semantic lint --fix`."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from arena_assets.semantic import CURRENT_ASA
from arena_assets.semantic.taxonomy import Spec, class_from_dirname, parse_tag

_DOMAIN_BARE_TAGS = {"common", "hospital", "office", "residential", "warehouse"}

_ROOM_BARE_TAGS = {
    "bathroom": "bathroom",
    "bathrooms": "bathroom",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "livingroom": "living_room",
}
_REDUNDANT_BARE_TAGS = {"furniture", "appliance"}

_COLOR_MODIFIERS = {"light", "dark", "pale", "deep", "bright"}
_COLOR_DROP = {"colored", "tinted", "light", "dark"}
_COLOR_SYNONYMS = {"grey": "gray"}
_MATERIAL_SYNONYMS = {"porceplain": "porcelain", "pottery": "ceramic"}

_OBJECT_KIND_GROUPS: dict[str, list[str]] = {
    "chair": ["chair", "armchair", "recliner"],
    "stool": ["stool"],
    "bench": ["bench"],
    "sofa": ["sofa", "couch"],
    "bed": ["bed", "gurney"],
    "table": ["table", "desk"],
    "cabinet": ["cabinet", "cupboard", "closet", "closets", "dresser"],
    "shelf": ["shelf", "shelves", "bookcase", "rack"],
    "counter": ["counter", "reception", "information"],
    "box": ["box"],
    "container": ["crate", "barrel", "bottle", "container", "pallet", "tray"],
    "bin": ["trash", "trashcan", "disposal"],
    "appliance": ["oven", "stove", "microwave", "refrigerator", "refrigerated", "fridge", "dishwasher", "griddle", "fan", "cooler"],
    "equipment": ["machine", "autoclave", "defibrillator", "ventilator", "microscope", "station", "scale", "xraylightbox", "iv", "monitor", "laptop", "keyboard", "mouse", "pc", "tablet", "printer", "phone", "intercom", "tv", "wheelchair", "mop"],
    "dispenser": ["dispenser", "toweldispenser"],
    "sanitary": ["toilet", "urinal", "washbasin", "sink", "faucet", "dryer"],
    "safety": ["extinguisher", "fire", "aid", "biohazard"],
    "lamp": ["lamp"],
    "mirror": ["mirror"],
    "decor": ["picture", "painting", "collage", "poster", "frame", "clock", "calendar", "decor", "decorative", "mat"],
    "plant": ["plant", "tree"],
    "sign": ["sign", "board", "display", "chart"],
    "barrier": ["personenleitsystem", "cone"],
    "bag": ["bag", "briefcase"],
    "cart": ["cart", "trolley"],
    "cookware": ["pot", "pan"],
    "tableware": ["cup", "goblet"],
    "food": ["donut"],
    "curtain": ["curtain"],
    "structure": ["beam"],
}

_MATERIAL_KIND_GROUPS: dict[str, list[str]] = {
    "metal": ["aluminum", "aluminium", "brass", "bronze", "blued", "chrome", "chromium", "chain", "copper", "corrugatedmetal", "gold", "iron", "mercury", "metal", "nickel", "plate", "platinum", "rustedmetal", "silver", "solder", "stainless", "steel", "titanium", "tungsten", "zinc"],
    "gem": ["alexandrite", "amethyst", "ametrine", "aquamarine", "citrine", "diamond", "emerald", "garnet", "iolite", "jade", "morganite", "onyx", "pearl", "peridot", "ruby", "sapphire", "tanzanite", "topaz", "tourmaline", "turquoise", "zircon"],
    "wood": ["ash", "bamboo", "bark", "beadboard", "beech", "birch", "cherry", "cork", "laminate", "mahogany", "maple", "mdf", "oak", "osb", "parquet", "pine", "plywood", "poplar", "timber", "veneer", "walnut", "wood"],
    "fabric": ["carpet", "cashmere", "cloth", "cotton", "denim", "fabric", "felt", "linen", "polyester", "silk", "textile", "tweed", "velvet", "wool"],
    "leather": ["leather", "suede"],
    "glass": ["glass", "glazing"],
    "plastic": ["abs", "acrylic", "pet", "plastic", "polycarbonate", "polyethylene", "polymethylmethacrylate", "polypropylene", "polystyrene", "pvc", "styrofoam", "vinyl"],
    "composite": ["carbon", "fiberglass", "pcb"],
    "rubber": ["rubber"],
    "ceramic": ["ceramic", "clay", "grog", "porcelain", "terracotta"],
    "tile": ["tile", "shingles"],
    "brick": ["adobe", "brick"],
    "stone": ["cobblestone", "fieldstone", "granite", "marble", "mosaic", "paving", "sandstone", "slate", "stone", "terrazzo"],
    "concrete": ["cement", "concrete", "mortar", "retaining"],
    "plaster": ["gypsum", "plaster", "stucco"],
    "paint": ["paint"],
    "asphalt": ["asphalt"],
    "gravel": ["ballast", "gravel"],
    "ground": ["dirt", "grass", "ground", "leaves", "mulch", "sand", "soil"],
    "water": ["water"],
    "paper": ["cardboard", "paper"],
    "light": ["light"],
}


def _flatten(groups: dict[str, list[str]]) -> dict[str, str]:
    return {token: kind for kind, tokens in groups.items() for token in tokens}


_OBJECT_KIND_TABLE = _flatten(_OBJECT_KIND_GROUPS)
_MATERIAL_KIND_TABLE = _flatten(_MATERIAL_KIND_GROUPS)

# exact-name escape hatch for names whose leading token misleads
_OBJECT_KIND_OVERRIDES: dict[str, str] = {
    "Lab_Bench": "counter",
    "SM_ReceptionClosets": "cabinet",
    "SM_BedSideTable_01b": "table",
    "SM_CoffeeToGo": "tableware",
}

# granite trade names and metal tread plate
_MATERIAL_KIND_OVERRIDES: dict[str, str] = {
    "Devil_Black": "stone",
    "Volga_Blue": "stone",
    "Rosa_Beta": "stone",
    "Star_Galaxy": "stone",
    "Diamond_Plate_Double_Tear": "metal",
    "Diamond_Plate_Quadruple_Tear": "metal",
    "Diamond_Plate_Quintuple_Tear": "metal",
    "Diamond_Plate_Single_Tear": "metal",
    "Diamond_Plate_Spike": "metal",
    "Diamond_Plate_Triple_Tear": "metal",
}


def _name_tokens(name: str) -> list[str]:
    text = re.sub(r"[_\-]+", " ", name)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text)
    return [token.lower() for token in text.split() if token]


def _infer_kind(name: str, table: dict[str, str], overrides: dict[str, str]) -> str | None:
    """Exact-name override first, else the first name token (in name order) that hits the table, singular or a simple plural."""
    if name in overrides:
        return overrides[name]
    for token in _name_tokens(name):
        if token in table:
            return table[token]
        if token.endswith("s") and token[:-1] in table:
            return table[token[:-1]]
    return None


def _derive_domain_class(path: Path, storage_class_domain: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
    """Derive (domain, asset_class) from a <Domain>/<Class>/<name>/annotation.yaml path, falling back to the given override."""
    override_domain, override_class = storage_class_domain

    class_dir = path.parent.parent
    domain_dir = class_dir.parent

    asset_class = class_from_dirname(class_dir.name)
    domain = domain_dir.name.lower() if asset_class is not None else None

    return domain or override_domain, asset_class or override_class


def _sort_tags(tags: list[str], spec: Spec) -> list[str]:
    """Facet tags ordered by facet declaration order (stable on tag order), unknown/bare tags last."""
    facet_order = {key: index for index, key in enumerate(spec.facets)}
    known: list[tuple[int, str]] = []
    unknown: list[str] = []
    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is not None and parsed[0] in facet_order:
            known.append((facet_order[parsed[0]], tag))
        else:
            unknown.append(tag)
    known.sort(key=lambda pair: pair[0])
    return [tag for _, tag in known] + unknown


def _has_facet(tags: list[str], facet_key: str) -> bool:
    return any((parsed := parse_tag(tag)) is not None and parsed[0] == facet_key for tag in tags)


def _normalize_color(value: str) -> str | None:
    """Grammar-safe single-token color, modifier words dropped, None when the value carries no information."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(value).lower()) if w]
    if len(words) > 1:
        words = [w for w in words if w not in _COLOR_MODIFIERS]
    if not words:
        return None
    value = "_".join(words)
    value = _COLOR_SYNONYMS.get(value, value)
    return None if value in _COLOR_DROP else value


def _normalize_material(value: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", str(value).lower())
    if not words:
        return None
    value = "_".join(words)
    return _MATERIAL_SYNONYMS.get(value, value)


def _renormalize_facet_values(tags: list[str]) -> list[str]:
    """Normalize color/material tag values, dropping ones that carry no information."""
    normalizers = {"color": _normalize_color, "material": _normalize_material}
    result: list[str] = []
    for tag in tags:
        parsed = parse_tag(tag)
        if parsed is not None and parsed[0] in normalizers:
            value = normalizers[parsed[0]](parsed[1])
            if value is None:
                continue
            tag = f"{parsed[0]}::{value}"
        result.append(tag)
    return result


def fix_file(path: Path, spec: Spec, storage_class_domain: tuple[str | None, str | None] = (None, None)) -> bool:
    """Apply the mechanical ASA migration rules to a single annotation.yaml file. Returns whether it changed."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    original = copy.deepcopy(raw)
    domain, asset_class = _derive_domain_class(path, storage_class_domain)

    tags_raw = raw.get("tags")
    if isinstance(tags_raw, list):
        tags = [str(tag) for tag in tags_raw]
    elif tags_raw is None:
        tags = []
    else:
        tags = [str(tags_raw)]

    # bare tags with a facet equivalent
    rewritten: list[str] = []
    for tag in tags:
        if "::" in tag:
            rewritten.append(tag)
        elif tag in _DOMAIN_BARE_TAGS:
            rewritten.append(f"domain::{tag}")
        elif tag in _ROOM_BARE_TAGS:
            rewritten.append(f"room::{_ROOM_BARE_TAGS[tag]}")
        elif tag == "pedestrian" or tag in _REDUNDANT_BARE_TAGS:
            continue
        elif tag == "makehuman":
            rewritten.append("origin::makehuman")
        else:
            rewritten.append(tag)
    tags = _renormalize_facet_values(rewritten)

    # domain from storage path
    if not _has_facet(tags, "domain") and domain:
        tags.append(f"domain::{domain}")

    # kind inference from the asset name
    if not _has_facet(tags, "kind"):
        name = str(raw.get("name", ""))
        lookup = {
            "object": (_OBJECT_KIND_TABLE, _OBJECT_KIND_OVERRIDES),
            "material": (_MATERIAL_KIND_TABLE, _MATERIAL_KIND_OVERRIDES),
        }.get(asset_class)
        kind = _infer_kind(name, *lookup) if lookup is not None else None
        if kind is not None:
            tags.append(f"kind::{kind}")

    tags = _sort_tags(tags, spec)
    raw["tags"] = tags
    raw["asa"] = CURRENT_ASA

    if raw == original:
        return False

    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=True, default_flow_style=False, allow_unicode=True)
    return True
