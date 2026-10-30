"""PartNet-Mobility articulated objects from the gated Hugging Face copy, each URDF staged as one OBJ whose objects name its links."""

from __future__ import annotations

import json
import re
import urllib.error
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.articulated import ASSET_TYPE, DOMAIN, blender_name, build, merge_obj, read_urdf, refused, slug, unique, urdf_articulation, write_articulation
from arena_assets.importers.gso import TEXTURES, annotate
from arena_assets.importers.providers import HFHub, Provider

REPO = "sapien-sim/PartNetMobility"
REVISION = "ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f"
SOURCE = "PartNet-Mobility"
URDF = "mobility.urdf"
META = "meta.json"
SEMANTICS = "semantics.txt"
UPSTREAM = "upstream"
ORIGIN = "origin::partnet_mobility"
LICENSE = "PartNet-Mobility terms of use: non-commercial research and education, models derived from ShapeNet"
AUTHORS = "SAPIEN team (UC San Diego), part annotations from PartNet, models from ShapeNet"
CITATION = "Xiang et al., SAPIEN: A SimulAted Part-based Interactive ENvironment, CVPR 2020"

SIZES: dict[str, tuple[str, float]] = {
    "Bottle": ("height", 0.25),
    "Box": ("size", 0.4),
    "Bucket": ("height", 0.3),
    "Camera": ("size", 0.13),
    "Cart": ("height", 0.95),
    "Chair": ("height", 0.95),
    "Clock": ("size", 0.3),
    "CoffeeMachine": ("height", 0.35),
    "Dishwasher": ("height", 0.85),
    "Dispenser": ("height", 0.2),
    "Display": ("size", 0.6),
    "Door": ("height", 2.05),
    "Eyeglasses": ("size", 0.14),
    "Fan": ("size", 0.5),
    "Faucet": ("height", 0.3),
    "FoldingChair": ("height", 0.85),
    "Globe": ("height", 0.45),
    "Kettle": ("height", 0.25),
    "Keyboard": ("size", 0.45),
    "KitchenPot": ("size", 0.3),
    "Knife": ("size", 0.2),
    "Lamp": ("height", 0.5),
    "Laptop": ("size", 0.35),
    "Lighter": ("size", 0.08),
    "Microwave": ("size", 0.5),
    "Mouse": ("size", 0.11),
    "Oven": ("size", 0.8),
    "Pen": ("size", 0.14),
    "Phone": ("size", 0.15),
    "Pliers": ("size", 0.2),
    "Printer": ("size", 0.45),
    "Refrigerator": ("height", 1.75),
    "Remote": ("size", 0.2),
    "Safe": ("height", 0.5),
    "Scissors": ("size", 0.2),
    "Stapler": ("size", 0.18),
    "StorageFurniture": ("depth", 0.5),
    "Suitcase": ("height", 0.65),
    "Switch": ("size", 0.1),
    "Table": ("height", 0.75),
    "Toaster": ("size", 0.3),
    "Toilet": ("height", 0.75),
    "TrashCan": ("height", 0.5),
    "USB": ("size", 0.06),
    "WashingMachine": ("height", 0.85),
    "Window": ("height", 1.2),
}
LONGEST: dict[str, float] = {"Table": 2.0}

_ITEM = re.compile(r"\d+")
_WORD_START = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def object_name(item: str) -> str:
    return f"partnet_mobility_{item}"


def semantics(path: Path) -> dict[str, str]:
    """Semantic label of each link from a semantics.txt of `<link> <joint kind> <label>` lines."""
    if not path.is_file():
        return {}
    return {words[0]: words[-1] for words in (line.split() for line in path.read_text().splitlines()) if len(words) >= 2}


def extent(parts: Sequence[tuple[str, Path, np.ndarray]]) -> np.ndarray:
    """Size along x, y and z of OBJ parts placed by their transforms."""
    meshes: dict[Path, np.ndarray] = {}
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for _, mesh, transform in parts:
        if mesh not in meshes:
            meshes[mesh] = np.array([[float(word) for word in line.split()[1:4]] for line in mesh.read_text(errors="replace").splitlines() if line.startswith("v ")]).reshape(-1, 3)
        points = meshes[mesh] @ transform[:3, :3].T + transform[:3, 3]
        if len(points):
            low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
    return np.maximum(high - low, 0.0) if np.isfinite(low).all() else np.zeros(3)


def metric_scale(category: str, size: np.ndarray) -> float:
    """Scale giving a unit-normalized model its category's typical height, depth (shorter horizontal side) or size (longest side) in meters, its longest side capped where LONGEST names one, 1 for an unknown category."""
    if category not in SIZES:
        return 1.0
    measure, meters = SIZES[category]
    value = {"height": size[2], "depth": min(size[0], size[1]), "size": max(size)}[measure]
    if value <= 0.0:
        return 1.0
    return min(meters / value, LONGEST[category] / max(size)) if category in LONGEST else meters / value


def stage(item: str, src: Path, dataset: Path) -> Path:
    """Lay a PartNet-Mobility zip out as builder input under dataset: its link meshes as one OBJ in meters, its textures, its joints as articulation."""
    name = object_name(item)
    source = dataset / DOMAIN / ASSET_TYPE / name
    upstream = source / UPSTREAM
    with zipfile.ZipFile(src) as archive:
        archive.extractall(upstream)
    urdf_path = next(iter(sorted(upstream.rglob(URDF))), None)
    if urdf_path is None:
        raise ValueError(f"{src.name} holds no {URDF}")
    folder = urdf_path.parent
    urdf = read_urdf(urdf_path)
    taken: set[str] = set()
    nodes = {link: [blender_name(unique(slug(link), taken))] for link in dict.fromkeys(visual.link for visual in urdf.visuals)}
    parts = [(nodes[visual.link][0], folder / visual.mesh, visual.transform) for visual in urdf.visuals]
    category = json.loads((folder / META).read_text()).get("model_cat", "") if (folder / META).is_file() else ""
    factor = metric_scale(category, extent(parts))
    scale = np.diag([factor, factor, factor, 1.0])
    merge_obj([(node, mesh, scale @ transform) for node, mesh, transform in parts], source / f"{name}.obj")
    labels = semantics(folder / SEMANTICS)
    for path in sorted(upstream.rglob("*")):
        if path.is_file() and path.suffix.lower() not in TEXTURES:
            path.unlink()
    kind = slug(category) if category else None
    annotate(source / ANNOTATION_NAME, name, _WORD_START.sub(" ", category).lower() or f"{SOURCE} object {item}", [ORIGIN, *([kind] if kind else [])], kind)
    articulation = urdf_articulation(urdf, nodes, labels, lower_closed=True)
    write_articulation(source / ANNOTATION_NAME, None if articulation is None else articulation.transformed(scale))
    return source


class PartNetMobility(Importer):
    meta = IMPORTERS.meta("object", "partnet_mobility")

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO, revision=REVISION)

    def items(self) -> Iterator[str]:
        try:
            names = sorted((path.removesuffix(".zip") for path in self.provider.list() if path.endswith(".zip") and _ITEM.fullmatch(path.removesuffix(".zip"))), key=int)
        except urllib.error.HTTPError as error:
            raise refused(error, REPO, SOURCE) from None
        return iter(names)

    def fetch(self, item: str) -> Path:
        if not _ITEM.fullmatch(item):
            raise KeyError(f"no {SOURCE} object '{item}', items are model ids such as 100013, list them with: import object {self.meta.name}")
        try:
            return self.provider.get(f"{item}.zip", self.cache / f"{item}.zip")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise KeyError(f"no {SOURCE} object '{item}' in {REPO}, list the items with: import object {self.meta.name}") from None
            raise refused(error, REPO, SOURCE) from None

    def convert(self, item: str, src: Path, out: Path) -> str:
        return build(lambda dataset: stage(item, src, dataset), self.cache, out, object_name(item))

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/{item}.zip", license=LICENSE, authors=AUTHORS, citation=CITATION)
