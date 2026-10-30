"""ArtVIP articulated objects from Hugging Face, their USD stages built into Common/Object items with the UsdPhysics joints as articulation."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.articulated import ASSET_TYPE, DOMAIN, build, slug, usd_articulation, write_articulation
from arena_assets.importers.gso import annotate
from arena_assets.importers.providers import HFHub, Provider
from arena_assets.importers.simready import MAX_FOOTPRINT_M, footprint

REPO = "X-Humanoid/ArtVIP"
REVISION = "d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e"
OBJECTS = "Articulated_objects"
INDEX = "paths.json"
ORIGIN = "origin::artvip"
LICENSE = "Apache-2.0"
AUTHORS = "X-Humanoid (Beijing Innovation Center of Humanoid Robotics)"
CITATION = "Jin et al., ArtVIP: Articulated Digital Assets of Visual Realism, Modular Interaction, and Physical Fidelity for Robot Learning, ICLR 2026"

_MODEL = re.compile(r"model_[^/]*\.usd[ac]?")
_ELEMENT = re.compile(r"^E_")


def link_name(prim: str) -> str:
    """Link of an ArtVIP body prim, `E_door_3` as `door_3`."""
    return slug(_ELEMENT.sub("", prim))


def model_paths(files: Iterable[str]) -> dict[str, str]:
    """Repository path of the model stage of every object folder, keyed by the item id: the folder name in snake case."""
    found: dict[str, str] = {}
    for path in files:
        folder, _, name = path.rpartition("/")
        if "/.thumbs" in folder or folder.endswith("/origin") or not _MODEL.fullmatch(name):
            continue
        item = slug(folder.rpartition("/")[2])
        if item in found and found[item] != path:
            raise ValueError(f"{found[item]} and {path} both name the ArtVIP item {item}")
        found[item] = path
    return dict(sorted(found.items()))


def needed(stage: Path, dest: Path, listed: set[str]) -> set[str]:
    """Listed files the composed stage refers to that are not on disk, a search-path reference such as `img/a.png` matched by its ending."""
    from pxr import UsdUtils

    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(stage))
    root = dest.resolve()
    found: set[str] = set()
    for path in unresolved:
        if Path(path).is_absolute():
            resolved = Path(path).resolve()
            if resolved.is_relative_to(root):
                found.add(resolved.relative_to(root).as_posix())
        else:
            found.update(name for name in listed if name == path or name.endswith(f"/{path}"))
    return {name for name in found if name in listed and not (dest / name).is_file()}


class ArtVIP(Importer):
    meta = IMPORTERS.meta("object", "artvip")

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO, revision=REVISION)

    def paths(self) -> dict[str, str]:
        """Model stage path of every item, listed once into the cache."""
        index = self.cache / INDEX
        if not index.is_file():
            listed = model_paths(self.provider.list(OBJECTS))
            index.parent.mkdir(parents=True, exist_ok=True)
            part = index.with_name(index.name + ".part")
            part.write_text(json.dumps(listed))
            part.replace(index)
        return json.loads(index.read_text())

    def items(self) -> Iterator[str]:
        return iter(self.paths())

    def fetch(self, item: str) -> Path:
        """Download an item's model stage, then every layer and texture the composed stage refers to, until none is missing."""
        paths = self.paths()
        if item not in paths:
            raise KeyError(f"no ArtVIP object '{item}' in {REPO}, list the items with: import object {self.meta.name}")
        folder, _, name = paths[item].rpartition("/")
        listed = {path.removeprefix(f"{folder}/") for path in self.provider.list(folder) if "/.thumbs/" not in path}
        dest = self.cache / "objects" / item
        stage = self.provider.get(paths[item], dest / name)
        while wanted := needed(stage, dest, listed):
            for relative in sorted(wanted):
                self.provider.get(f"{folder}/{relative}", dest / relative)
        return stage

    def convert(self, item: str, src: Path, out: Path) -> str:
        size = footprint(src)
        if max(size[0], size[1]) > MAX_FOOTPRINT_M:
            raise ValueError(f"{item} is {size[0]:.2f} x {size[1]:.2f} m, the object builder rescales a footprint over {MAX_FOOTPRINT_M:g} m")
        try:
            return build(lambda dataset: self.stage(item, src, dataset), self.cache, out, item)
        finally:
            shutil.rmtree(src.parent, ignore_errors=True)

    def stage(self, item: str, src: Path, dataset: Path) -> Path:
        """Lay a downloaded object out as builder input under dataset, its joints as articulation, and return its directory."""
        source = dataset / DOMAIN / ASSET_TYPE / item
        shutil.copytree(src.parent, source)
        *parents, folder, _ = self.paths()[item].split("/")[1:]
        category = slug(parents[-1]) if len(parents) > 1 else None
        annotate(source / ANNOTATION_NAME, item, " ".join(folder.replace("_", " ").split()), [ORIGIN, *([category] if category else [])], category)
        write_articulation(source / ANNOTATION_NAME, usd_articulation(source / src.name, link_name))
        return source

    def attribution(self, item: str) -> Attribution:
        folder = self.paths()[item].rpartition("/")[0]
        return Attribution(source=self.meta.name, url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{folder}", license=LICENSE, authors=AUTHORS, citation=CITATION)
