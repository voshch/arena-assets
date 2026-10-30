"""Infinigen-Articulated sim-ready assets generated per category and seed by a local Infinigen install, built from their USD export."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.articulated import ASSET_TYPE, DOMAIN, build, slug, usd_articulation, write_articulation
from arena_assets.importers.gso import annotate
from arena_assets.importers.providers import Http, Provider
from arena_assets.utils.ModelConverter import environment

ENV = "ARENA_ASSETS_INFINIGEN"
REPOSITORY = "https://github.com/princeton-vl/infinigen"
INSTALL = f"git clone {REPOSITORY}.git && cd infinigen && uv venv --python 3.11 && INFINIGEN_MINIMAL_INSTALL=True uv pip install -e '.[sim]' && export {ENV}=$PWD/.venv/bin/python"
SEEDS = range(1001, 1011)
EXPORTER = "usd"
ORIGIN = "origin::infinigen"
LICENSE = "BSD-3-Clause"
AUTHORS = "Princeton Vision & Learning Lab (Infinigen)"
CITATION = "Joshi et al., Infinigen-Sim: Procedural Generation of Articulated Simulation Assets, arXiv:2505.10755, 2025"
CATEGORIES = "from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP\nprint('\\n'.join(OBJECT_CLASS_MAP))"
SPAWN = "import sys\nfrom pathlib import Path\nfrom infinigen.core.sim import sim_factory\nsim_factory.spawn_simready(name=sys.argv[1], seed=int(sys.argv[2]), exporter=sys.argv[3], export_dir=Path(sys.argv[4]), visual_only=True)"


def interpreter() -> str:
    """Python of the Infinigen install: ARENA_ASSETS_INFINIGEN, else the python3 on PATH when it imports Infinigen."""
    if configured := os.environ.get(ENV):
        return configured
    found = shutil.which("python3")
    if found is not None and subprocess.run([found, "-c", "import infinigen.core.sim"], capture_output=True, env=environment()).returncode == 0:
        return found
    raise RuntimeError(f"no Infinigen install, set {ENV} to the python of one or install it with: {INSTALL}")


def split(item: str) -> tuple[str, int]:
    """Category and seed of an item id `<category>/<seed>`."""
    category, _, seed = item.partition("/")
    if not category or not seed.isdigit():
        raise KeyError(f"no Infinigen item '{item}', items are <category>/<seed> such as door/1001, list the categories with: import object infinigen")
    return category, int(seed)


def repair(stage: Path) -> list[str]:
    """Split the indices of every mesh written with one face size per index into faces of that size, return the meshes repaired."""
    from pxr import Usd, UsdGeom

    opened = Usd.Stage.Open(str(stage))
    repaired = []
    for prim in opened.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = len(mesh.GetFaceVertexIndicesAttr().Get() or [])
        sizes = set(counts)
        if sum(counts) == indices or len(sizes) != 1 or indices % next(iter(sizes)):
            continue
        mesh.GetFaceVertexCountsAttr().Set([counts[0]] * (indices // counts[0]))
        repaired.append(str(prim.GetPath()))
    if repaired:
        opened.Save()
    return repaired


def object_name(item: str) -> str:
    category, seed = split(item)
    return f"infinigen_{slug(category)}_{seed}"


def link_name(prim: str) -> str:
    """Link of an Infinigen body prim, the fixed `world` body as `base`."""
    return "base" if prim == "world" else slug(prim)


class Infinigen(Importer):
    meta = IMPORTERS.meta("object", "infinigen")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(REPOSITORY)

    def run(self, *args: str) -> str:
        python = interpreter()
        self.cache.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([python, "-c", *args], cwd=self.cache, capture_output=True, text=True, env=environment())
        if result.returncode != 0:
            lines = [line for line in result.stderr.splitlines() if line.strip()]
            raise RuntimeError(f"Infinigen ({python}) failed: {lines[-1] if lines else f'exit code {result.returncode}'}")
        return result.stdout

    def items(self) -> Iterator[str]:
        return (f"{category}/{seed}" for category in self.run(CATEGORIES).split() for seed in SEEDS)

    def fetch(self, item: str) -> Path:
        """Generate one item with the Infinigen install, kept in the cache for the next import."""
        category, seed = split(item)
        export = self.cache / "exports"
        stage = export / EXPORTER / category / str(seed) / f"{category}.usda"
        if not stage.is_file():
            self.run(SPAWN, category, str(seed), EXPORTER, str(export))
        if not stage.is_file():
            raise RuntimeError(f"Infinigen generated no {stage.relative_to(self.cache)} for {item}")
        return stage

    def convert(self, item: str, src: Path, out: Path) -> str:
        return build(lambda dataset: self.stage(item, src, dataset), self.cache, out, object_name(item))

    def stage(self, item: str, src: Path, dataset: Path) -> Path:
        """Lay a generated export out as builder input under dataset, its joints as articulation, and return its directory."""
        category, _ = split(item)
        name = object_name(item)
        source = dataset / DOMAIN / ASSET_TYPE / name
        shutil.copytree(src.parent, source)
        repair(source / src.name)
        annotate(source / ANNOTATION_NAME, name, category.replace("_", " "), [ORIGIN, slug(category)], slug(category))
        write_articulation(source / ANNOTATION_NAME, usd_articulation(source / src.name, link_name))
        return source

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=REPOSITORY, license=LICENSE, authors=AUTHORS, citation=CITATION)
