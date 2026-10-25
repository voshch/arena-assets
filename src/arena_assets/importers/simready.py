"""NVIDIA SimReady assets from Hugging Face, built into Common/Object items with their joints as articulation and their SimReady metadata and physics payloads kept beside."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode, replacing
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.articulated import slug, usd_articulation, write_articulation
from arena_assets.importers.gso import annotate
from arena_assets.importers.providers import HFHub, Provider
from arena_assets.utils.logging import get_logger

DOMAIN = "Common"
ASSET_TYPE = "Object"
FORMATS = "usdz,sdf,fbx"
ORIGIN = "origin::simready"
LICENSE = "CC-BY-4.0"
AUTHORS = "NVIDIA Corporation"
PACKAGE = "com.nvidia.simready.packaging.json"
ROOTS = ".metadata/com.nvidia.simready.root_usds.json"
LAYERS = (".usd", ".usda", ".usdc")
PHYSICS = ("payloads/Physics", "payloads/PhysX", "payloads/MuJoCo", "payloads/Newton")
KEPT = "simready"
MAX_FOOTPRINT_M = 5.0

_WORD_START = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_VARIANT = re.compile(r"^[a-z]\d{2}$")
_BODY = re.compile(r"_obj_\d+$")
_JOINT = re.compile(r"^joint_(.+)_joint_\d+$")

logger = get_logger("import.simready")


def description(item: str) -> str:
    """Words of an asset folder name without its variant suffix, `bed_hospital_standard_a01` as `bed hospital standard`."""
    words = [word for part in item.split("_") for word in _WORD_START.sub(" ", part).split()]
    return " ".join(word.lower() if not word.isupper() else word for word in words if not _VARIANT.match(word))


def kind(item: str) -> str:
    return _WORD_START.sub("_", item.split("_")[0]).lower()


def link_name(prim: str) -> str:
    """Link of a SimReady body prim, `wheelPin_01_obj_00` as `wheel_pin_01`."""
    return slug(_BODY.sub("", prim))


def joint_name(prim: str) -> str:
    """Joint of a SimReady joint prim, `joint_wheelPin_01_joint_00` as `wheel_pin_01`."""
    match = _JOINT.match(prim)
    return slug(match[1] if match else prim)


def footprint(root: Path) -> tuple[float, float, float]:
    """Size of a USD stage's default and render geometry in meters."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(root))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    size = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange().GetSize()
    return size[0] * scale, size[1] * scale, size[2] * scale


def missing(root: Path) -> list[Path]:
    """Files the composed stage of root refers to that are not on disk."""
    from pxr import UsdUtils

    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(root))
    return sorted({Path(path).resolve() for path in unresolved})


class SimReady(Importer):
    meta = IMPORTERS.meta("object", "simready")
    REPO = "nvidia/simready-assets"
    REVISION = "524747d9b5e1f21b5b18b56041788c01d745f1a1"

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(cls.REPO, revision=cls.REVISION)

    def items(self) -> Iterator[str]:
        assert isinstance(self.provider, HFHub)
        return iter(sorted(self.provider.directories()))

    def package(self, item: str) -> Path:
        return self.cache / "packages" / item

    def fetch(self, item: str) -> Path:
        """Download a package's metadata and USD layers, then the textures and materials its stage refers to."""
        files = sorted(path for path in self.provider.list(item) if "/.thumbs/" not in path)
        if not files:
            raise KeyError(f"no SimReady asset '{item}' in {self.REPO}, list the items with: import object {self.meta.name}")
        dest = self.package(item)
        listed = {path.removeprefix(f"{item}/") for path in files}
        for path in listed:
            if not path.endswith(LAYERS) and not path.endswith(".json") and path != ".metadata/LICENSE.txt":
                continue
            target = dest / path
            if not target.is_file():
                self.provider.get(f"{item}/{path}", target)
        stated = json.loads((dest / PACKAGE).read_text()).get("license")
        if stated != LICENSE:
            raise ValueError(f"{item} is under {stated}, only {LICENSE} SimReady assets are imported")
        root = self.root(item)
        for path in missing(root):
            if not path.is_relative_to(dest.resolve()):
                continue
            relative = path.relative_to(dest.resolve()).as_posix()
            if relative in listed:
                self.provider.get(f"{item}/{relative}", dest / relative)
        return dest

    def root(self, item: str) -> Path:
        return self.package(item) / json.loads((self.package(item) / ROOTS).read_text())["entries"][0]

    def convert(self, item: str, src: Path, out: Path) -> str:
        try:
            return self.build(item, src, out)
        finally:
            shutil.rmtree(src, ignore_errors=True)

    def build(self, item: str, src: Path, out: Path) -> str:
        """Build a downloaded package into out and keep its SimReady metadata beside the model."""
        root = self.root(item)
        size = footprint(root)
        if max(size[0], size[1]) > MAX_FOOTPRINT_M:
            raise ValueError(f"{item} is {size[0]:.2f} x {size[1]:.2f} m, the object builder rescales a footprint over {MAX_FOOTPRINT_M:g} m")
        target = out / DOMAIN / ASSET_TYPE / item
        self.cache.mkdir(parents=True, exist_ok=True)
        dataset = Path(tempfile.mkdtemp(prefix="stage-", dir=self.cache))
        try:
            with replacing(target):
                self.stage(item, dataset)
                builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
                builder.enable("formats", FORMATS)
                builder.enable("previews")
                builder.build()
                if not (target / ANNOTATION_NAME).is_file():
                    raise RuntimeError(f"object build failed for {item}, the build log above names the step")
                self.keep(src, root.parent, target / KEPT)
        finally:
            shutil.rmtree(dataset)
        return str(Path(DOMAIN) / ASSET_TYPE / item)

    def stage(self, item: str, dataset: Path) -> Path:
        """Lay a downloaded package out as builder input under dataset, its joints as articulation, and return its directory."""
        root = self.root(item)
        source = dataset / DOMAIN / ASSET_TYPE / item
        shutil.copytree(root.parent, source)
        annotate(source / ANNOTATION_NAME, item, description(item), [ORIGIN, kind(item)], kind(item))
        write_articulation(source / ANNOTATION_NAME, usd_articulation(source / root.name, link_name, joint_name))
        return source

    @staticmethod
    def keep(package: Path, usd: Path, kept: Path) -> None:
        """Copy the package's SimReady metadata and physics payloads into kept."""
        for path in sorted((package / ".metadata").glob("*")):
            (kept / "metadata").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, kept / "metadata" / path.name)
        shutil.copyfile(package / PACKAGE, kept / PACKAGE)
        for folder in PHYSICS:
            if (usd / folder).is_dir():
                shutil.copytree(usd / folder, kept / folder)

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=f"https://huggingface.co/datasets/{self.REPO}/tree/{self.REVISION}/{item}", license=LICENSE, authors=AUTHORS, citation=f"NVIDIA SimReady assets, huggingface.co/datasets/{self.REPO}")


class SimReadyDSX(SimReady):
    meta = IMPORTERS.meta("object", "simready_dsx")
    REPO = "nvidia/simready-dsx"
    REVISION = "0a9e9ddd94d1bb6936588d2a4f7ee49b92ada794"
