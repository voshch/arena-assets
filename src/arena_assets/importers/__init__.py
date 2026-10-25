"""Importer registry and the adapter contract every dataset importer implements."""

from __future__ import annotations

import abc
import enum
import http.client
import importlib
import json
import logging
import os
import time
import typing
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import yaml

from arena_assets.importers.providers import TRANSPORT_ERRORS, Provider
from arena_assets.utils.ModelConverter import BlenderUnavailable

logger = logging.getLogger(__name__)

ATTRIBUTION_FILE = "ATTRIBUTION.yaml"
LISTING_FILE = "items.{kind}.json"
LISTING_TTL_S = 24 * 3600.0
ITEM_ERRORS = (KeyError, ValueError, RuntimeError, OSError, http.client.HTTPException)


class License(enum.Enum):
    YES = "yes"
    NC = "nc"
    SA = "sa"
    NO = "no"
    UNCLEAR = "unclear"


class OnDemand(enum.Enum):
    ITEM = "item"
    CHUNK = "chunk"
    BULK = "bulk"


@attrs.frozen
class ImporterMeta:
    name: str
    kind: str
    package: str
    target: str
    license: License
    host: str
    on_demand: OnDemand
    description: str
    listing_ttl_s: float | None = LISTING_TTL_S


@attrs.frozen
class Listing:
    """Item ids of one source and kind as last fetched, expires_at None for a list that never changes."""

    items: list[str]
    fetched_at: float
    expires_at: float | None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


def listing_path(cache: Path, kind: str) -> Path:
    return cache / LISTING_FILE.format(kind=kind)


def read_listing(path: Path) -> Listing | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return Listing(items=[str(item) for item in data["items"]], fetched_at=float(data["fetched_at"]), expires_at=None if data["expires_at"] is None else float(data["expires_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_listing(path: Path, items: list[str], ttl_s: float | None, now: float) -> Listing:
    listing = Listing(items=items, fetched_at=now, expires_at=None if ttl_s is None else now + ttl_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(attrs.asdict(listing)))
    part.replace(path)
    return listing


@attrs.frozen
class Attribution:
    source: str
    url: str
    license: str
    authors: str
    citation: str = ""


def write_attribution(path: Path, attribution: Attribution) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(attrs.asdict(attribution), f, sort_keys=False)


def read_attribution(path: Path) -> Attribution:
    with open(path) as f:
        return Attribution(**yaml.safe_load(f))


class Importer(abc.ABC):
    meta: typing.ClassVar[ImporterMeta]

    def __init__(self, cache: Path | None = None, provider: Provider | None = None):
        self.cache = cache if cache is not None else default_cache(self.meta.name)
        self.provider = provider if provider is not None else self.default_provider()

    @classmethod
    def default_provider(cls) -> Provider:
        raise NotImplementedError(f"{cls.__name__} has no default provider, pass one")

    @classmethod
    def default_out(cls) -> Path:
        """Root the native resolvers of this kind read local items from."""
        return import_root()

    @abc.abstractmethod
    def items(self) -> Iterable[str]: ...

    @abc.abstractmethod
    def fetch(self, item: str) -> Path: ...

    @abc.abstractmethod
    def convert(self, item: str, src: Path, out: Path) -> str: ...

    @abc.abstractmethod
    def attribution(self, item: str) -> Attribution: ...

    def cached_items(self) -> list[str]:
        """Item ids from the listing cache while it is fresh, else fetched and cached, the stale list when the host fails."""
        path = listing_path(self.cache, self.meta.kind)
        cached = read_listing(path)
        now = time.time()
        if cached is not None and not cached.expired(now):
            return cached.items
        try:
            items = list(self.items())
        except TRANSPORT_ERRORS as error:
            if cached is None:
                raise
            logger.warning("%s %s: listing the items failed (%s), showing the list fetched %s", self.meta.kind, self.meta.name, error, time.strftime("%Y-%m-%d %H:%M", time.localtime(cached.fetched_at)))
            return cached.items
        return write_listing(path, items, self.meta.listing_ttl_s, now).items

    def import_item(self, item: str, out: Path) -> Path:
        src = self.fetch(item)
        relpath = self.convert(item, src, out)
        target = out / relpath
        write_attribution(target / ATTRIBUTION_FILE, self.attribution(item))
        return target

    def import_items(self, items: Iterable[str], out: Path) -> dict[str, Path]:
        """Import each item, a failed one logged and skipped."""
        imported: dict[str, Path] = {}
        for item in items:
            try:
                imported[item] = self.import_item(item, out)
            except BlenderUnavailable:
                raise
            except ITEM_ERRORS as error:
                logger.warning("%s %s %s failed: %s", self.meta.kind, self.meta.name, item, reason(error))
        return imported


def reason(error: Exception) -> str:
    """First line of an error's message, a KeyError's without the quotes."""
    text = (str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)).strip()
    return text.splitlines()[0] if text else repr(error)


def import_root() -> Path:
    """Imported assets root the resolvers read: ARENA_ASSETS_DIR_IMPORTED, else ARENA_ASSETS_DIR/_imported."""
    imported = os.environ.get("ARENA_ASSETS_DIR_IMPORTED")
    if imported:
        return Path(imported)
    root = os.environ.get("ARENA_ASSETS_DIR")
    if not root:
        raise RuntimeError("neither ARENA_ASSETS_DIR_IMPORTED nor ARENA_ASSETS_DIR is set, pass an explicit target dir")
    return Path(root) / "_imported"


def default_cache(name: str) -> Path:
    root = os.environ.get("ARENA_ASSETS_DIR")
    if not root:
        raise RuntimeError("ARENA_ASSETS_DIR is not set, pass an explicit cache dir")
    return Path(root) / ".import" / name


class ImporterRegistry:
    """Importers keyed by the kind they produce and their source, so one source can serve several kinds."""

    def __init__(self):
        self._metas: dict[tuple[str, str], ImporterMeta] = {}
        self._classes: dict[tuple[str, str], type[Importer]] = {}

    def register(self, meta: ImporterMeta) -> None:
        if (meta.kind, meta.name) in self._metas:
            raise ValueError(f"{meta.kind} importer '{meta.name}' is already registered")
        self._metas[(meta.kind, meta.name)] = meta

    def __iter__(self) -> Iterator[ImporterMeta]:
        return iter(self._metas[key] for key in sorted(self._metas))

    def __len__(self) -> int:
        return len(self._metas)

    def __contains__(self, key: object) -> bool:
        return key in self._metas

    def kinds(self) -> list[str]:
        return sorted({kind for kind, _ in self._metas})

    def names(self, kind: str) -> list[str]:
        names = sorted(name for registered, name in self._metas if registered == kind)
        if not names:
            raise KeyError(f"no importer kind '{kind}', kinds: {', '.join(self.kinds())}")
        return names

    def kinds_of(self, name: str) -> list[str]:
        """Kinds a source is registered under, empty for an unknown source."""
        return sorted(kind for kind, registered in self._metas if registered == name)

    def meta(self, kind: str, name: str) -> ImporterMeta:
        try:
            return self._metas[(kind, name)]
        except KeyError:
            raise KeyError(f"no {kind} importer '{name}', registered: {', '.join(self.names(kind))}") from None

    def load(self, kind: str, name: str) -> type[Importer]:
        key = (kind, name)
        if key not in self._classes:
            meta = self.meta(kind, name)
            module_name, _, attr = meta.target.partition(":")
            cls = getattr(importlib.import_module(module_name), attr)
            if not (isinstance(cls, type) and issubclass(cls, Importer)):
                raise TypeError(f"{meta.target} is not an Importer subclass")
            self._classes[key] = cls
        return self._classes[key]


IMPORTERS = ImporterRegistry()

for _meta in (
    ImporterMeta("gso", "object", "arena_assets", "arena_assets.importers.gso:GSO", License.YES, "fuel", OnDemand.ITEM, "Google Scanned Objects, 1k household objects from Gazebo Fuel"),
    ImporterMeta("ai2thor", "object", "arena_assets", "arena_assets.importers.ai2thor:AI2THOR", License.YES, "hfhub", OnDemand.ITEM, "AI2-THOR objects, the ProcTHOR, iTHOR and RoboTHOR asset set, ranged reads from the ManiSkill copy of the HSSD glTF conversion", listing_ttl_s=None),
    ImporterMeta("openrobotics", "object", "arena_assets", "arena_assets.importers.openrobotics:OpenRobotics", License.YES, "fuel", OnDemand.ITEM, "Open Robotics furniture, appliance, hospital, office and warehouse models from Gazebo Fuel, the CC0 and CC BY 4.0 ones"),
    ImporterMeta("simready", "object", "arena_assets", "arena_assets.importers.simready:SimReady", License.YES, "hfhub", OnDemand.ITEM, "NVIDIA SimReady assets, 2.5k household, hospital, warehouse, factory and vehicle part USD models with physics payloads, CC BY 4.0", listing_ttl_s=None),
    ImporterMeta("artvip", "object", "arena_assets", "arena_assets.importers.artvip:ArtVIP", License.YES, "hfhub", OnDemand.ITEM, "ArtVIP articulated objects, 439 household, furniture, appliance, lab, medical and industrial USD models with UsdPhysics joints, Apache 2.0", listing_ttl_s=None),
    ImporterMeta("partnet_mobility", "object", "arena_assets", "arena_assets.importers.partnet_mobility:PartNetMobility", License.NC, "hfhub", OnDemand.ITEM, "PartNet-Mobility articulated objects, 2.3k size-normalized URDF models of 46 categories built on ShapeNet, gated, non-commercial terms", listing_ttl_s=None),
    ImporterMeta("infinigen", "object", "arena_assets", "arena_assets.importers.infinigen:Infinigen", License.YES, "local", OnDemand.ITEM, "Infinigen-Articulated procedural doors, appliances, cabinets and tools, one articulated USD model per category and seed from a local Infinigen install, BSD 3-Clause"),
    ImporterMeta("artiverse", "object", "arena_assets", "arena_assets.importers.artiverse:Artiverse", License.NC, "hfhub", OnDemand.BULK, "Artiverse articulated objects, 3.5k part-segmented glTF models of 88 categories in one 4.6 GB archive, gated, source dataset terms", listing_ttl_s=None),
    ImporterMeta("simready_dsx", "object", "arena_assets", "arena_assets.importers.simready:SimReadyDSX", License.YES, "hfhub", OnDemand.ITEM, "NVIDIA SimReady AI factory assets (GB300 rack) with physics, electrical and thermal metadata, CC BY 4.0", listing_ttl_s=None),
    ImporterMeta("polyhaven", "material", "arena_assets", "arena_assets.importers.polyhaven:PolyHaven", License.YES, "polyhaven", OnDemand.ITEM, "Poly Haven CC0 textures as OmniPBR materials, 2k color, normal and occlusion-roughness-metalness maps"),
    ImporterMeta("ambientcg", "material", "arena_assets", "arena_assets.importers.ambientcg:AmbientCG", License.YES, "ambientcg", OnDemand.ITEM, "ambientCG CC0 materials as OmniPBR materials, 2K color, normal, roughness, metalness and occlusion maps"),
    ImporterMeta("procthor", "world", "arena_simulation_setup", "arena_simulation_setup.adapters.procthor:ProcTHOR", License.YES, "github", OnDemand.CHUNK, "ProcTHOR-10K houses as worlds of typed rooms, walls, doors and AI2-THOR furniture, split files from GitHub LFS", listing_ttl_s=None),
    ImporterMeta("barn", "world", "arena_simulation_setup", "arena_simulation_setup.adapters.barn:BARNWorld", License.YES, "github", OnDemand.ITEM, "BARN challenge worlds 0 to 299 with the challenge start and goal as scenario, world files from Daffan/the-barn-challenge", listing_ttl_s=None),
    ImporterMeta("houseexpo", "world", "arena_simulation_setup", "arena_simulation_setup.adapters.houseexpo:HouseExpo", License.UNCLEAR, "github", OnDemand.BULK, "HouseExpo floor plans as worlds of labeled rooms and walls, 35k plans in one archive, an MIT repository derived from SUNCG", listing_ttl_s=None),
    ImporterMeta("barn", "suite", "arena_evaluation", "arena_evaluation.adapters.barn:BARN", License.YES, "github", OnDemand.ITEM, "BARN challenge suite with its 300 worlds bundled plus a contest and a report manifest, world files and reference paths from Daffan/the-barn-challenge", listing_ttl_s=None),
    ImporterMeta("socnavbench", "suite", "arena_evaluation", "arena_evaluation.adapters.socnavbench:SocNavBenchSuite", License.UNCLEAR, "github", OnDemand.ITEM, "SocNavBench curated episodes on the ETH and Zara scenes as a suite with a contest and a report manifest", listing_ttl_s=None),
    ImporterMeta("eth_ucy", "scenario", "task_generator", "task_generator.adapters.eth_ucy:ETHUCY", License.UNCLEAR, "github", OnDemand.ITEM, "ETH and UCY pedestrian sequences (eth, hotel, univ, zara01, zara02) as replay scenarios in a world of the area walked, SocNavBench's copies", listing_ttl_s=None),
    ImporterMeta("socnavbench", "scenario", "task_generator", "task_generator.adapters.socnavbench:SocNavBench", License.UNCLEAR, "github", OnDemand.ITEM, "SocNavBench curated episodes on the ETH and Zara scenes as replay scenarios with a robot start and goal, each bringing the world of its scene", listing_ttl_s=None),
    ImporterMeta("juelich", "scenario", "task_generator", "task_generator.adapters.juelich:Juelich", License.YES, "http", OnDemand.CHUNK, "Juelich pedestrian experiments (BaSiGo 2013, HERMES 2009) as replay scenarios in a world per lab layout", listing_ttl_s=None),
    ImporterMeta("madras", "scenario", "task_generator", "task_generator.adapters.madras:MADRAS", License.YES, "http", OnDemand.BULK, "MADRAS TopView sequences of the Fete des Lumieres 2022 in Lyon as dense crowd replay scenarios on an open floor, one archive", listing_ttl_s=None),
    ImporterMeta("glow", "scenario", "task_generator", "task_generator.adapters.glow:GLOW", License.YES, "http", OnDemand.BULK, "Eindhoven GLOW 2019 crowd recording as replay scenarios of 5-minute periods on an open floor, one 1 GB recording", listing_ttl_s=None),
    ImporterMeta("fsd50k", "sound", "arena_simulation_setup", "arena_simulation_setup.adapters.fsd50k:FSD50K", License.YES, "hfhub", OnDemand.ITEM, "FSD50K CC0 and CC BY 3.0 clips as sound items", listing_ttl_s=None),
    ImporterMeta("cmu_arctic", "sound", "arena_simulation_setup", "arena_simulation_setup.adapters.cmu_arctic:CMUArctic", License.YES, "http", OnDemand.ITEM, "CMU ARCTIC read-speech utterances of seven speakers as speech sound items", listing_ttl_s=None),
    ImporterMeta("musohu", "episode", "arena_evaluation", "arena_evaluation.adapters.musohu:MuSoHu", License.YES, "http", OnDemand.ITEM, "MuSoHu rosbag1 human navigation runs as evaluation recorder episodes"),
):
    IMPORTERS.register(_meta)
del _meta
