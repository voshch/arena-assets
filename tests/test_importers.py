import contextlib
import functools
import http.server
import json
import logging
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest
import yaml

from arena_assets.importers import (
    ATTRIBUTION_FILE,
    IMPORTERS,
    Attribution,
    Importer,
    ImporterMeta,
    ImporterRegistry,
    License,
    LISTING_TTL_S,
    OnDemand,
    listing_path,
    read_attribution,
    read_listing,
    write_attribution,
    write_listing,
)
from arena_assets.impl import ANNOTATION_NAME
from arena_assets.impl.build import replacing
from arena_assets.importers.providers import Http


def test_registry_lists_importers_by_kind_then_source():
    assert [(meta.kind, meta.name) for meta in IMPORTERS] == [
        ("episode", "musohu"),
        ("material", "ambientcg"),
        ("material", "polyhaven"),
        ("object", "ai2thor"),
        ("object", "artiverse"),
        ("object", "artvip"),
        ("object", "gso"),
        ("object", "infinigen"),
        ("object", "openrobotics"),
        ("object", "partnet_mobility"),
        ("object", "simready"),
        ("object", "simready_dsx"),
        ("scenario", "eth_ucy"),
        ("scenario", "glow"),
        ("scenario", "juelich"),
        ("scenario", "madras"),
        ("scenario", "socnavbench"),
        ("sound", "cmu_arctic"),
        ("sound", "fsd50k"),
        ("suite", "barn"),
        ("suite", "socnavbench"),
        ("world", "barn"),
        ("world", "houseexpo"),
        ("world", "procthor"),
    ]


def test_kinds_and_the_sources_of_a_kind():
    assert IMPORTERS.kinds() == ["episode", "material", "object", "scenario", "sound", "suite", "world"]
    assert IMPORTERS.names("world") == ["barn", "houseexpo", "procthor"]
    assert IMPORTERS.kinds_of("barn") == ["suite", "world"]
    assert IMPORTERS.kinds_of("socnavbench") == ["scenario", "suite"]
    assert IMPORTERS.kinds_of("nope") == []


def test_listing_imports_nothing():
    code = "import sys\nfrom arena_assets.importers import IMPORTERS\nlist(IMPORTERS)\nprint(' '.join(sorted(m for m in sys.modules if m.startswith('arena_assets.importers.') and m != 'arena_assets.importers.providers')))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == ""


def test_meta_lookup_and_membership():
    assert ("object", "gso") in IMPORTERS
    assert ("world", "gso") not in IMPORTERS
    assert ("object", "nope") not in IMPORTERS
    meta = IMPORTERS.meta("object", "gso")
    assert meta.kind == "object"
    assert meta.license is License.YES
    assert meta.on_demand is OnDemand.ITEM


def test_meta_error_lists_the_sources_of_the_kind():
    with pytest.raises(KeyError, match="no object importer 'nope', registered: ai2thor, artiverse, artvip, gso, infinigen, openrobotics, partnet_mobility, simready, simready_dsx"):
        IMPORTERS.meta("object", "nope")


def test_unknown_kind_lists_the_kinds():
    with pytest.raises(KeyError, match="no importer kind 'nope', kinds: episode, material, object, scenario, sound, suite, world"):
        IMPORTERS.meta("nope", "gso")


def test_register_rejects_a_duplicate_of_kind_and_source():
    registry = ImporterRegistry()
    meta = ImporterMeta("x", "object", "pkg", "pkg.x:X", License.NO, "http", OnDemand.BULK, "")
    registry.register(meta)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(meta)
    assert list(registry) == [meta]


def test_one_source_registers_under_several_kinds():
    registry = ImporterRegistry()
    as_object = ImporterMeta("x", "object", "pkg", "pkg.x:X", License.NO, "http", OnDemand.BULK, "")
    as_world = ImporterMeta("x", "world", "pkg", "pkg.x:XWorld", License.NO, "http", OnDemand.BULK, "")
    registry.register(as_world)
    registry.register(as_object)
    assert list(registry) == [as_object, as_world]
    assert registry.meta("world", "x") is as_world


def test_load_rejects_non_importer():
    registry = ImporterRegistry()
    registry.register(ImporterMeta("bad", "object", "pkg", "pathlib:Path", License.NO, "http", OnDemand.BULK, ""))
    with pytest.raises(TypeError, match="not an Importer"):
        registry.load("object", "bad")


def test_attribution_roundtrip(tmp_path):
    attribution = Attribution(source="x", url="https://example.org/x", license="CC-BY-4.0", authors="A, B", citation="A et al. 2020")
    path = tmp_path / "item" / ATTRIBUTION_FILE
    write_attribution(path, attribution)
    with open(path) as f:
        assert list(yaml.safe_load(f)) == ["source", "url", "license", "authors", "citation"]
    assert read_attribution(path) == attribution


class Counting(Importer):
    meta = ImporterMeta("counting", "object", "arena_assets", "tests.test_importers:Counting", License.YES, "http", OnDemand.ITEM, "test double")

    def items(self):
        return ["a", "b"]

    def fetch(self, item: str) -> Path:
        src = self.cache / f"{item}.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(item)
        return src

    def convert(self, item: str, src: Path, out: Path) -> str:
        target = out / "object" / item
        target.mkdir(parents=True)
        (target / "model.txt").write_text(src.read_text().upper())
        return f"object/{item}"

    def attribution(self, item: str) -> Attribution:
        return Attribution(source="counting", url=f"https://example.org/{item}", license="CC0-1.0", authors="nobody")


def test_import_item_template(tmp_path):
    importer = Counting(cache=tmp_path / "cache", provider=Http("https://example.org"))
    target = importer.import_item("a", tmp_path / "out")
    assert target == tmp_path / "out" / "object" / "a"
    assert (target / "model.txt").read_text() == "A"
    assert read_attribution(target / ATTRIBUTION_FILE).url == "https://example.org/a"
    assert sorted(p.name for p in target.iterdir()) == [ATTRIBUTION_FILE, "model.txt"]
    assert (tmp_path / "cache" / "a.txt").is_file()


def test_default_cache_needs_env(without_assets_dir):
    with pytest.raises(RuntimeError, match="ARENA_ASSETS_DIR"):
        Counting(provider=Http("https://example.org"))


@pytest.fixture
def without_assets_dir():
    import os

    saved = os.environ.pop("ARENA_ASSETS_DIR", None)
    yield
    if saved is not None:
        os.environ["ARENA_ASSETS_DIR"] = saved


class Quick(Http):
    backoff = 0.0


class Listed(Counting):
    meta = ImporterMeta("listed", "world", "arena_assets", "tests.test_importers:Listed", License.YES, "http", OnDemand.ITEM, "test double")

    def items(self):
        return self.provider.read(self.provider.url("items.txt"))[0].decode().split()


class Pinned(Listed):
    meta = ImporterMeta("pinned", "world", "arena_assets", "tests.test_importers:Pinned", License.YES, "http", OnDemand.ITEM, "test double", listing_ttl_s=None)


@contextlib.contextmanager
def serving(root: Path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object):
        pass


def closed_port() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


def test_item_list_is_fetched_once_and_kept_for_a_day_per_kind(tmp_path):
    (tmp_path / "host").mkdir()
    (tmp_path / "host" / "items.txt").write_text("a b c")
    with serving(tmp_path / "host") as base:
        start = time.time()
        assert Listed(cache=tmp_path / "cache", provider=Quick(base)).cached_items() == ["a", "b", "c"]
    path = tmp_path / "cache" / "items.world.json"
    assert path == listing_path(tmp_path / "cache", "world")
    listing = read_listing(path)
    assert listing.items == ["a", "b", "c"]
    assert start <= listing.fetched_at <= time.time()
    assert listing.expires_at == listing.fetched_at + LISTING_TTL_S
    assert Listed(cache=tmp_path / "cache", provider=Quick(closed_port())).cached_items() == ["a", "b", "c"]


def test_expired_item_list_is_fetched_again(tmp_path):
    write_listing(listing_path(tmp_path / "cache", "world"), ["old"], LISTING_TTL_S, time.time() - LISTING_TTL_S - 1)
    (tmp_path / "host").mkdir()
    (tmp_path / "host" / "items.txt").write_text("new")
    with serving(tmp_path / "host") as base:
        assert Listed(cache=tmp_path / "cache", provider=Quick(base)).cached_items() == ["new"]
    assert read_listing(listing_path(tmp_path / "cache", "world")).items == ["new"]


def test_item_list_of_a_pinned_source_never_expires(tmp_path):
    (tmp_path / "host").mkdir()
    (tmp_path / "host" / "items.txt").write_text("x")
    with serving(tmp_path / "host") as base:
        Pinned(cache=tmp_path / "cache", provider=Quick(base)).cached_items()
    listing = read_listing(listing_path(tmp_path / "cache", "world"))
    assert listing.expires_at is None
    assert not listing.expired(time.time() + 10 * LISTING_TTL_S)


def test_stale_item_list_stands_in_while_the_host_fails(tmp_path, caplog):
    write_listing(listing_path(tmp_path / "cache", "world"), ["old"], LISTING_TTL_S, time.time() - LISTING_TTL_S - 1)
    with caplog.at_level(logging.WARNING, logger="arena_assets.importers"):
        assert Listed(cache=tmp_path / "cache", provider=Quick(closed_port())).cached_items() == ["old"]
    assert read_listing(listing_path(tmp_path / "cache", "world")).items == ["old"]


def test_item_list_without_a_cache_raises_while_the_host_fails(tmp_path):
    with pytest.raises(urllib.error.URLError):
        Listed(cache=tmp_path / "cache", provider=Quick(closed_port())).cached_items()
    assert not (tmp_path / "cache").exists()


def test_unreadable_item_list_counts_as_missing(tmp_path):
    path = listing_path(tmp_path, "world")
    path.write_text("{not json")
    assert read_listing(path) is None
    path.write_text(json.dumps({"items": ["a"]}))
    assert read_listing(path) is None


def _built(root: Path, name: str, annotation: str) -> Path:
    item = root / "Object" / name
    item.mkdir(parents=True)
    (item / ANNOTATION_NAME).write_text(annotation)
    return item


def _listed(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "Object").iterdir())


def test_replacing_keeps_the_rebuilt_item_and_drops_the_old_one(tmp_path):
    chair = _built(tmp_path, "chair", "old")
    with replacing(chair):
        assert not chair.exists()
        _built(tmp_path, "chair", "new")
    assert (chair / ANNOTATION_NAME).read_text() == "new"
    assert _listed(tmp_path) == ["chair"]


def test_replacing_restores_an_item_rebuilt_without_an_annotation(tmp_path):
    chair = _built(tmp_path, "chair", "old")
    with replacing(chair):
        chair.mkdir()
        (chair / "chair.usdz").write_text("partial")
    assert sorted(path.name for path in chair.iterdir()) == [ANNOTATION_NAME]
    assert (chair / ANNOTATION_NAME).read_text() == "old"
    assert _listed(tmp_path) == ["chair"]


def test_replacing_restores_every_item_when_the_rebuild_raises(tmp_path):
    chair = _built(tmp_path, "chair", "old chair")
    table = _built(tmp_path, "table", "old table")
    with pytest.raises(RuntimeError, match="build failed"), replacing(chair, table, tmp_path / "Object" / "lamp"):
        _built(tmp_path, "chair", "new chair")
        raise RuntimeError("build failed")
    assert [(item / ANNOTATION_NAME).read_text() for item in (chair, table)] == ["old chair", "old table"]
    assert _listed(tmp_path) == ["chair", "table"]
