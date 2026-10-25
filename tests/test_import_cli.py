import io
import json
import shutil
import urllib.error
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS
from arena_assets.importers.ai2thor import ARCHIVE, MEMBERS, TYPES
from cli_output import plain

ITEM = "Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L"
FIXTURE = Path(__file__).parent / "fixtures" / "gso" / f"{ITEM}.zip"
RELPATH = Path("Common") / "Object" / ITEM
THOR_ITEM = "Dining_Table_203_1"
THOR_FIXTURE = Path(__file__).parent / "fixtures" / "ai2thor" / f"{THOR_ITEM}.glb"
THOR_BROKEN = "Broken_Table_1"
NO_BLENDER = {"ARENA_ASSETS_BLENDER": "/nonexistent/arena-assets-blender"}

runner = CliRunner()


def seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, cache / f"{ITEM}.zip")
    (cache / f"{ITEM}.json").write_text(json.dumps({"name": ITEM, "categories": ["Consumer Goods"]}))


def seed_ai2thor(cache: Path) -> None:
    """An AI2-THOR cache listing a good and a broken object, both fetched, so --all runs offline."""
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        for item in (THOR_ITEM, THOR_BROKEN):
            archive.writestr(f"{MEMBERS}{item}.glb", b"")
    data = body.getvalue()
    start = zipfile.ZipFile(io.BytesIO(data)).start_dir
    (cache / "objects").mkdir(parents=True)
    (cache / f"{ARCHIVE}.directory.json").write_text(json.dumps({"start": start, "size": len(data)}))
    (cache / f"{ARCHIVE}.directory").write_bytes(data[start:])
    shutil.copy(THOR_FIXTURE, cache / "objects" / f"{THOR_ITEM}.glb")
    (cache / "objects" / f"{THOR_BROKEN}.glb").write_bytes(b"not a glb")
    (cache / TYPES).write_text(json.dumps({"DiningTable": [{"assetId": THOR_ITEM, "objectType": "DiningTable"}]}))


def test_bare_import_lists_the_registry():
    result = runner.invoke(app, ["import"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0].split() == ["kind", "source", "license", "host", "on_demand", "description"]
    assert [tuple(line.split()[:2]) for line in lines[1:]] == [(meta.kind, meta.name) for meta in IMPORTERS]
    gso = next(line for line in lines if line.split()[:2] == ["object", "gso"])
    assert gso.split()[2:5] == ["yes", "fuel", "item"]


def test_bare_import_json():
    result = runner.invoke(app, ["import", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert [(row["kind"], row["name"]) for row in rows] == [(meta.kind, meta.name) for meta in IMPORTERS]
    assert next(row for row in rows if row["name"] == "gso") == {"kind": "object", "name": "gso", "license": "yes", "host": "fuel", "on_demand": "item", "description": IMPORTERS.meta("object", "gso").description}


def test_kind_alone_lists_its_sources():
    result = runner.invoke(app, ["import", "world"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert [tuple(line.split()[:2]) for line in lines if line.startswith("world ")] == [("world", "barn"), ("world", "houseexpo"), ("world", "procthor")]
    assert not any(line.startswith("object ") for line in lines)
    assert "3 source(s) of kind world, list one's items with: import world SOURCE" in lines


def test_unknown_kind_lists_the_kinds():
    result = runner.invoke(app, ["import", "nope", "x"])
    assert result.exit_code != 0
    assert "no importer kind 'nope'" in plain(result.output)
    assert all(kind in result.output for kind in IMPORTERS.kinds())


def test_a_source_in_place_of_the_kind_names_the_command():
    result = runner.invoke(app, ["import", "gso", ITEM])
    assert result.exit_code != 0
    assert "'gso' is a source of kind object: import object gso" in plain(result.output)


def test_unknown_source_lists_the_sources_of_the_kind():
    result = runner.invoke(app, ["import", "object", "nope", "x", "--out", "/tmp/x", "--cache", "/tmp/x"])
    assert result.exit_code != 0
    assert "no object importer 'nope'" in plain(result.output)
    assert all(name in result.output for name in IMPORTERS.names("object"))


def test_a_source_under_the_wrong_kind_names_its_kinds():
    result = runner.invoke(app, ["import", "object", "barn", "0"])
    assert result.exit_code != 0
    assert "'barn' is a source of kind suite and world: import suite barn" in plain(result.output)


def test_items_and_all_are_refused(tmp_path):
    result = runner.invoke(app, ["import", "object", "gso", ITEM, "--all", "--out", str(tmp_path), "--cache", str(tmp_path)])
    assert result.exit_code != 0
    assert "not both" in plain(result.output)


def test_out_default_needs_assets_dir(tmp_path):
    result = runner.invoke(app, ["import", "object", "gso", ITEM, "--cache", str(tmp_path)], env={"ARENA_ASSETS_DIR": None, "ARENA_ASSETS_DIR_IMPORTED": None})
    assert result.exit_code != 0
    assert "ARENA_ASSETS_DIR" in plain(result.output)
    assert "--out" in plain(result.output)


def test_import_from_seeded_cache_writes_item_and_attribution(blender, tmp_path):
    cache = tmp_path / "cache"
    seed(cache)
    out = tmp_path / "out"
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM, "--out", str(out), "--cache", str(cache)])
    assert result.exit_code == 0, result.output
    target = out / RELPATH
    assert result.output.strip() == str(target)
    assert (target / ATTRIBUTION_FILE).is_file()
    assert (target / f"{ITEM}.usdz").is_file()


def test_an_item_failing_past_its_fetch_is_a_one_line_error(tmp_path):
    cache = tmp_path / "cache"
    seed(cache)
    result = runner.invoke(app, ["import", "object", "gso", ITEM, "--out", str(tmp_path / "out"), "--cache", str(cache)], env=NO_BLENDER)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"arena-assets import: object gso {ITEM}: cannot run the Blender worker" in plain(result.output)
    assert "Traceback" not in plain(result.output)


def test_import_all_imports_what_it_can_and_counts_the_failed_items(blender, tmp_path):
    cache = tmp_path / "cache"
    seed_ai2thor(cache)
    out = tmp_path / "out"
    result = runner.invoke(app, ["import", "object", "ai2thor", "--all", "--out", str(out), "--cache", str(cache)])
    assert result.exit_code == 1, result.output
    assert str(out / "Common" / "Object" / THOR_ITEM) in result.output.splitlines()
    assert "arena-assets import: imported 1 object item(s) from ai2thor, 1 failed, the warnings above name them" in plain(result.output)
    assert f"object ai2thor {THOR_BROKEN} failed" in plain(result.output)
    assert (out / "Common" / "Object" / THOR_ITEM / ATTRIBUTION_FILE).is_file()
    assert not (out / "Common" / "Object" / THOR_BROKEN).exists()


def test_import_all_without_a_blender_worker_is_a_one_line_error(tmp_path):
    cache = tmp_path / "cache"
    seed_ai2thor(cache)
    result = runner.invoke(app, ["import", "object", "ai2thor", "--all", "--out", str(tmp_path / "out"), "--cache", str(cache)], env=NO_BLENDER)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "arena-assets import: object ai2thor: cannot run the Blender worker" in plain(result.output)
    assert "Traceback" not in plain(result.output)


def test_out_defaults_to_the_imported_root_and_cache_to_the_assets_dir(blender, tmp_path):
    seed(tmp_path / ".import" / "gso")
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM], env={"ARENA_ASSETS_DIR": str(tmp_path), "ARENA_ASSETS_DIR_IMPORTED": None})
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(tmp_path / "_imported" / RELPATH)


def test_out_default_follows_the_imported_assets_dir(blender, tmp_path):
    seed(tmp_path / ".import" / "gso")
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM], env={"ARENA_ASSETS_DIR": str(tmp_path), "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "mine")})
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(tmp_path / "mine" / RELPATH)


def test_bare_source_listing_caches_the_items_for_completion(tmp_path):
    pytest.importorskip("arena_simulation_setup.adapters.barn")
    result = runner.invoke(app, ["-s", "import", "world", "barn", "--cache", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[:3] == ["0", "1", "2"]
    cached = json.loads((tmp_path / "items.world.json").read_text())
    assert cached["items"] == [str(index) for index in range(300)]
    assert cached["expires_at"] is None


def test_source_listing_json_gives_the_items_and_the_target(tmp_path):
    pytest.importorskip("arena_simulation_setup.adapters.barn")
    result = runner.invoke(app, ["-s", "import", "world", "barn", "--cache", str(tmp_path), "--json"], env={"ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")})
    assert result.exit_code == 0, result.output
    listed = json.loads(result.output)
    assert listed == {"kind": "world", "source": "barn", "target": str(tmp_path / "imported" / "worlds"), "count": 300, "items": [str(index) for index in range(300)]}


@pytest.mark.network
def test_a_gated_listing_is_a_one_line_error(tmp_path):
    from arena_assets.importers.artiverse import ARCHIVE, REPO, Artiverse

    try:
        Artiverse(cache=tmp_path / "probe").provider.size(ARCHIVE)
    except urllib.error.HTTPError as error:
        if error.code not in (401, 403):
            raise
    else:
        pytest.skip(f"{REPO} is accessible, listing it downloads the whole archive")
    result = runner.invoke(app, ["import", "object", "artiverse", "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert f"request access at https://huggingface.co/datasets/{REPO}" in plain(result.output)
    assert "Traceback" not in plain(result.output)
