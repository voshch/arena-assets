import json
import shutil
from pathlib import Path

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation, ObjectDatabaseBuilder
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.ai2thor import ARCHIVE, MEMBERS, REVISION, TYPES, stage
from arena_assets.semantic import CURRENT_ASA
from arena_assets.utils.Database import Database

ITEM = "Dining_Table_203_1"
OBJECT_TYPE = "DiningTable"
FIXTURE = Path(__file__).parent / "fixtures" / "ai2thor" / f"{ITEM}.glb"
RELPATH = Path("Common") / "Object" / ITEM
EXTENTS = (1.4696, 0.9881, 0.7475)
CENTER_XY = (0.0, 0.05145)
SMALL_ITEM = "CreditCard_3"
ABSENT_ITEM = "CD_1"


def importer(tmp_path: Path):
    return IMPORTERS.load("object", "ai2thor")(cache=tmp_path / "cache")


def seed(tmp_path: Path, types: dict[str, str] | None = None) -> Path:
    cache = tmp_path / "cache"
    src = cache / "objects" / f"{ITEM}.glb"
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, src)
    entries = {ITEM: OBJECT_TYPE} if types is None else types
    database: dict[str, list[dict]] = {}
    for asset, object_type in entries.items():
        database.setdefault(object_type, []).append({"assetId": asset, "objectType": object_type})
    (cache / TYPES).write_text(json.dumps(database))
    return src


def convert(tmp_path: Path) -> Path:
    out = tmp_path / "ai2thor"
    imp = importer(tmp_path)
    assert imp.convert(ITEM, imp.fetch(ITEM), out) == str(RELPATH)
    return out


def tags(source: Path) -> list[str]:
    return yaml.safe_load((source / ANNOTATION_NAME).read_text())["tags"]


def test_registry_resolves_ai2thor_with_a_pinned_default_provider():
    cls = IMPORTERS.load("object", "ai2thor")
    assert cls.meta.kind == "object"
    assert cls.default_provider().url(ARCHIVE) == f"https://huggingface.co/datasets/haosulab/AI2THOR/resolve/{REVISION}/ai2thor.zip"


def test_fetch_serves_a_cached_object_without_the_archive(tmp_path):
    src = seed(tmp_path)
    assert importer(tmp_path).fetch(ITEM) == src


def test_stage_holds_the_annotation_and_one_model_file(tmp_path):
    source = stage(ITEM, FIXTURE, tmp_path / "dataset", OBJECT_TYPE)
    assert sorted(path.name for path in source.iterdir()) == sorted([ANNOTATION_NAME, f"{ITEM}.glb"])
    assert (source / f"{ITEM}.glb").read_bytes() == FIXTURE.read_bytes()


def test_stage_kind_comes_from_the_name_and_keeps_the_object_type(tmp_path):
    source = stage(ITEM, FIXTURE, tmp_path / "dataset", OBJECT_TYPE)
    raw = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert raw["asa"] == CURRENT_ASA
    assert raw["desc"] == "Dining Table"
    assert raw["tags"] == ["domain::common", "kind::table", "origin::ai2thor", "dining_table"]


def test_stage_kind_falls_back_to_the_object_type(tmp_path):
    assert tags(stage("bin_11", FIXTURE, tmp_path / "dataset", "GarbageCan")) == ["domain::common", "kind::garbage_can", "origin::ai2thor"]


def test_stage_without_an_object_type_describes_the_name(tmp_path):
    source = stage("FloorPlan2_physics-polySurface1", FIXTURE, tmp_path / "dataset", "")
    raw = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert raw["desc"] == "FloorPlan2 physics-polySurface1"
    assert raw["tags"] == ["domain::common", "origin::ai2thor"]


def test_object_type_reads_the_procthor_asset_database(tmp_path):
    seed(tmp_path, {ITEM: OBJECT_TYPE, "bin_11": "GarbageCan"})
    imp = importer(tmp_path)
    assert imp.object_type("bin_11") == "GarbageCan"
    assert imp.object_type("FloorPlan2_physics-polySurface1") == ""


def test_attribution_is_cc_by_ai2(tmp_path):
    attribution = importer(tmp_path).attribution(ITEM)
    assert attribution.license == "CC-BY-4.0"
    assert attribution.url.endswith(f"ai2thor.zip#{MEMBERS}{ITEM}.glb")
    assert "Allen Institute for AI" in attribution.authors
    assert "Kolve" in attribution.citation


def test_fixture_attribution_names_the_fixture():
    attribution = yaml.safe_load((FIXTURE.parent / ATTRIBUTION_FILE).read_text())
    assert attribution["license"] == "CC-BY-4.0"
    assert [name for item in attribution["items"] for name in item] == [FIXTURE.name]


def test_convert_turns_the_y_up_glb_z_up_and_keeps_the_origin(blender, tmp_path):
    seed(tmp_path)
    out = convert(tmp_path)
    annotation = ObjectDatabaseBuilder.read_annotation_file(out / RELPATH)
    assert isinstance(annotation, ObjectAnnotation)
    box = annotation.bounding_box
    assert (box.max_x - box.min_x, box.max_y - box.min_y, box.max_z - box.min_z) == pytest.approx(EXTENTS, abs=2e-4)
    assert ((box.min_x + box.max_x) / 2, (box.min_y + box.max_y) / 2) == pytest.approx(CENTER_XY, abs=2e-4)
    assert box.min_z == pytest.approx(0.0, abs=1e-4)
    raw = yaml.safe_load((out / RELPATH / ANNOTATION_NAME).read_text())
    assert raw["path"] == f"Object/{ITEM}"
    assert raw["tags"] == ["domain::common", "kind::table", "origin::ai2thor", "dining_table"]


def test_a_failed_rebuild_keeps_the_item(blender, tmp_path):
    src = seed(tmp_path)
    out = convert(tmp_path)
    kept = out / RELPATH / "kept.txt"
    kept.write_text("x")
    src.write_bytes(b"not a glb")
    with pytest.raises(RuntimeError, match=f"object build failed for {ITEM}"):
        importer(tmp_path).convert(ITEM, src, out)
    assert kept.is_file()
    assert (out / RELPATH / f"{ITEM}.usdz").is_file()
    assert sorted(path.name for path in (out / RELPATH).parent.iterdir()) == [ITEM]
    assert Database(out / "Common" / DATABASE_NAME).list_all("object")["ids"] == [f"Object/{ITEM}"]


def test_a_failed_stage_leaves_no_stage_directory(tmp_path):
    seed(tmp_path)
    with pytest.raises(FileNotFoundError):
        importer(tmp_path).build({ITEM: tmp_path / "missing.glb"}, tmp_path / "ai2thor")
    assert not list((tmp_path / "cache").glob("stage-*"))


def test_import_items_builds_and_attributes_each_item(blender, tmp_path):
    seed(tmp_path)
    out = tmp_path / "ai2thor"
    assert importer(tmp_path).import_items([ITEM], out) == {ITEM: out / RELPATH}
    assert read_attribution(out / RELPATH / ATTRIBUTION_FILE).license == "CC-BY-4.0"


@pytest.mark.network
def test_import_items_fetches_through_ranged_reads_and_skips_what_the_source_lacks(blender, tmp_path):
    imp = importer(tmp_path)
    out = tmp_path / "ai2thor"
    targets = imp.import_items([ABSENT_ITEM, SMALL_ITEM], out)
    assert targets == {SMALL_ITEM: out / "Common" / "Object" / SMALL_ITEM}
    assert tags(targets[SMALL_ITEM])[:2] == ["domain::common", "kind::credit_card"]
    assert (tmp_path / "cache" / f"{ARCHIVE}.directory").stat().st_size > 1_000_000
    assert SMALL_ITEM in set(importer(tmp_path).items())
    with pytest.raises(KeyError, match=ABSENT_ITEM):
        imp.fetch(ABSENT_ITEM)
