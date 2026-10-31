import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation, ObjectDatabaseBuilder
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.gso import stage
from arena_assets.importers.providers import Fuel
from arena_assets.semantic import CURRENT_ASA

ITEM = "Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L"
FIXTURE = Path(__file__).parent / "fixtures" / "gso" / f"{ITEM}.zip"
RELPATH = Path("Common") / "Object" / ITEM
EXTENTS = (0.0812, 0.0583, 0.0152)


def importer(tmp_path: Path):
    return IMPORTERS.load("object", "gso")(cache=tmp_path / "cache", provider=Fuel("GoogleResearch"))


def seed(tmp_path: Path, metadata: dict | None = None) -> Path:
    src = tmp_path / "cache" / f"{ITEM}.zip"
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, src)
    if metadata is not None:
        src.with_suffix(".json").write_text(json.dumps(metadata))
    return src


def staged(tmp_path: Path, metadata: dict | None = None) -> Path:
    return stage(ITEM, seed(tmp_path, metadata), tmp_path / "dataset")


def convert(tmp_path: Path, metadata: dict | None = None) -> Path:
    out = tmp_path / "gso"
    assert importer(tmp_path).convert(ITEM, seed(tmp_path, metadata), out) == str(RELPATH)
    return out


def vertices(mesh: Path) -> list[tuple[float, float, float]]:
    return [tuple(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith("v ")]


def test_registry_resolves_gso_with_a_default_provider():
    cls = IMPORTERS.load("object", "gso")
    assert cls.meta.kind == "object"
    assert cls.default_provider().url(f"{ITEM}/tip/{ITEM}.zip") == f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{ITEM}/tip/{ITEM}.zip"


def test_stage_holds_the_annotation_and_one_textured_model(tmp_path):
    source = staged(tmp_path)
    assert sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()) == [ANNOTATION_NAME, "materials/textures/texture.png", "meshes/model.mtl", "meshes/model.obj"]


def test_stage_points_the_texture_map_at_an_existing_file(tmp_path):
    source = staged(tmp_path)
    maps = [line.split()[1] for line in (source / "meshes" / "model.mtl").read_text().splitlines() if line.startswith("map_Kd")]
    assert maps == ["../materials/textures/texture.png"]
    assert (source / "meshes" / maps[0]).is_file()


def test_stage_turns_the_z_up_mesh_y_up(tmp_path):
    with zipfile.ZipFile(FIXTURE) as archive:
        upstream = [tuple(map(float, line.split()[1:4])) for line in archive.read("meshes/model.obj").decode().splitlines() if line.startswith("v ")]
    turned = vertices(staged(tmp_path) / "meshes" / "model.obj")
    assert turned == [pytest.approx((x, z, -y)) for x, y, z in upstream]
    heights = [y for _, y, _ in turned]
    assert min(heights) == pytest.approx(0.0, abs=1e-4)
    assert max(heights) - min(heights) == pytest.approx(EXTENTS[2], abs=1e-4)


def test_stage_annotates_with_taxonomy_tags(tmp_path):
    source = staged(tmp_path, {"name": ITEM, "categories": ["Consumer Goods", "Toys"]})
    raw = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert raw["asa"] == CURRENT_ASA
    assert raw["desc"]
    assert raw["tags"] == ["domain::common", "kind::consumer_goods", "origin::gso", "toys"]


def test_stage_without_categories_keeps_origin_and_domain(tmp_path):
    raw = yaml.safe_load((staged(tmp_path) / ANNOTATION_NAME).read_text())
    assert raw["tags"] == ["domain::common", "origin::gso"]


def test_attribution_is_cc_by_google(tmp_path):
    attribution = importer(tmp_path).attribution(ITEM)
    assert attribution.license == "CC-BY-4.0"
    assert attribution.url == f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{ITEM}"
    assert "Google" in attribution.authors


def test_convert_annotation_is_measured_upright(blender, tmp_path):
    out = convert(tmp_path, {"name": ITEM, "categories": ["Consumer Goods"]})
    annotation = ObjectDatabaseBuilder.read_annotation_file(out / RELPATH)
    assert isinstance(annotation, ObjectAnnotation)
    box = annotation.bounding_box
    assert (box.max_x - box.min_x, box.max_y - box.min_y, box.max_z - box.min_z) == pytest.approx(EXTENTS, abs=2e-4)
    assert box.min_z == pytest.approx(0.0, abs=1e-4)
    raw = yaml.safe_load((out / RELPATH / ANNOTATION_NAME).read_text())
    assert raw["path"] == f"Object/{ITEM}"
    assert raw["asa"] == CURRENT_ASA
    assert raw["tags"] == ["domain::common", "kind::consumer_goods", "origin::gso"]


def test_a_failed_reconvert_keeps_the_item(blender, tmp_path):
    out = convert(tmp_path)
    kept = out / RELPATH / "kept.txt"
    kept.write_text("x")
    broken = tmp_path / "broken" / f"{ITEM}.zip"
    broken.parent.mkdir()
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("model.config", "<model/>")
    with pytest.raises(ValueError, match="has no model.sdf"):
        importer(tmp_path).convert(ITEM, broken, out)
    assert kept.is_file()
    assert (out / RELPATH / f"{ITEM}.usdz").is_file()
    assert sorted(path.name for path in (out / RELPATH).parent.iterdir()) == [ITEM]


@pytest.mark.network
def test_fetch_of_an_unknown_model_names_the_listing_command(tmp_path):
    with pytest.raises(KeyError, match="no model 'No_Such_Model' of GoogleResearch on Fuel, list the items with: import object gso"):
        importer(tmp_path).fetch("No_Such_Model")


@pytest.mark.network
def test_import_item_end_to_end(blender, tmp_path):
    target = importer(tmp_path).import_item(ITEM, tmp_path / "gso")
    assert target == tmp_path / "gso" / RELPATH
    assert (target / f"{ITEM}.usdz").is_file()
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert attribution.license == "CC-BY-4.0"
    assert ObjectDatabaseBuilder.read_annotation_file(target).tags
