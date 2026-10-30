import json
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, License
from arena_assets.importers.partnet_mobility import REPO, REVISION, PartNetMobility, metric_scale, semantics, stage
from arena_assets.importers.providers import HFHub
from blender_scene import owners, scene, unmatched
from http_host import Host
from urdf_item import urdf_item

ITEM = "100013"
NAME = f"partnet_mobility_{ITEM}"


def item_zip(tmp_path: Path, category: str = "StorageFurniture") -> Path:
    """A PartNet-Mobility zip: the item folder with its URDF, OBJ parts, texture, meta.json and semantics.txt."""
    folder = tmp_path / "upstream" / ITEM
    urdf_item(folder)
    (folder / "meta.json").write_text(json.dumps({"model_cat": category, "model_id": ITEM}))
    (folder / "semantics.txt").write_text("link_0 hinge door\nlink_1 static frame\n")
    archive = tmp_path / f"{ITEM}.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                out.write(path, path.relative_to(folder.parent).as_posix())
    return archive


def importer(host: Host, tmp_path: Path) -> PartNetMobility:
    return PartNetMobility(cache=tmp_path / "cache", provider=HFHub(REPO, revision=REVISION, host=host.base))


def resolve(item: str) -> str:
    return f"/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(f'{item}.zip')}"


def test_registry_marks_partnet_mobility_non_commercial():
    assert IMPORTERS.load("object", "partnet_mobility") is PartNetMobility
    assert IMPORTERS.meta("object", "partnet_mobility").license is License.NC


def test_items_are_the_model_zips_in_numeric_order(host, tmp_path):
    host.pages[f"/api/datasets/{REPO}/tree/{REVISION}?recursive=true"] = json.dumps([{"type": "file", "path": path} for path in (".gitattributes", "100013.zip", "7128.zip", "notes.zip")]).encode()
    assert list(importer(host, tmp_path).items()) == ["7128", "100013"]


def test_gated_download_ends_with_the_request_access_line(host, tmp_path):
    host.statuses[resolve(ITEM)] = 403
    with pytest.raises(PermissionError, match=rf"PartNet-Mobility is gated .* request access at https://huggingface.co/datasets/{REPO} "):
        importer(host, tmp_path).fetch(ITEM)


def test_unknown_or_malformed_items_name_the_listing_command(host, tmp_path):
    with pytest.raises(KeyError, match="no PartNet-Mobility object '42' in sapien-sim/PartNetMobility, list the items with: import object partnet_mobility"):
        importer(host, tmp_path).fetch("42")
    with pytest.raises(KeyError, match="items are model ids such as 100013"):
        importer(host, tmp_path).fetch("door")


def test_fetch_downloads_the_zip(host, tmp_path):
    host.pages[resolve(ITEM)] = item_zip(tmp_path).read_bytes()
    assert importer(host, tmp_path).fetch(ITEM) == tmp_path / "cache" / f"{ITEM}.zip"


def test_semantics_label_each_link(tmp_path):
    (tmp_path / "semantics.txt").write_text("link_0 hinge door\nlink_1 slider drawer\n\nlink_2\n")
    assert semantics(tmp_path / "semantics.txt") == {"link_0": "door", "link_1": "drawer"}
    assert semantics(tmp_path / "missing.txt") == {}


def test_stage_merges_the_links_into_one_obj_with_the_urdf_joints(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    source = stage(ITEM, item_zip(tmp_path), tmp_path / "dataset")
    assert source == tmp_path / "dataset" / "Common" / "Object" / NAME
    assert sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()) == [ANNOTATION_NAME, f"{NAME}.mtl", f"{NAME}.obj", f"upstream/{ITEM}/images/wood.png"]
    assert f"map_Kd upstream/{ITEM}/images/wood.png" in (source / f"{NAME}.mtl").read_text()
    annotation = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert annotation["desc"] == "storage furniture"
    assert "origin::partnet_mobility" in annotation["tags"]
    articulation = annotation["articulation"]
    assert articulation["links"] == {"base": [], "link_1": ["link_1"], "link_0": ["link_0"]}
    assert [(joint["name"], joint["type"], joint["parent"], joint["child"]) for joint in articulation["joints"]] == [("joint_1", "fixed", "base", "link_1"), ("joint_0", "revolute", "link_1", "link_0")]
    assert articulation["states"] == {"closed": {"joint_0": 0.0}, "open": {"joint_0": 1.57}}


def test_metric_scale_meets_the_category_measure():
    assert metric_scale("Door", np.array([0.5, 0.2, 1.0])) == pytest.approx(2.05)
    assert metric_scale("StorageFurniture", np.array([1.0, 2.0, 1.6])) == pytest.approx(0.5)
    assert metric_scale("Microwave", np.array([1.0, 1.5, 0.8])) == pytest.approx(0.5 / 1.5)
    assert metric_scale("Oven", np.array([1.15, 1.72, 0.22])) * 1.72 == pytest.approx(0.8)
    assert metric_scale("Table", np.array([1.0, 1.6, 0.8])) == pytest.approx(0.75 / 0.8)
    assert metric_scale("Table", np.array([0.53, 1.97, 0.39])) * 1.97 == pytest.approx(2.0)
    assert metric_scale("Spaceship", np.array([1.0, 1.0, 1.0])) == 1.0


def depth(source: Path) -> float:
    points = np.array([[float(word) for word in line.split()[1:4]] for line in (source / f"{NAME}.obj").read_text().splitlines() if line.startswith("v ")])
    size = points.max(axis=0) - points.min(axis=0)
    return min(size[0], size[2])


def hinge(source: Path) -> np.ndarray:
    return np.array(yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]["joints"][1]["origin"])


def test_stage_scales_the_mesh_and_joints_to_the_category_size(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    scaled = stage(ITEM, item_zip(tmp_path / "scaled"), tmp_path / "scaled" / "dataset")
    raw = stage(ITEM, item_zip(tmp_path / "raw", category="Spaceship"), tmp_path / "raw" / "dataset")
    assert depth(scaled) == pytest.approx(0.5)
    assert hinge(scaled) == pytest.approx(hinge(raw) * 0.5 / depth(raw))


def test_staged_obj_loads_with_every_link_node_in_blender(blender_python, tmp_path):
    source = stage(ITEM, item_zip(tmp_path), tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    objects = scene(blender_python, source / f"{NAME}.obj")
    assert unmatched(objects, articulation) == []
    assert sorted(owners(objects, articulation).values()) == ["link_0", "link_0", "link_1"]


@pytest.mark.network
def test_live_item_stages_or_names_the_gate(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    imp = PartNetMobility(cache=tmp_path / "cache")
    try:
        src = imp.fetch(ITEM)
    except PermissionError as error:
        assert f"https://huggingface.co/datasets/{REPO}" in str(error)
        pytest.skip(f"no access to {REPO}: {error}")
    articulation = yaml.safe_load((stage(ITEM, src, tmp_path / "dataset") / ANNOTATION_NAME).read_text())["articulation"]
    assert any(joint["type"] != "fixed" for joint in articulation["joints"])


@pytest.mark.network
def test_live_listing_needs_no_access(tmp_path):
    items = list(PartNetMobility(cache=tmp_path / "cache").items())
    assert len(items) > 2300
    assert ITEM in items
