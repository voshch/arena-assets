import os
import subprocess
import sys

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS
from arena_assets.importers.infinigen import ENV, INSTALL, Infinigen, interpreter, link_name, object_name, repair, split
from blender_scene import owners, scene, unmatched


def installed() -> bool:
    try:
        interpreter()
    except RuntimeError:
        return False
    return True


def test_registry_resolves_infinigen():
    assert IMPORTERS.load("object", "infinigen") is Infinigen
    assert IMPORTERS.meta("object", "infinigen").license.value == "yes"


def test_items_are_category_and_seed():
    assert split("door/1001") == ("door", 1001)
    assert object_name("soap_dispenser/7") == "infinigen_soap_dispenser_7"
    for item in ("door", "door/", "door/one", "/1001"):
        with pytest.raises(KeyError, match="items are <category>/<seed> such as door/1001"):
            split(item)


def test_fixed_world_body_becomes_the_base_link():
    assert [link_name(prim) for prim in ("world", "link_1")] == ["base", "link_1"]


def test_missing_install_names_the_install_command(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != ENV}
    env["PATH"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-c", "from arena_assets.importers.infinigen import interpreter\ninterpreter()"], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert f"no Infinigen install, set {ENV} to the python of one or install it with: {INSTALL}" in result.stderr


@pytest.mark.skipif(not installed(), reason=f"no Infinigen install, set {ENV} or install it with: {INSTALL}")
@pytest.mark.parametrize(("item", "moving"), [("door/1001", {"door_hinge_left_0": "revolute", "door_handle_left_0": "revolute"}), ("dishwasher/1001", {"rack_joint_0": "prismatic", "rack_joint_1": "prismatic", "door_joint_0": "revolute", "knob_joint_0": "continuous"})])
def test_generated_asset_stages_its_joints_and_loads_in_blender(blender_python, tmp_path, item, moving):
    imp = Infinigen(cache=tmp_path / "cache")
    src = imp.fetch(item)
    source = imp.stage(item, src, tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    assert {joint["name"]: joint["type"] for joint in articulation["joints"]} == moving
    assert next(iter(articulation["links"])) == "base"
    assert articulation["states"]["open"]
    objects = scene(blender_python, source / src.name)
    assert unmatched(objects, articulation) == []
    assert set(owners(objects, articulation).values()) == set(articulation["links"])


def test_repair_splits_the_face_sizes_infinigen_writes_per_index(tmp_path):
    from pxr import Usd, UsdGeom

    path = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name, counts in (("Broken", [3] * 6), ("Whole", [3, 3])):
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateFaceVertexCountsAttr(counts)
    stage.Save()
    assert repair(path) == ["/Asset/Broken"]
    reopened = Usd.Stage.Open(str(path))
    assert [list(UsdGeom.Mesh(reopened.GetPrimAtPath(f"/Asset/{name}")).GetFaceVertexCountsAttr().Get()) for name in ("Broken", "Whole")] == [[3, 3], [3, 3]]
    assert repair(path) == []
