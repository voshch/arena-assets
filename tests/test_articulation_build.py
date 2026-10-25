import asyncio
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME

pytest.importorskip("arena_simulation_setup")

BOX_FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
PARTS = {
    "base": ((0.0, 1.0), (0.0, 0.5), (1.0, 2.0)),
    "panel": ((1.0, 1.05), (0.0, 0.5), (1.0, 2.0)),
    "tray": ((0.1, 0.9), (0.05, 0.45), (1.1, 1.3)),
    "plate": ((0.0, 1.0), (0.0, 0.5), (2.0, 2.02)),
}
ARTICULATION = {
    "links": {"cabinet": ["base"], "door": ["door"], "slider": [], "drawer": ["drawer"], "plate": ["plate"]},
    "joints": [
        {"name": "hinge", "type": "revolute", "parent": "cabinet", "child": "door", "origin": [1.0, 0.0, 1.0], "axis": [0, 0, 1], "limit": [0.0, 1.5]},
        {"name": "slide", "type": "prismatic", "parent": "cabinet", "child": "slider", "origin": [0.5, 0.25, 1.2], "axis": [1, 0, 0], "limit": [0.0, 0.3], "velocity": 0.1},
        {"name": "spin", "type": "continuous", "parent": "slider", "child": "drawer", "origin": [0.5, 0.25, 1.2], "axis": [0, 1, 0]},
        {"name": "mount", "type": "fixed", "parent": "cabinet", "child": "plate", "origin": [0.0, 0.0, 2.0]},
    ],
    "states": {"closed": {"hinge": 0.0, "slide": 0.0}, "open": {"hinge": 1.5, "slide": 0.3}},
}
LIFT = np.array([0.0, 0.0, -1.0])
BUILT_BOXES = {
    "cabinet": ((0.0, 1.0), (0.0, 0.5), (0.0, 1.0)),
    "door": ((1.0, 1.05), (0.0, 0.5), (0.0, 1.0)),
    "drawer": ((0.1, 0.9), (0.05, 0.45), (0.1, 0.3)),
    "plate": ((0.0, 1.0), (0.0, 0.5), (1.0, 1.02)),
}
VALUES = [{}, {"hinge": 0.7, "slide": 0.2, "spin": 0.4}, {"hinge": 1.5, "slide": 0.3, "spin": -2.0}]


def _mesh(name, extents):
    xs, ys, zs = extents
    points = ", ".join(f"({x}, {y}, {z})" for x in xs for y in ys for z in zs)
    return f'def Mesh "{name}"\n{{\nint[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\nint[] faceVertexIndices = {BOX_FACES}\npoint3f[] points = [{points}]\n}}\n'


def _write_cabinet(directory, articulation=ARTICULATION, root_extra="", door_extra=""):
    """Cabinet a meter off the floor: base and plate meshes, door and drawer Xforms holding one mesh each."""
    directory.mkdir(parents=True, exist_ok=True)
    body = root_extra + _mesh("base", PARTS["base"]) + _mesh("plate", PARTS["plate"]) + f'def Xform "door"\n{{\n{_mesh("panel", PARTS["panel"])}{door_extra}}}\n' + f'def Xform "drawer"\n{{\n{_mesh("tray", PARTS["tray"])}}}\n'
    (directory / "cabinet.usda").write_text(f'#usda 1.0\n(\ndefaultPrim = "Root"\nmetersPerUnit = 1\nupAxis = "Z"\n)\ndef Xform "Root"\n{{\n{body}}}\n')
    annotation = {"desc": "cabinet"}
    if articulation is not None:
        annotation["articulation"] = articulation
    (directory / ANNOTATION_NAME).write_text(yaml.safe_dump(annotation))


def _flat(box):
    return [bound for side in box for bound in side]


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def _origin(element):
    origin = element.find("origin")
    transform = np.eye(4)
    roll, pitch, yaw = (float(value) for value in origin.get("rpy").split())
    transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    transform[:3, 3] = [float(value) for value in origin.get("xyz").split()]
    return transform


def _urdf_frames(robot, values):
    """Link frames of a URDF at the joint values, by forward kinematics."""
    joints = robot.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    frames = {link.get("name"): np.eye(4) for link in robot.findall("link") if link.get("name") not in children}
    while len(frames) < len(robot.findall("link")):
        for joint in joints:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in frames or child in frames:
                continue
            motion = np.eye(4)
            value = values.get(joint.get("name"), 0.0)
            if joint.get("type") in ("revolute", "continuous"):
                motion[:3, :3] = _rotation([float(c) for c in joint.find("axis").get("xyz").split()], value)
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = np.array([float(c) for c in joint.find("axis").get("xyz").split()]) * value
            frames[child] = frames[parent] @ _origin(joint) @ motion
    return frames


def _urdf_geometry_poses(robot, values):
    """Model-frame pose of each link mesh at the joint values, the OBJ axis roll left out."""
    frames = _urdf_frames(robot, values)
    poses = {}
    for link in robot.findall("link[visual]"):
        offset = np.eye(4)
        offset[:3, 3] = [float(value) for value in link.find("visual/origin").get("xyz").split()]
        poses[link.get("name")] = frames[link.get("name")] @ offset
    return poses


def _usd_box(prim):
    from pxr import Usd, UsdGeom

    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    return _flat(zip(bound.GetMin(), bound.GetMax(), strict=True))


def _obj_points(path):
    return np.array([[float(value) for value in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")])


@pytest.fixture()
def built_cabinet(blender, tmp_path):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    source = tmp_path / "input" / "Object" / "cabinet"
    _write_cabinet(source)
    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz,sdf")
    dest = tmp_path / "output" / "Object" / "cabinet"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(source), dest)
    assert built is not None
    return built, dest


def test_build_carries_the_articulation_into_the_built_frame(built_cabinet):
    from arena_simulation_setup.utils.articulation import Articulation

    built, _ = built_cabinet
    source = Articulation.parse(ARTICULATION)
    joints = built.articulation.joint_map

    assert _flat(built.bounding_box) == pytest.approx([0.0, 1.05, 0.0, 0.5, 0.0, 1.02], abs=1e-4)
    for joint in source.joints:
        assert list(joints[joint.name].origin) == pytest.approx(list(np.array(list(joint.origin)) + LIFT), abs=1e-6)
        assert list(joints[joint.name].axis) == pytest.approx(list(joint.axis), abs=1e-6)
    assert built.articulation.states == source.states
    assert built.articulation.links == source.links
    swept = built.articulation.swept
    assert swept == built.articulation.sweep(BUILT_BOXES)
    assert all(low <= rest_low and high >= rest_high for (low, high), (rest_low, rest_high) in zip(swept, built.bounding_box, strict=True))
    assert swept[0][1] >= 1.2


def test_build_writes_a_loadable_bundle_per_link_with_meshes(built_cabinet):
    from arena_simulation_setup.utils.models.model_loader import ModelProvider_OBJ, ModelProvider_SDF, ModelProvider_USD
    from pxr import Usd, UsdGeom, UsdPhysics

    _, dest = built_cabinet

    assert sorted(path.name for path in (dest / "links").iterdir()) == ["cabinet", "door", "drawer", "plate"]
    for link, box in BUILT_BOXES.items():
        directory = dest / "links" / link
        annotation = yaml.safe_load((directory / ANNOTATION_NAME).read_text())
        assert (annotation["name"], annotation["path"]) == (link, f"Object/cabinet/links/{link}")
        assert _flat(annotation["bounding_box"]) == pytest.approx(_flat(box), abs=1e-4)
        for provider, path in ((ModelProvider_USD, f"{link}.usdz"), (ModelProvider_SDF, f"{link}.sdf/{link}.sdf"), (ModelProvider_OBJ, f"{link}.obj/{link}.obj")):
            assert asyncio.run(provider.load(directory, link, None)).path == directory / path
        stage = Usd.Stage.Open(str(directory / f"{link}.usdz"))
        meshes = [prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh)]
        assert meshes and all(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
        assert _usd_box(stage.GetDefaultPrim()) == pytest.approx(_flat(box), abs=1e-4)
    assert not (dest / "links" / "slider").exists()
    assert (dest / "cabinet.usdz").is_file() and (dest / "cabinet.sdf" / "cabinet.sdf").is_file()


def test_urdf_link_poses_follow_the_articulation(built_cabinet):
    from arena_simulation_setup.utils.articulation import matrix

    built, dest = built_cabinet
    robot = ET.parse(dest / "cabinet.urdf").getroot()

    assert robot.get("name") == "cabinet"
    assert [link.get("name") for link in robot.findall("link")] == list(built.articulation.links)
    assert [joint.get("name") for joint in robot.findall("joint")] == ["hinge", "slide", "spin", "mount"]
    assert robot.find("link[@name='slider']/visual") is None
    assert robot.find("link[@name='door']/visual/geometry/mesh").get("filename") == "links/door/door.obj/door.obj"
    assert robot.find("link[@name='door']/collision/geometry/mesh").get("filename") == "links/door/door.obj/door.obj"
    assert all(float(link.find("inertial/mass").get("value")) == 1.0 for link in robot.findall("link"))
    limits = {joint.get("name"): joint.find("limit") for joint in robot.findall("joint")}
    assert {name: None if limit is None else [float(limit.get(key)) for key in ("lower", "upper", "velocity")] for name, limit in limits.items()} == {"hinge": [0.0, 1.5, 1.0], "slide": [0.0, 0.3, 0.1], "spin": None, "mount": None}
    assert robot.find("joint[@name='mount']/axis") is None
    for values in VALUES:
        poses = _urdf_geometry_poses(robot, values)
        for link, pose in built.articulation.link_poses(values).items():
            if link in poses:
                np.testing.assert_allclose(poses[link], matrix(pose), atol=1e-6)
        assert sorted(poses) == sorted(BUILT_BOXES)


def test_urdf_visual_origin_stands_the_link_obj_in_the_model_frame(built_cabinet):
    _, dest = built_cabinet
    robot = ET.parse(dest / "cabinet.urdf").getroot()
    frames = _urdf_frames(robot, {})

    for link, box in BUILT_BOXES.items():
        points = _obj_points(dest / "links" / link / f"{link}.obj" / f"{link}.obj")
        visual = frames[link] @ _origin(robot.find(f"link[@name='{link}']/visual"))
        model = points @ visual[:3, :3].T + visual[:3, 3]
        assert _flat(zip(model.min(axis=0), model.max(axis=0), strict=True)) == pytest.approx(_flat(box), abs=1e-4)


def test_articulated_usd_resolves_its_links_and_reads_back_the_joints(built_cabinet):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    _, dest = built_cabinet
    path = dest / "cabinet.articulated.usda"
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()

    assert (UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)) == (UsdGeom.Tokens.z, 1.0)
    assert root.GetPath() == Sdf.Path("/cabinet") and root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))[2]
    for link in ("cabinet", "door", "slider", "drawer", "plate"):
        prim = root.GetChild(link)
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == 1.0
        assert prim.HasAuthoredReferences() == (link in BUILT_BOXES)
        if link in BUILT_BOXES:
            assert _usd_box(prim) == pytest.approx(_flat(BUILT_BOXES[link]), abs=1e-4)

    joints = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)}
    assert sorted(joints) == ["hinge", "mount", "slide", "spin", "world"]

    def bodies(name):
        joint = UsdPhysics.Joint(joints[name])
        return [str(target) for target in joint.GetBody0Rel().GetTargets()], [str(target) for target in joint.GetBody1Rel().GetTargets()]

    assert joints["world"].IsA(UsdPhysics.FixedJoint) and bodies("world") == ([], ["/cabinet/cabinet"])
    assert joints["mount"].IsA(UsdPhysics.FixedJoint) and bodies("mount") == (["/cabinet/cabinet"], ["/cabinet/plate"])
    expected = {
        "hinge": (UsdPhysics.RevoluteJoint, "door", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, math.degrees(1.5))),
        "slide": (UsdPhysics.PrismaticJoint, "slider", (0.5, 0.25, 0.2), (1.0, 0.0, 0.0), (0.0, 0.3)),
        "spin": (UsdPhysics.RevoluteJoint, "drawer", (0.5, 0.25, 0.2), (0.0, 1.0, 0.0), None),
    }
    for name, (schema, child, origin, axis, limit) in expected.items():
        joint = schema(joints[name])
        assert joint
        parent = "/cabinet/slider" if name == "spin" else "/cabinet/cabinet"
        assert bodies(name) == ([parent], [f"/cabinet/{child}"])
        assert joint.GetAxisAttr().Get() == UsdPhysics.Tokens.x
        assert list(joint.GetLocalPos0Attr().Get()) == pytest.approx(origin, abs=1e-6)
        assert list(joint.GetLocalPos1Attr().Get()) == pytest.approx(origin, abs=1e-6)
        for rotation in (joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()):
            assert list(Gf.Rotation(Gf.Quatd(rotation)).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))) == pytest.approx(axis, abs=1e-6)
        if limit is None:
            assert not joint.GetLowerLimitAttr().HasAuthoredValue() and not joint.GetUpperLimitAttr().HasAuthoredValue()
        else:
            assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx(limit, abs=1e-4)


def test_metadata_round_trip_keeps_the_articulation(built_cabinet):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation

    built, _ = built_cabinet
    metadata = built.as_metadata
    restored = ObjectAnnotation.from_metadata(metadata)

    assert metadata["articulated"] is True
    assert restored.articulation.unstructure() == built.articulation.unstructure()
    assert ObjectAnnotation(name="crate", path="Object/crate").as_metadata["articulated"] is False


def test_annotation_refresh_keeps_the_built_articulation(blender, tmp_path):
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    source = tmp_path / "input" / "Object" / "cabinet"
    _write_cabinet(source)
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "usdz")
    first.build()
    built = tmp_path / "output" / "Object" / "cabinet" / ANNOTATION_NAME
    before = yaml.safe_load(built.read_text())["articulation"]
    changed = {**ARTICULATION, "joints": [{**ARTICULATION["joints"][0], "limit": [0.0, 1.6]}, *ARTICULATION["joints"][1:]]}
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "tall cabinet", "articulation": changed}))
    builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    after = yaml.safe_load(built.read_text())
    assert after["desc"] == "tall cabinet"
    assert after["articulation"] == before
    assert "swept" in before


def test_annotation_refresh_warns_that_a_new_articulation_needs_a_model_build(blender, tmp_path, caplog):
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    source = tmp_path / "input" / "Object" / "cabinet"
    _write_cabinet(source, articulation=None)
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "usdz")
    first.build()
    _write_cabinet(source)
    with caplog.at_level(logging.WARNING):
        builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    assert "Object/cabinet has an articulation its last build lacks, run a model build to export its links" in caplog.text
    assert "articulation" not in yaml.safe_load((tmp_path / "output" / "Object" / "cabinet" / ANNOTATION_NAME).read_text())


def test_build_fails_an_object_whose_articulation_names_missing_nodes(blender, tmp_path, caplog):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    source = tmp_path / "input" / "Object" / "cabinet"
    _write_cabinet(source, {**ARTICULATION, "links": {**ARTICULATION["links"], "drawer": ["drawer", "lid"]}})
    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz")
    dest = tmp_path / "output" / "Object" / "cabinet"
    dest.mkdir(parents=True)
    with caplog.at_level(logging.ERROR):
        assert builder.process_entity(builder.read_annotation_file(source), dest) is None

    (message,) = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert "\n" not in message
    assert message.startswith(f"Cannot build {source}: its articulation does not fit the model: ValueError('articulation nodes lid match no object of the model")


def test_build_skips_an_object_whose_articulation_does_not_parse(tmp_path):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    source = tmp_path / "input" / "Object" / "cabinet"
    _write_cabinet(source, {**ARTICULATION, "joints": [{**ARTICULATION["joints"][0], "limit": None}]})
    annotation = ObjectDatabaseBuilder.read_annotation_file(Path(source))

    assert isinstance(annotation, ValueError)
    assert "joint 'hinge': revolute joint hinge needs a limit: [lower, upper]" in str(annotation)


def test_build_puts_each_light_on_the_link_its_node_sits_under(blender, tmp_path):
    from arena_assets.impl import converter
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    source = tmp_path / "input" / "Object" / "cabinet"
    lamp = 'def SphereLight "lamp"\n{\ndouble3 xformOp:translate = (1.02, 0.25, 1.8)\nuniform token[] xformOpOrder = ["xformOp:translate"]\nfloat inputs:radius = 0.02\nfloat inputs:intensity = 2000\n}\n'
    glow = 'def RectLight "glow"\n{\ndouble3 xformOp:translate = (0.5, 0.25, 1.9)\nuniform token[] xformOpOrder = ["xformOp:translate"]\nfloat inputs:width = 0.2\nfloat inputs:height = 0.2\nfloat inputs:intensity = 500\n}\n'
    _write_cabinet(source, root_extra=glow, door_extra=lamp)
    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz,sdf")
    dest = tmp_path / "output" / "Object" / "cabinet"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(source), dest)

    assert built is not None
    assert [(light.name, light.fixture.value, light.link) for light in built.lights] == [("light", "panel", "cabinet"), ("light_1", "bulb", "door")]
    assert built.lights[1].pose[0] == pytest.approx((1.02, 0.25, 0.8), abs=1e-3)
    assert [light["link"] for light in converter.unstructure(built)["lights"]] == ["cabinet", "door"]
