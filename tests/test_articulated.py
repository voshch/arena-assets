import logging
import math
from pathlib import Path

import numpy as np
import pytest

from arena_assets.importers.articulated import Body, Edge, assemble, merge_obj, opening_states, read_urdf, slug, unique, urdf_articulation, usd_articulation
from blender_scene import owners, scene, unmatched
from urdf_item import UP, urdf_item

pxr = pytest.importorskip("pxr")

CUBE = [(x, y, z) for x in (0.0, 10.0) for y in (0.0, 10.0) for z in (0.0, 10.0)]
FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]


def _quaternion(axis: tuple[float, float, float], degrees: float):
    half = math.radians(degrees) / 2
    return pxr.Gf.Quatf(math.cos(half), *(component * math.sin(half) for component in axis))


def articulated_stage(path: Path) -> Path:
    """A Y-up centimeter stage of rigid bodies: a hinged door, a drawer on a half-scale rail, a wheel joined child-first, a lid whose authored pose sits 30 degrees into its joint, and loose parts."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    for name, at in {"Base": (0, 0, 0), "Door": (100, 0, 0), "Rail": (0, 50, 0), "Drawer": (0, 50, 0), "Wheel": (0, 0, 80), "Lid": (0, 100, 0), "Loose": (0, 200, 0)}.items():
        body = UsdGeom.Xform.Define(stage, f"/Root/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*at))
        if name == "Rail":
            body.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, f"/Root/{name}/{name}Box")
        mesh.CreatePointsAttr(CUBE)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(FACES)
        UsdGeom.Xform.Define(stage, f"/Root/{name}/{name}Marker")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/Root/Base"))

    def joint(kind, name, body0, body1, pos0, pos1=(0, 0, 0), axis="X", limits=None, rot0=None, rot1=None):
        made = kind.Define(stage, f"/Root/Joints/{name}")
        if body0:
            made.CreateBody0Rel().SetTargets([f"/Root/{body0}"])
        made.CreateBody1Rel().SetTargets([f"/Root/{body1}"])
        made.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos0))
        made.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos1))
        if rot0 is not None:
            made.CreateLocalRot0Attr().Set(rot0)
        if rot1 is not None:
            made.CreateLocalRot1Attr().Set(rot1)
        if kind is not UsdPhysics.FixedJoint:
            made.CreateAxisAttr().Set(axis)
            if limits is not None:
                made.CreateLowerLimitAttr().Set(limits[0])
                made.CreateUpperLimitAttr().Set(limits[1])

    turn = _quaternion((0.0, 0.0, 1.0), 90.0)
    joint(UsdPhysics.RevoluteJoint, "Hinge", "Base", "Door", (100, 0, 0), axis="Y", limits=(0.0, 90.0))
    joint(UsdPhysics.PrismaticJoint, "Slide", "Rail", "Drawer", (0, 0, 0), axis="Z", limits=(0.0, 40.0))
    joint(UsdPhysics.RevoluteJoint, "Spin", "Wheel", "Base", (0, 0, 0), (0, 0, 80), axis="X", limits=(-math.inf, math.inf))
    joint(UsdPhysics.RevoluteJoint, "LidHinge", "Base", "Lid", (0, 100, 0), axis="X", limits=(0.0, 60.0), rot0=turn, rot1=turn * _quaternion((1.0, 0.0, 0.0), 30.0))
    joint(UsdPhysics.FixedJoint, "Anchor", None, "Base", (0, 0, 0))
    stage.Save()
    return path


def by_name(articulation) -> dict:
    return {joint.name: joint for joint in articulation.joints}


def test_slug_and_unique_names():
    assert [slug(name) for name in ("wheelPin_01", "E_door 3", "Door-Left", "---")] == ["wheel_pin_01", "e_door_3", "door_left", "part"]
    taken: set[str] = set()
    assert [unique("door", taken) for _ in range(3)] == ["door", "door_2", "door_3"]


def test_opening_states_close_doors_and_drawers_at_the_modeled_pose():
    from arena_simulation_setup.utils.articulation import Joint

    joints = [
        Joint(name="door", type="revolute", parent="base", child="door", limit=[-1.5, 0.0]),
        Joint(name="drawer", type="prismatic", parent="base", child="drawer", limit=[0.0, 0.3]),
        Joint(name="lid", type="revolute", parent="base", child="lid", limit=[-0.5, 0.5]),
        Joint(name="handle", type="revolute", parent="door", child="handle", limit=[0.0, 1.0]),
        Joint(name="wheel", type="continuous", parent="base", child="wheel"),
        Joint(name="sash", type="prismatic", parent="base", child="sash", limit=[0.0, 0.4]),
    ]
    labels = {"door": "E_door_3", "drawer": "drawer_2", "lid": "Lid", "handle": "door_handle_left_0", "wheel": "wheel", "sash": "translation_window"}
    assert opening_states(joints, labels) == {"closed": {"door": 0.0, "drawer": 0.0, "lid": 0.0, "sash": 0.0}, "open": {"door": -1.5, "drawer": 0.3, "lid": 0.5, "sash": 0.4}}
    assert opening_states(joints[3:5], labels) == {}
    assert opening_states(joints, labels, lower_closed=True) == {"closed": {"door": -1.5, "drawer": 0.0, "lid": -0.5, "sash": 0.0}, "open": {"door": 0.0, "drawer": 0.3, "lid": 0.5, "sash": 0.4}}


def test_assemble_turns_a_revolute_limit_half_a_turn_off_back_onto_the_modeled_pose(caplog):
    bodies = {key: Body(link=key) for key in ("body", "door")}
    edges = [Edge(name="door_hinge", type="revolute", parent="body", child="door", limit=(-3.141593, -1.22173))]
    with caplog.at_level(logging.WARNING):
        articulation = assemble(bodies, edges, "body")
    assert articulation is not None
    assert articulation.joints[0].limit == (0.0, pytest.approx(math.pi - 1.22173, abs=1e-5))
    assert articulation.states == {"closed": {"door_hinge": 0.0}, "open": {"door_hinge": pytest.approx(math.pi - 1.22173, abs=1e-5)}}
    assert "joint door_hinge has the limit [-3.14159, -1.22173] that excludes its modeled pose, it is turned by 3.14159" in caplog.text


def test_assemble_widens_a_limit_that_excludes_the_modeled_pose(caplog):
    bodies = {key: Body(link=key) for key in ("body", "door", "drawer")}
    edges = [
        Edge(name="door_hinge", type="revolute", parent="body", child="door", limit=(0.5, 1.5)),
        Edge(name="drawer_slide", type="prismatic", parent="body", child="drawer", limit=(-0.4, -0.1)),
    ]
    with caplog.at_level(logging.WARNING):
        articulation = assemble(bodies, edges, "body")
    assert articulation is not None
    assert [joint.limit for joint in articulation.joints] == [(0.0, 1.5), (-0.4, 0.0)]
    assert "joint door_hinge has the limit [0.5, 1.5] that excludes its modeled pose, it is widened to reach 0" in caplog.text
    assert "joint drawer_slide has the limit [-0.4, -0.1] that excludes its modeled pose, it is widened to reach 0" in caplog.text


def test_assemble_renames_a_joint_named_like_a_reserved_field():
    bodies = {key: Body(link=key) for key in ("body", "dial")}
    articulation = assemble(bodies, [Edge(name="Angle", type="continuous", parent="body", child="dial")], "body")
    assert articulation is not None
    assert [joint.name for joint in articulation.joints] == ["angle_2"]


def test_assemble_keeps_a_tree_hangs_loose_bodies_fixed_and_drops_loops(caplog):
    bodies = {key: Body(link=key) for key in ("a", "b", "c", "d")}
    edges = [Edge(name="ab", type="continuous", parent="a", child="b"), Edge(name="bc", type="fixed", parent="b", child="c"), Edge(name="ca", type="fixed", parent="c", child="a")]
    with caplog.at_level(logging.WARNING):
        articulation = assemble(bodies, edges, "a")
    assert articulation is not None
    assert [(joint.name, joint.parent, joint.child, str(joint.type)) for joint in articulation.joints] == [("ab", "a", "b", "continuous"), ("ca", "a", "c", "fixed"), ("d_fixed", "a", "d", "fixed")]
    assert "joint bc closes a loop" in caplog.text
    assert assemble(bodies, [Edge(name="ab", type="fixed", parent="a", child="b")], "a") is None


def test_usd_joints_land_in_the_model_frame_in_meters_and_radians(tmp_path):
    articulation = usd_articulation(articulated_stage(tmp_path / "stage.usda"))
    assert articulation is not None
    assert articulation.links == {"base": ["Base"], "door": ["Door"], "rail": ["Rail"], "drawer": ["Drawer"], "wheel": ["Wheel"], "lid": ["Lid"], "loose": ["Loose"]}
    joints = by_name(articulation)
    assert set(joints) == {"hinge", "slide", "spin", "lid_hinge", "loose_fixed", "rail_fixed"}
    assert (joints["slide"].parent, joints["rail_fixed"].parent) == ("rail", "base")
    hinge, slide, spin, lid = joints["hinge"], joints["slide"], joints["spin"], joints["lid_hinge"]
    assert (str(hinge.type), hinge.parent, hinge.child) == ("revolute", "base", "door")
    assert list(hinge.origin) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert list(hinge.axis) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert hinge.limit == pytest.approx((0.0, math.pi / 2))
    assert (str(slide.type), list(slide.axis), slide.limit) == ("prismatic", pytest.approx([0.0, -1.0, 0.0], abs=1e-6), pytest.approx((0.0, 0.4)))
    assert (str(spin.type), spin.parent, spin.child) == ("continuous", "base", "wheel")
    assert list(spin.axis) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)
    assert list(spin.origin) == pytest.approx([0.0, -0.8, 0.0], abs=1e-6)
    assert list(lid.axis) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert lid.limit == pytest.approx((-math.pi / 6, math.pi / 6), abs=1e-5)
    assert (str(joints["loose_fixed"].type), joints["loose_fixed"].parent) == ("fixed", "base")
    assert articulation.states == {"closed": {"hinge": 0.0, "slide": 0.0, "lid_hinge": 0.0}, "open": {"hinge": pytest.approx(math.pi / 2), "slide": pytest.approx(0.4), "lid_hinge": pytest.approx(math.pi / 6)}}


def test_usd_joint_physics_and_link_materials_land_in_si_units_at_the_joint_values(tmp_path):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    path = articulated_stage(tmp_path / "stage.usda")
    stage = Usd.Stage.Open(str(path))
    hinge = stage.GetPrimAtPath("/Root/Joints/Hinge")
    hinge.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.3)
    hinge.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(1000.0)
    drive = UsdPhysics.DriveAPI.Apply(hinge, "angular")
    drive.CreateStiffnessAttr(2.0)
    drive.CreateDampingAttr(0.5)
    drive.CreateTargetPositionAttr(45.0)
    slide = stage.GetPrimAtPath("/Root/Joints/Slide")
    slide.CreateAttribute("physxJointAxis:linear:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(300.0)
    UsdPhysics.DriveAPI.Apply(slide, "linear").CreateDampingAttr(4.0)
    lid = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath("/Root/Joints/LidHinge"), "angular")
    lid.CreateStiffnessAttr(1.0)
    lid.CreateTargetPositionAttr(30.0)
    surface = UsdShade.Material.Define(stage, "/Root/Looks/Rubber")
    api = UsdPhysics.MaterialAPI.Apply(surface.GetPrim())
    api.CreateStaticFrictionAttr(0.9)
    api.CreateDynamicFrictionAttr(0.7)
    api.CreateRestitutionAttr(0.2)
    collider = stage.GetPrimAtPath("/Root/Door/DoorBox")
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(surface, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.Save()
    articulation = usd_articulation(path)
    assert articulation is not None
    joints = by_name(articulation)
    gain = 1e-4 * 180.0 / math.pi
    hinge_dynamics = joints["hinge"].dynamics
    assert (joints["hinge"].role, hinge_dynamics.friction_coefficient, hinge_dynamics.friction) == ("door", pytest.approx(0.3), None)
    assert (hinge_dynamics.armature, hinge_dynamics.damping) == (pytest.approx(0.1), pytest.approx(0.5 * gain))
    assert hinge_dynamics.spring is None
    slide_dynamics = joints["slide"].dynamics
    assert (joints["slide"].role, slide_dynamics.friction, slide_dynamics.damping, slide_dynamics.spring) == ("drawer", pytest.approx(3.0), pytest.approx(4.0), None)
    assert (joints["lid_hinge"].dynamics.spring.stiffness, joints["lid_hinge"].dynamics.spring.rest) == (pytest.approx(gain), 0.0)
    assert joints["spin"].dynamics is None
    assert {link: material.unstructure() for link, material in articulation.materials.items()} == {"door": {"static_friction": pytest.approx(0.9), "dynamic_friction": pytest.approx(0.7), "restitution": pytest.approx(0.2)}}


def test_a_turned_limit_carries_its_spring_rest_along(caplog):
    bodies = {key: Body(link=key) for key in ("body", "door")}
    edges = [Edge(name="door_hinge", type="revolute", parent="body", child="door", limit=(-3.141593, -1.22173), dynamics={"damping": 0.2, "spring": {"stiffness": 1.0, "rest": -3.041593}})]
    articulation = assemble(bodies, edges, "body")
    assert articulation is not None
    dynamics = articulation.joints[0].dynamics
    assert (dynamics.damping, dynamics.spring.rest) == (0.2, pytest.approx(0.1, abs=1e-5))


def test_urdf_joint_dynamics_and_roles_reach_the_articulation(tmp_path):
    path = urdf_item(tmp_path / "item")
    path.write_text(path.read_text().replace('<limit lower="0" upper="1.57"/>', '<limit lower="0" upper="1.57"/><dynamics damping="0.3" friction="0.2"/>'))
    articulation = urdf_articulation(read_urdf(path), {"link_0": ["link_0"], "link_1": ["link_1"]}, {"link_0": "door"})
    assert articulation is not None
    door = by_name(articulation)["joint_0"]
    assert (door.role, door.dynamics.unstructure()) == ("door", {"friction": 0.2, "damping": 0.3})


def test_usd_stage_without_moving_joints_has_no_articulation(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(tmp_path / "rigid.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    for name in ("A", "B"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Root/{name}").GetPrim())
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Root/Glue")
    fixed.CreateBody0Rel().SetTargets(["/Root/A"])
    fixed.CreateBody1Rel().SetTargets(["/Root/B"])
    stage.Save()
    assert usd_articulation(tmp_path / "rigid.usda") is None


def test_blender_loads_the_usd_bodies_under_their_link_nodes_at_the_joint_origins(blender_python, tmp_path):
    path = articulated_stage(tmp_path / "stage.usda")
    articulation = usd_articulation(path)
    assert articulation is not None
    objects = scene(blender_python, path)
    raw = articulation.unstructure()
    assert unmatched(objects, raw) == []
    assert owners(objects, raw) == {"BaseBox": "base", "DoorBox": "door", "RailBox": "rail", "DrawerBox": "drawer", "WheelBox": "wheel", "LidBox": "lid", "LooseBox": "loose"}
    joints = by_name(articulation)
    assert objects["DoorMarker"]["location"] == pytest.approx(list(joints["hinge"].origin), abs=1e-6)
    assert objects["WheelMarker"]["location"] == pytest.approx(list(joints["spin"].origin), abs=1e-6)


def test_urdf_tree_lands_in_the_model_frame(tmp_path):
    urdf = read_urdf(urdf_item(tmp_path / "item"))
    assert urdf.root == "base"
    assert urdf.links["link_1"][:3, :3] == pytest.approx(UP, abs=1e-9)
    articulation = urdf_articulation(urdf, {"link_0": ["link_0"], "link_1": ["link_1"]}, {"link_0": "door"})
    assert articulation is not None
    door = by_name(articulation)["joint_0"]
    assert (door.parent, door.child, door.limit) == ("link_1", "link_0", pytest.approx((0.0, 1.57)))
    assert list(door.origin) == pytest.approx([0.0, -0.5, 0.0], abs=1e-9)
    assert list(door.axis) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert articulation.states == {"closed": {"joint_0": 0.0}, "open": {"joint_0": pytest.approx(1.57)}}
    assert [visual.link for visual in urdf.visuals] == ["link_1", "link_0", "link_0"]


def test_merged_obj_names_its_links_and_keeps_each_files_materials_apart(tmp_path):
    urdf_path = urdf_item(tmp_path / "item")
    urdf = read_urdf(urdf_path)
    out = tmp_path / "item" / "merged.obj"
    merge_obj([(visual.link, urdf_path.parent / visual.mesh, visual.transform) for visual in urdf.visuals], out)
    text = out.read_text()
    assert [line for line in text.splitlines() if line.startswith(("o ", "usemtl"))] == ["o link_1", "usemtl m0_wood", "o link_0", "usemtl m1_wood", "o link_0", "usemtl m0_wood"]
    faces = [line for line in text.splitlines() if line.startswith("f ")]
    assert faces[:2] == ["f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/3/1"]
    assert faces[-1] == "f 9/7/3 11/9/3 12/9/3"
    assert out.with_suffix(".mtl").read_text() == "newmtl m0_wood\nillum 2\nKd 0.5 0.4 0.3\nmap_Kd images/wood.png\nnewmtl m1_wood\nillum 2\nKd 0.5 0.4 0.3\nmap_Kd images/wood.png\n"


def test_blender_loads_the_merged_obj_as_link_objects_in_the_model_frame(blender_python, tmp_path):
    urdf_path = urdf_item(tmp_path / "item")
    urdf = read_urdf(urdf_path)
    out = tmp_path / "item" / "merged.obj"
    merge_obj([(visual.link, urdf_path.parent / visual.mesh, visual.transform) for visual in urdf.visuals], out)
    articulation = urdf_articulation(urdf, {"link_0": ["link_0"], "link_1": ["link_1"]})
    assert articulation is not None
    objects = scene(blender_python, out)
    raw = articulation.unstructure()
    assert unmatched(objects, raw) == []
    assert sorted(owners(objects, raw).values()) == ["link_0", "link_0", "link_1"]
    assert [value for entry in objects.values() for value in entry["metallic"]] == [0.0, 0.0, 0.0]
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.4, 0.0], [0.0, 0.4, -0.1]]) @ UP.T
    assert sum(objects["link_1"]["box"], []) == pytest.approx([bound for low, high in zip(points.min(axis=0), points.max(axis=0), strict=True) for bound in (low, high)], abs=1e-6)


def test_opening_states_read_side_prefixed_parts_covers_hinges_and_skip_outdoor():
    from arena_simulation_setup.utils.articulation import Joint

    joints = [
        Joint(name="left", type="revolute", parent="body", child="left", limit=[-2.27, 0.0]),
        Joint(name="down", type="prismatic", parent="body", child="down", limit=[-0.200001, -1e-06]),
        Joint(name="table", type="revolute", parent="body", child="table", limit=[0.0, 1.0]),
        Joint(name="cover", type="revolute", parent="body", child="cover", limit=[0.0, 0.785]),
        Joint(name="cabinet_hinge_0", type="revolute", parent="body", child="panel", limit=[-2.3, 0.0]),
        Joint(name="flat_rotating_lid_0", type="revolute", parent="body", child="swing", limit=[-1.5708, 1.2]),
    ]
    labels = {"left": "revolute_joint_fridge_left fridge_e_leftdoor_84", "down": "fridge_e_leftdrawer_54", "table": "outdoor_table", "cover": "E_cover_1", "cabinet_hinge_0": "cabinet_hinge_0", "flat_rotating_lid_0": "flat_rotating_lid_0"}
    assert opening_states(joints, labels) == {
        "closed": {"left": 0.0, "down": -1e-06, "cover": 0.0, "cabinet_hinge_0": 0.0, "flat_rotating_lid_0": 0.0},
        "open": {"left": -2.27, "down": -0.200001, "cover": 0.785, "cabinet_hinge_0": -2.3, "flat_rotating_lid_0": -1.5708},
    }
