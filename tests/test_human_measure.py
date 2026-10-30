import attrs
import numpy as np
import pytest

from arena_assets.impl import Face
from arena_assets.impl.build.HumanDatabaseBuilder import HumanAnnotation, measure_body
from arena_assets.utils.human import Legs, PelvisClearance, Skin, legs, pelvis_clearance, proportions, read_skin, seated_clearance


def rigid(dominant: list[str]) -> dict[str, np.ndarray]:
    return {joint: np.array([1.0 if name == joint else 0.0 for name in dominant]) for joint in set(dominant)}


def body(coat: bool = False, hem: bool = False) -> Skin:
    """Pelvis 0.9 m up, hip joints 0.02 m lower and 0.01 m behind it, buttocks 0.08 m below and 0.1 m behind the hip joints moving with the thighs."""
    joints = {"Hips": np.array([0.0, 0.0, 0.9]), "LeftUpLeg": np.array([-0.01, 0.1, 0.88]), "RightUpLeg": np.array([-0.01, -0.1, 0.88])}
    points = [(-0.11, 0.1, 0.8), (0.05, 0.1, 0.6), (-0.11, -0.1, 0.8), (0.0, 0.0, 1.0)]
    dominant = ["LeftUpLeg", "LeftUpLeg", "RightUpLeg", "Hips"]
    if coat:
        points.append((-0.4, 0.0, 0.2))
        dominant.append("Hips")
    if hem:
        points.append((-0.1, 0.1, 0.7))
        dominant.append("LeftUpLeg")
    return Skin(joints=joints, points=np.array(points), dominant=dominant, weights=rigid(dominant))


def test_clearance_is_the_thigh_behind_the_hip_joint_seen_from_the_pelvis():
    clearance = pelvis_clearance(body(), np.array([1.0, 0.0]))
    assert (clearance.sit, clearance.lie) == pytest.approx((0.02 + 0.1, 0.01 + 0.1))


def test_seated_clearance_swings_the_buttocks_under_the_hips_with_the_thighs():
    seated = dict(seated_clearance(body(), np.array([1.0, 0.0])))
    assert seated[0.0] == pytest.approx(0.02 + 0.08)
    assert seated[round(np.pi / 2, 4)] == pytest.approx(0.02 + 0.1)


def test_buttocks_blended_with_the_pelvis_sink_less_than_the_thighs_swing():
    skin = body()
    half = {joint: weight * (0.5 if joint != "Hips" else 1.0) for joint, weight in skin.weights.items()}
    half["Hips"] = half["Hips"] + np.array([0.5, 0.5, 0.5, 0.0])
    blended = dict(seated_clearance(Skin(joints=skin.joints, points=skin.points, dominant=skin.dominant, weights=half), np.array([1.0, 0.0])))
    rigid = dict(seated_clearance(skin, np.array([1.0, 0.0])))
    level = round(np.pi / 2, 4)
    assert rigid[level] > blended[level] > rigid[0.0]


def test_hems_below_the_buttocks_do_not_count():
    assert pelvis_clearance(body(hem=True), np.array([1.0, 0.0])) == pelvis_clearance(body(), np.array([1.0, 0.0]))


def test_sit_at_interpolates_the_seated_table_and_falls_back_to_sit():
    clearance = PelvisClearance(sit=0.12, lie=0.11, seated=((0.0, 0.2), (2.0, 0.1)))
    assert clearance.sit_at(1.0) == pytest.approx(0.15)
    assert clearance.sit_at(3.0) == pytest.approx(0.1)
    assert PelvisClearance(sit=0.12, lie=0.11).sit_at(1.0) == 0.12


def test_seated_table_survives_the_annotation_round_trip():
    clearance = pelvis_clearance(body(), np.array([1.0, 0.0]))
    assert PelvisClearance(**{**attrs.asdict(clearance), "seated": [list(pair) for pair in clearance.seated]}) == clearance


def test_clearance_holds_plain_floats_for_yaml():
    clearance = pelvis_clearance(body(), np.array([1.0, 0.0]))
    assert type(clearance.sit) is float and type(clearance.lie) is float


def test_garments_hanging_from_the_pelvis_do_not_count():
    assert pelvis_clearance(body(coat=True), np.array([1.0, 0.0])) == pelvis_clearance(body(), np.array([1.0, 0.0]))


def test_clearance_follows_the_facing():
    turned = body()
    rotate = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = Skin(joints={k: rotate @ v for k, v in turned.joints.items()}, points=turned.points @ rotate.T, dominant=turned.dominant, weights=turned.weights)
    assert pelvis_clearance(turned, np.array([0.0, 1.0])) == pelvis_clearance(body(), np.array([1.0, 0.0]))


def test_clearance_needs_both_thighs():
    skin = body()
    with pytest.raises(ValueError, match="RightUpLeg"):
        pelvis_clearance(Skin(joints=skin.joints, points=skin.points[:2], dominant=skin.dominant[:2], weights={k: v[:2] for k, v in skin.weights.items()}), np.array([1.0, 0.0]))


DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_geometries><geometry id="body-mesh"><mesh>
    <source id="body-positions"><float_array id="pa" count="9">0 0 1  0.1 0 0  0 0.1 0</float_array>
      <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
    <vertices id="body-vertices"><input semantic="POSITION" source="#body-positions"/></vertices>
  </mesh></geometry></library_geometries>
  <library_controllers><controller id="body-skin"><skin source="#body-mesh">
    <bind_shape_matrix>1 0 0 0  0 1 0 0  0 0 1 0.5  0 0 0 1</bind_shape_matrix>
    <source id="joints"><Name_array id="ja" count="2">Hips LeftUpLeg</Name_array></source>
    <source id="binds"><float_array id="ba" count="32">1 0 0 0  0 1 0 0  0 0 1 -0.9  0 0 0 1   1 0 0 0  0 1 0 -0.1  0 0 1 -0.8  0 0 0 1</float_array>
      <technique_common><accessor source="#ba" count="2" stride="16"/></technique_common></source>
    <source id="weights"><float_array id="wa" count="3">1 0.7 0.3</float_array>
      <technique_common><accessor source="#wa" count="3" stride="1"/></technique_common></source>
    <joints><input semantic="JOINT" source="#joints"/><input semantic="INV_BIND_MATRIX" source="#binds"/></joints>
    <vertex_weights count="3"><input semantic="JOINT" source="#joints" offset="0"/><input semantic="WEIGHT" source="#weights" offset="1"/>
      <vcount>1 2 1</vcount><v>0 0  0 1 1 2  1 0</v></vertex_weights>
  </skin></controller></library_controllers>
</COLLADA>
"""


def test_read_skin_takes_joints_from_inverse_binds_and_points_from_the_bind_shape(tmp_path):
    (tmp_path / "body.dae").write_text(DAE)
    skin = read_skin(tmp_path / "body.dae")
    assert skin.joints["Hips"] == pytest.approx([0.0, 0.0, 0.9])
    assert skin.joints["LeftUpLeg"] == pytest.approx([0.0, 0.1, 0.8])
    assert skin.points == pytest.approx(np.array([[0.0, 0.0, 1.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]))
    assert skin.dominant == ["Hips", "Hips", "LeftUpLeg"]
    assert skin.weights["Hips"] == pytest.approx([1.0, 0.7, 0.0])
    assert skin.weights["LeftUpLeg"] == pytest.approx([0.0, 0.3, 1.0])


def test_legs_run_from_the_pelvis_through_both_leg_chains_to_the_soles():
    joints = {"Hips": np.array([0.0, 0.0, 0.92])}
    for side, y in (("Left", 0.1), ("Right", -0.1)):
        joints |= {f"{side}UpLeg": np.array([0.0, y, 0.9]), f"{side}Leg": np.array([0.03, y, 0.5]), f"{side}Foot": np.array([0.0, y, 0.08])}
    skin = Skin(joints=joints, points=np.array([(0.1, 0.1, -0.01), (0.0, 0.0, 1.6)]), dominant=["LeftFoot", "Head"])
    assert legs(skin) == Legs(hip=0.02, thigh=round(float(np.hypot(0.03, 0.4)), 3), shin=round(float(np.hypot(0.03, 0.42)), 3), ankle=0.09)


def test_build_leaves_an_unmeasurable_body_unannotated(tmp_path):
    for annotation in (HumanAnnotation(name="Nurse01", path="Human/Nurse01", face=Face.POS_X), HumanAnnotation(name="Odd", path="Human/Odd", face=Face.XY)):
        measure_body(annotation, tmp_path / "missing.dae")
        assert annotation.pelvis_clearance is None and annotation.legs is None


def arms_skin() -> Skin:
    """T-posed arms: shoulders 0.5 m above the pelvis and 0.2 m out, upper arm 0.3 m, forearm 0.25 m, the hand reaching 0.2 m."""
    joints = {"Hips": np.array([0.0, 0.0, 1.0]), "LeftUpLeg": np.array([0.0, 0.1, 0.95]), "RightUpLeg": np.array([0.0, -0.1, 0.95]), "Head": np.array([0.0, 0.0, 1.72])}
    points, dominant = [], []
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        for name, y in (("Arm", 0.2), ("ForeArm", 0.5), ("Hand", 0.75)):
            joints[f"{side}{name}"] = np.array([0.0, sign * y, 1.5])
        points += [(0.0, sign * 0.85, 1.5), (0.0, sign * 0.95, 1.5)]
        dominant += [f"{side}Hand", f"{side}HandFinger1"]
    return Skin(joints=joints, points=np.array(points), dominant=dominant)


def test_proportions_measure_each_arm_and_the_trunk_in_the_bind_pose():
    p = proportions(arms_skin())
    assert (p.shoulders, p.shoulder_width, p.hip_width, p.head) == pytest.approx((0.5, 0.2, 0.1, 0.22))
    assert (p.upperarm, p.forearm, p.hand) == pytest.approx((0.3, 0.25, 0.2))


def test_proportions_need_hand_vertices():
    skin = arms_skin()
    with pytest.raises(ValueError, match="LeftHand"):
        proportions(Skin(joints=skin.joints, points=skin.points[2:], dominant=skin.dominant[2:]))
