"""Body measurements read off a skinned human model (COLLADA), seated clearances from its buttocks skinned through hip flexion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import attrs
import numpy as np

PELVIS_JOINT = "Hips"  # cmu_mb rig, the skeleton's root joint
THIGH_JOINTS = ("LeftUpLeg", "RightUpLeg")  # cmu_mb rig, the hip joints the thighs hang from
LEG_CHAINS = (("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot"))  # cmu_mb rig, hip, knee, ankle
ARM_CHAINS = (("LeftArm", "LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm", "RightHand"))  # cmu_mb rig, shoulder, elbow, wrist
HEAD_JOINT = "Head"  # cmu_mb rig, the base of the skull
SEATED_FLEXIONS = tuple(float(np.radians(degrees)) for degrees in range(0, 141, 10))  # rad, hip flexions the seated clearance is measured at
BUTTOCK_BAND = 0.12  # m below the hip joints
BUTTOCK_REACH = 0.3  # m behind the hip joints
BUTTOCK_WIDTH = 0.12  # m beyond each hip joint sideways


@attrs.define(frozen=True)
class PelvisClearance:
    """Height of the pelvis joint above the surface a body rests on, in m: seated with thighs level, lying on its back, seated per hip flexion (rad, m)."""

    sit: float = attrs.field(converter=float)
    lie: float = attrs.field(converter=float)
    seated: tuple[tuple[float, float], ...] = attrs.field(default=(), converter=lambda pairs: tuple((float(flexion), float(height)) for flexion, height in pairs))

    def sit_at(self, flexion: float) -> float:
        """Seated clearance at a hip flexion in rad, interpolated, sit without a seated table."""
        if not self.seated:
            return self.sit
        flexions, heights = zip(*self.seated, strict=True)
        return float(np.interp(flexion, flexions, heights))


@attrs.define(frozen=True)
class Legs:
    """Leg geometry of a standing body in m: pelvis joint above the hip joints, thigh, shin, ankle joint above the soles."""

    hip: float = attrs.field(converter=float)
    thigh: float = attrs.field(converter=float)
    shin: float = attrs.field(converter=float)
    ankle: float = attrs.field(converter=float)


@attrs.define(frozen=True)
class Proportions:
    """Trunk and arm geometry of a standing body in m: shoulder joints above the pelvis joint, half the shoulder and hip
    joint spans, upper arm, forearm, wrist to fingertips, shoulder joints to the base of the skull."""

    shoulders: float = attrs.field(converter=float)
    shoulder_width: float = attrs.field(converter=float)
    hip_width: float = attrs.field(converter=float)
    upperarm: float = attrs.field(converter=float)
    forearm: float = attrs.field(converter=float)
    hand: float = attrs.field(converter=float)
    head: float = attrs.field(converter=float)


@attrs.define(frozen=True)
class Skin:
    """Bind pose of a skinned mesh: joint positions by name, vertex positions, the joint that moves each vertex most and every joint's per-vertex weight."""

    joints: dict[str, np.ndarray]
    points: np.ndarray
    dominant: list[str]
    weights: dict[str, np.ndarray] = attrs.field(factory=dict)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def seated_clearance(skin: Skin, forward: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Pelvis joint height above the lowest buttock vertex per hip flexion of a body facing `forward`, both thighs flexed by linear blend skinning."""
    ahead = np.array([forward[0], forward[1], 0.0])
    side = np.cross(np.array([0.0, 0.0, 1.0]), ahead)
    hips = np.mean([skin.joints[name] for name in THIGH_JOINTS], axis=0)
    offset = skin.points - hips
    width = 0.5 * abs(float((skin.joints[THIGH_JOINTS[0]] - skin.joints[THIGH_JOINTS[1]]) @ side)) + BUTTOCK_WIDTH
    buttock = (offset[:, 2] >= -BUTTOCK_BAND) & (offset[:, 2] <= 0.0) & (offset @ ahead < 0.0) & (offset @ ahead > -BUTTOCK_REACH) & (np.abs(offset @ side) < width)
    if not buttock.any():
        raise ValueError("no buttock vertex behind the hip joints")
    points = skin.points[buttock]
    missing = np.zeros(len(skin.points))
    legs = [(skin.joints[chain[0]], sum((skin.weights.get(joint, missing) for joint in chain), missing)[buttock]) for chain in LEG_CHAINS]
    pelvis = float(skin.joints[PELVIS_JOINT][2])
    seated = []
    for flexion in SEATED_FLEXIONS:
        rotation = _rotation(side, -flexion)
        moved = points + sum(weight[:, None] * ((points - hip) @ rotation.T + hip - points) for hip, weight in legs)
        seated.append((round(flexion, 4), round(pelvis - float(moved[:, 2].min()), 3)))
    return tuple(seated)


def pelvis_clearance(skin: Skin, forward: np.ndarray) -> PelvisClearance:
    """Clearances of a body facing `forward` (a horizontal unit vector), lying from its bind pose, seated skinned through hip flexion."""
    pelvis = skin.joints[PELVIS_JOINT]
    dominant = np.asarray(skin.dominant)
    lie = []
    for name in THIGH_JOINTS:
        hip = skin.joints[name]
        thigh = skin.points[dominant == name]
        if len(thigh) == 0:
            raise ValueError(f"no vertex follows {name}")
        behind = float(((hip[:2] - thigh[:, :2]) @ forward).max())
        lie.append(float((pelvis[:2] - hip[:2]) @ forward) + behind)
    seated = seated_clearance(skin, forward)
    level = PelvisClearance(sit=0.0, lie=0.0, seated=seated).sit_at(np.pi / 2.0)
    return PelvisClearance(sit=round(level, 3), lie=round(max(lie), 3), seated=seated)


def legs(skin: Skin) -> Legs:
    """Leg geometry of a body in its bind pose, both legs averaged, the soles at its lowest vertex."""
    floor = float(skin.points[:, 2].min())
    hip, knee, ankle = (np.mean([skin.joints[chain[i]] for chain in LEG_CHAINS], axis=0) for i in range(3))
    return Legs(
        hip=round(float(skin.joints[PELVIS_JOINT][2] - hip[2]), 3),
        thigh=round(float(np.linalg.norm(hip - knee)), 3),
        shin=round(float(np.linalg.norm(knee - ankle)), 3),
        ankle=round(float(ankle[2]) - floor, 3),
    )


def proportions(skin: Skin) -> Proportions:
    """Trunk and arm geometry of a body in its bind pose, both sides averaged; the hand reaches its farthest vertex."""
    joints = skin.joints
    left, right = ([joints[name] for name in chain] for chain in ARM_CHAINS)
    length = lambda i: float(np.mean([np.linalg.norm(arm[i] - arm[i + 1]) for arm in (left, right)]))  # noqa: E731
    shoulder_z = float(np.mean([left[0][2], right[0][2]]))
    dominant = np.asarray(skin.dominant, dtype=str)
    hands = []
    for side, chain in zip(("Left", "Right"), ARM_CHAINS, strict=True):
        hand = skin.points[np.char.startswith(dominant, f"{side}Hand") | np.char.startswith(dominant, f"{side}Finger") | (dominant == f"{side[0]}Thumb")]
        if len(hand) == 0:
            raise ValueError(f"no vertex follows {chain[2]}")
        hands.append(float(np.linalg.norm(hand - joints[chain[2]], axis=1).max()))
    return Proportions(
        shoulders=round(shoulder_z - float(joints[PELVIS_JOINT][2]), 3),
        shoulder_width=round(float(np.linalg.norm(joints[ARM_CHAINS[0][0]] - joints[ARM_CHAINS[1][0]])) / 2, 3),
        hip_width=round(float(np.linalg.norm(joints[THIGH_JOINTS[0]] - joints[THIGH_JOINTS[1]])) / 2, 3),
        upperarm=round(length(0), 3),
        forearm=round(length(1), 3),
        hand=round(float(np.mean(hands)), 3),
        head=round(float(joints[HEAD_JOINT][2]) - shoulder_z, 3),
    )


def read_skin(dae: Path) -> Skin:
    """The bind pose of every skin controller in a COLLADA file, merged."""
    root = ET.parse(dae).getroot()
    ns = root.tag[: root.tag.index("}") + 1]
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}

    def source(ref: str | None) -> ET.Element:
        if ref is None or (element := by_id.get(ref.lstrip("#"))) is None:
            raise ValueError(f"{dae}: missing COLLADA source {ref}")
        return element

    def floats(ref: str | None) -> np.ndarray:
        element = source(ref)
        accessor = element.find(f"{ns}technique_common/{ns}accessor")
        stride = int(accessor.get("stride", "1")) if accessor is not None else 1
        return np.array((element.findtext(f"{ns}float_array") or "").split(), dtype=float).reshape(-1, stride)

    joints: dict[str, np.ndarray] = {}
    points: list[np.ndarray] = []
    dominant: list[str] = []
    influences: list[dict[str, float]] = []
    for skin in root.iter(f"{ns}skin"):
        bind_shape = np.array((skin.findtext(f"{ns}bind_shape_matrix") or "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1").split(), dtype=float).reshape(4, 4)
        inputs = {element.get("semantic"): element.get("source") for element in skin.iterfind(f"{ns}joints/{ns}input")}
        names = (source(inputs.get("JOINT")).findtext(f"{ns}Name_array") or "").split()
        inverse_binds = floats(inputs.get("INV_BIND_MATRIX")).reshape(-1, 4, 4)
        for name, inverse_bind in zip(names, inverse_binds, strict=True):
            joints.setdefault(name, np.linalg.inv(inverse_bind)[:3, 3])

        weights_element = skin.find(f"{ns}vertex_weights")
        if weights_element is None:
            raise ValueError(f"{dae}: skin without vertex_weights")
        offsets = {element.get("semantic"): (int(element.get("offset", "0")), element.get("source")) for element in weights_element.iterfind(f"{ns}input")}
        joint_offset, weight_offset, weight_source = offsets["JOINT"][0], offsets["WEIGHT"][0], offsets["WEIGHT"][1]
        weights = floats(weight_source).reshape(-1)
        counts = np.array((weights_element.findtext(f"{ns}vcount") or "").split(), dtype=int)
        pairs = np.array((weights_element.findtext(f"{ns}v") or "").split(), dtype=int).reshape(-1, max(joint_offset, weight_offset) + 1)
        start = 0
        for count in counts:
            vertex = pairs[start : start + count]
            dominant.append(names[vertex[np.argmax(weights[vertex[:, weight_offset]]), joint_offset]] if count else "")
            influence: dict[str, float] = {}
            for joint, weight in zip(vertex[:, joint_offset], vertex[:, weight_offset], strict=True):
                influence[names[joint]] = influence.get(names[joint], 0.0) + float(weights[weight])
            influences.append(influence)
            start += count

        mesh = source(skin.get("source"))
        position = mesh.find(f"{ns}mesh/{ns}vertices/{ns}input[@semantic='POSITION']")
        vertices = floats(position.get("source") if position is not None else None)
        points.append((np.c_[vertices, np.ones(len(vertices))] @ bind_shape.T)[:, :3])
    if not points:
        raise ValueError(f"{dae}: no skin controller")
    merged = np.concatenate(points)
    if len(merged) != len(dominant):
        raise ValueError(f"{dae}: {len(dominant)} skin weights for {len(merged)} vertices")
    by_joint = {name: np.zeros(len(merged)) for name in {name for influence in influences for name in influence}}
    for index, influence in enumerate(influences):
        for name, weight in influence.items():
            by_joint[name][index] = weight
    return Skin(joints=joints, points=merged, dominant=dominant, weights=by_joint)
