"""Articulations of staged source models in their model frame, read from UsdPhysics stages and URDF trees."""

from __future__ import annotations

import collections
import math
import os
import re
import shutil
import tempfile
import typing
import urllib.error
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode, replacing
from arena_assets.utils.logging import get_logger

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.articulation import Articulation, Joint
    from pxr import Usd, UsdGeom

logger = get_logger("import.articulated")

Vector = tuple[float, float, float]

DOMAIN = "Common"
ASSET_TYPE = "Object"
FORMATS = "usdz,sdf,fbx"
BLENDER_NAME = 63
GATED = (401, 403)
TOLERANCE = 1e-4
UNLIMITED = 1e30
Y_UP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
"""Y-up coordinates to the Z-up model frame, as Blender loads a Y-up USD stage, glTF or OBJ file."""
AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
DOFS = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
MAPS = ("map_", "bump", "disp", "decal", "refl", "norm")

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_OTHER = re.compile(r"[^a-z0-9]+")
_OPENING = re.compile(r"(?:^|_)(?:left|right|top|bottom|upper|lower|front|back|up|down)?(?:door|drawer|lid|cover|hatch|flap|window|hinge)s?(?=_|\d|$)")
_OPERATING = re.compile(r"(?:^|_)(?:handle|knob|button|switch|latch|lock)s?(?=_|\d|$)")


def slug(name: str) -> str:
    """Link or joint name of an upstream name: snake case in [a-z0-9_], `part` when nothing is left."""
    return _OTHER.sub("_", _CAMEL.sub("_", name).lower()).strip("_") or "part"


def unique(name: str, taken: set[str]) -> str:
    """Name, or name with the first free `_<n>` suffix, recorded as taken."""
    found = name
    n = 2
    while found in taken:
        found = f"{name}_{n}"
        n += 1
    taken.add(found)
    return found


def blender_name(name: str) -> str:
    """Object name Blender gives a source node of this name."""
    return name[:BLENDER_NAME]


@attrs.frozen
class Body:
    """One upstream rigid part: its link name, the source nodes whose meshes it takes and a label."""

    link: str
    nodes: tuple[str, ...] = ()
    label: str = ""


@attrs.frozen
class Edge:
    """One upstream joint between two bodies, origin and axis in the model frame, limit in radians or meters."""

    name: str
    type: str
    parent: str
    child: str
    origin: Vector = (0.0, 0.0, 0.0)
    axis: Vector = (0.0, 0.0, 1.0)
    limit: tuple[float, float] | None = None
    label: str = ""

    def reversed(self) -> Edge:
        """The same joint with parent and child swapped, the axis turned so a joint value means the same pose."""
        return attrs.evolve(self, parent=self.child, child=self.parent, axis=typing.cast(Vector, tuple(-component for component in self.axis)))


def opening_states(joints: Iterable[Joint], labels: Mapping[str, str], lower_closed: bool = False) -> dict[str, dict[str, float]]:
    """`closed` and `open` over the door, drawer, lid, cover, hatch, flap, window and hinge joints: closed at the modeled pose and open at the far limit end, or closed at the lower end when lower_closed."""
    closed: dict[str, float] = {}
    opened: dict[str, float] = {}
    for joint in joints:
        label = slug(labels.get(joint.name, ""))
        if joint.limit is None or not _OPENING.search(label) or _OPERATING.search(label):
            continue
        low, high = joint.limit
        if (lower_closed and high - low > TOLERANCE) or abs(low) <= TOLERANCE < abs(high):
            closed[joint.name], opened[joint.name] = low, high
        elif abs(high) <= TOLERANCE < abs(low):
            closed[joint.name], opened[joint.name] = high, low
        elif low < -TOLERANCE and high > TOLERANCE:
            closed[joint.name], opened[joint.name] = 0.0, high if high >= -low - TOLERANCE else low
    return {"closed": closed, "open": opened} if closed else {}


def reaching(edge: Edge) -> tuple[float, float] | None:
    """The limit of a revolute or prismatic edge widened to reach the modeled pose, which upstream limits sometimes exclude."""
    if edge.type not in ("revolute", "prismatic") or edge.limit is None:
        return None
    low, high = edge.limit
    if low <= TOLERANCE and high >= -TOLERANCE:
        return edge.limit
    logger.warning("joint %s has the limit [%g, %g] that excludes its modeled pose, it is widened to reach 0", edge.name, low, high)
    return (min(low, 0.0), max(high, 0.0))


def assemble(bodies: Mapping[str, Body], edges: Sequence[Edge], root: str, lower_closed: bool = False) -> Articulation | None:
    """The link tree grown from root, unreached bodies fixed to root, None when no joint moves."""
    from arena_simulation_setup.utils.articulation import Articulation, Joint

    taken: set[str] = set()
    names = {key: unique(slug(bodies[key].link), taken) for key in [root, *(key for key in bodies if key != root)]}
    adjacent: dict[str, list[tuple[int, bool]]] = collections.defaultdict(list)
    for n, edge in enumerate(edges):
        adjacent[edge.parent].append((n, True))
        adjacent[edge.child].append((n, False))
    reached = {root}
    used: set[int] = set()
    tree: list[Edge] = []

    def grow(start: str) -> None:
        queue = collections.deque([start])
        while queue:
            key = queue.popleft()
            for n, forward in adjacent[key]:
                if n in used:
                    continue
                used.add(n)
                edge = edges[n] if forward else edges[n].reversed()
                if edge.child in reached:
                    logger.warning("joint %s closes a loop between %s and %s, a link tree keeps it out", edge.name, names[edge.parent], names[edge.child])
                    continue
                reached.add(edge.child)
                tree.append(edge)
                queue.append(edge.child)

    grow(root)
    children = {edge.child for edge in edges}
    for key in sorted(bodies, key=lambda key: key in children):
        if key not in reached:
            tree.append(Edge(name=f"{names[key]}_fixed", type="fixed", parent=root, child=key))
            reached.add(key)
            grow(key)
    joint_names: set[str] = set()
    joints = [
        Joint(
            name=unique(slug(edge.name), joint_names),
            type=edge.type,
            parent=names[edge.parent],
            child=names[edge.child],
            origin=edge.origin,
            axis=edge.axis,
            limit=reaching(edge),
        )
        for edge in tree
    ]
    if not any(joint.movable for joint in joints):
        return None
    labels = {joint.name: f"{edge.label} {edge.name} {bodies[edge.child].label} {bodies[edge.child].link}" for joint, edge in zip(joints, tree, strict=True)}
    return Articulation(links={names[key]: list(bodies[key].nodes) for key in names}, joints=joints, states=opening_states(joints, labels, lower_closed))


def refused(error: urllib.error.HTTPError, repo: str, source: str) -> OSError:
    """The one-line error for a gated Hugging Face dataset the token cannot read, else the error itself."""
    if error.code not in GATED:
        return error
    return PermissionError(f"{source} is gated and the Hugging Face token here has no access to it, request access at https://huggingface.co/datasets/{repo} and set HF_TOKEN to a token that has it")


def build(stage: Callable[[Path], Path], cache: Path, out: Path, name: str) -> str:
    """Stage one object in a scratch dataset under cache and build it into out, return its relpath."""
    target = out / DOMAIN / ASSET_TYPE / name
    cache.mkdir(parents=True, exist_ok=True)
    dataset = Path(tempfile.mkdtemp(prefix="stage-", dir=cache))
    try:
        with replacing(target):
            stage(dataset)
            builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
            builder.enable("formats", FORMATS)
            builder.enable("previews")
            builder.build()
            if not (target / ANNOTATION_NAME).is_file():
                raise RuntimeError(f"object build failed for {name}, the build log above names the step")
    finally:
        shutil.rmtree(dataset)
    return str(Path(DOMAIN) / ASSET_TYPE / name)


def write_articulation(annotation: Path, articulation: Articulation | None) -> None:
    """Add an articulation to a staged object annotation, nothing for a rigid object."""
    if articulation is None:
        return
    with open(annotation) as f:
        raw = yaml.safe_load(f)
    raw["articulation"] = articulation.unstructure()
    with open(annotation, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def _quaternion(w: float, x: float, y: float, z: float) -> np.ndarray:
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _frame(position: Iterable[float], rotation: np.ndarray) -> np.ndarray:
    frame = np.eye(4)
    frame[:3, :3] = rotation
    frame[:3, 3] = list(position)
    return frame


def _unit(vector: np.ndarray) -> Vector:
    return typing.cast(Vector, tuple(float(component) for component in vector / np.linalg.norm(vector)))


def _rotation(matrix: np.ndarray) -> np.ndarray:
    """Rotation part of a linear map, its scale and shear taken out."""
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    return u @ vt


def _angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    """Angle of a rotation about an axis, the rotation's component about other axes ignored."""
    sine = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) / 2.0
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return math.atan2(float(sine @ axis), cosine)


def stage_frame(stage: Usd.Stage) -> np.ndarray:
    """Stage world coordinates to the model frame Blender loads the stage in: meters, Z up."""
    from pxr import UsdGeom

    frame = np.diag([UsdGeom.GetStageMetersPerUnit(stage)] * 3 + [1.0])
    return Y_UP @ frame if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else frame


def _world(cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    return np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T


def _body(prim: Usd.Prim) -> Usd.Prim:
    """Nearest prim at or above prim with a rigid body, prim itself when there is none."""
    from pxr import UsdPhysics

    found = prim
    while found.IsValid() and not found.IsPseudoRoot():
        if found.HasAPI(UsdPhysics.RigidBodyAPI):
            return found
        found = found.GetParent()
    return prim


def _usd_kind(prim: Usd.Prim) -> tuple[str, str, tuple[float | None, float | None]]:
    """Joint type, axis token and raw limits of a UsdPhysics joint."""
    from pxr import UsdPhysics

    path = prim.GetPath()
    if prim.IsA(UsdPhysics.FixedJoint):
        return "fixed", "Z", (None, None)
    if prim.IsA(UsdPhysics.RevoluteJoint):
        revolute = UsdPhysics.RevoluteJoint(prim)
        return "revolute", revolute.GetAxisAttr().Get(), (revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get())
    if prim.IsA(UsdPhysics.PrismaticJoint):
        prismatic = UsdPhysics.PrismaticJoint(prim)
        return "prismatic", prismatic.GetAxisAttr().Get(), (prismatic.GetLowerLimitAttr().Get(), prismatic.GetUpperLimitAttr().Get())
    if prim.GetTypeName() != "PhysicsJoint":
        raise ValueError(f"joint {path} is a {prim.GetTypeName()}, an articulation keeps revolute, prismatic, fixed and one-axis joints")
    free: list[tuple[str, tuple[float | None, float | None]]] = []
    for dof in DOFS:
        if not prim.HasAPI(UsdPhysics.LimitAPI, dof):
            free.append((dof, (None, None)))
            continue
        limit = UsdPhysics.LimitAPI(prim, dof)
        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
        if low is None or high is None or low <= high:
            free.append((dof, (low, high)))
    if not free:
        return "fixed", "Z", (None, None)
    if len(free) > 1:
        raise ValueError(f"joint {path} frees {', '.join(dof for dof, _ in free)}, an articulation joint moves along or about one axis")
    dof, limits = free[0]
    return ("prismatic" if dof.startswith("trans") else "revolute"), dof[-1], limits


def _limited(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and abs(value) < UNLIMITED


def _usd_edge(prim: Usd.Prim, stage: Usd.Stage, cache: UsdGeom.XformCache, model: np.ndarray, joint_name: Callable[[str], str]) -> Edge | None:
    """A UsdPhysics joint as an edge between body paths, None for a joint that anchors one body to the world."""
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint(prim)
    targets = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
    if not targets[0] or not targets[1]:
        return None
    kind, token, (low, high) = _usd_kind(prim)
    frames = []
    for n, target in enumerate(targets):
        position = (joint.GetLocalPos0Attr() if n == 0 else joint.GetLocalPos1Attr()).Get()
        rotation = (joint.GetLocalRot0Attr() if n == 0 else joint.GetLocalRot1Attr()).Get()
        local = _frame(position if position is not None else (0.0, 0.0, 0.0), np.eye(3) if rotation is None else _quaternion(rotation.GetReal(), *rotation.GetImaginary()))
        frames.append(model @ _world(cache, stage.GetPrimAtPath(target[0])) @ local)
    bodies = [_body(stage.GetPrimAtPath(target[0])) for target in targets]
    local_axis = np.array(AXES[str(token)])
    rotation0, rotation1 = _rotation(frames[0]), _rotation(frames[1])
    meters = float(np.linalg.norm(model[:3, 0]))
    axis = rotation0 @ local_axis
    rest = 0.0
    if kind == "revolute":
        rest = _angle(rotation0.T @ rotation1, local_axis)
        if not (_limited(low) and _limited(high)):
            kind = "continuous"
        elif typing.cast(float, low) > typing.cast(float, high):
            kind = "fixed"
    elif kind == "prismatic":
        rest = float((frames[1][:3, 3] - frames[0][:3, 3]) @ axis)
        if not (_limited(low) and _limited(high)):
            raise ValueError(f"prismatic joint {prim.GetPath()} has no limit, an articulation needs one")
    limit = None
    if kind == "revolute":
        limit = (math.radians(typing.cast(float, low)) - rest, math.radians(typing.cast(float, high)) - rest)
    elif kind == "prismatic":
        limit = (typing.cast(float, low) * meters - rest, typing.cast(float, high) * meters - rest)
    return Edge(
        name=joint_name(prim.GetName()),
        type=kind,
        parent=str(bodies[0].GetPath()),
        child=str(bodies[1].GetPath()),
        origin=typing.cast(Vector, tuple(float(component) for component in frames[0][:3, 3])),
        axis=_unit(axis),
        limit=limit,
        label=prim.GetName(),
    )


def usd_articulation(path: Path, link_name: Callable[[str], str] = slug, joint_name: Callable[[str], str] = slug) -> Articulation | None:
    """The UsdPhysics joints of a composed stage in the frame Blender loads it in, one link per rigid body, None when no joint moves."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    model = stage_frame(stage)
    cache = UsdGeom.XformCache()
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    edges = [edge for prim in prims if prim.IsA(UsdPhysics.Joint) and (edge := _usd_edge(prim, stage, cache, model, joint_name)) is not None]
    if not edges:
        return None
    found = {str(prim.GetPath()): prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    for edge in edges:
        for key in (edge.parent, edge.child):
            found.setdefault(key, stage.GetPrimAtPath(key))
    named: dict[str, list[str]] = collections.defaultdict(list)
    for key, prim in found.items():
        named[blender_name(prim.GetName())].append(key)
    if shared := sorted(name for name, keys in named.items() if len(keys) > 1):
        raise ValueError(f"bodies {', '.join(named[shared[0]])} share the name {shared[0]}, which the builder matches links by")
    bodies = {key: Body(link=link_name(prim.GetName()), nodes=(blender_name(prim.GetName()),), label=prim.GetName()) for key, prim in found.items()}
    marked = [key for key, prim in found.items() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    children = {edge.child for edge in edges}
    degree = collections.Counter(key for edge in edges for key in (edge.parent, edge.child))
    root = marked[0] if marked else max((key for key in bodies if key not in children), key=lambda key: degree[key], default=edges[0].parent)
    return assemble(bodies, edges, root)


def _origin(element: ET.Element | None) -> np.ndarray:
    """Transform of a URDF origin element: translation xyz after fixed-axis roll, pitch, yaw."""
    if element is None:
        return np.eye(4)
    x, y, z = (float(value) for value in (element.get("xyz") or "0 0 0").split())
    roll, pitch, yaw = (float(value) for value in (element.get("rpy") or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    return _frame((x, y, z), rotation)


@attrs.frozen(eq=False)
class Visual:
    """One mesh a URDF link shows, its file as the URDF names it and its placement in the model frame."""

    link: str
    mesh: str
    transform: np.ndarray


@attrs.frozen(eq=False)
class Urdf:
    """A URDF's link frames at all joint values 0, its joints as edges and its link meshes."""

    root: str
    links: dict[str, np.ndarray]
    edges: list[Edge]
    visuals: list[Visual]


URDF_TYPES = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "fixed": "fixed", "floating": "fixed", "planar": "fixed"}


def read_urdf(path: Path) -> Urdf:
    """Links, joints and visuals of a URDF in the frame of its root link."""
    robot = ET.parse(path).getroot()
    names = [link.get("name", "") for link in robot.findall("link")]
    joints = robot.findall("joint")
    children = {_end(path, joint, "child") for joint in joints}
    roots = [name for name in names if name not in children]
    if len(roots) != 1:
        raise ValueError(f"{path.name} has {len(roots)} root links ({', '.join(roots)}), a URDF tree has one")
    placed = {roots[0]: np.eye(4)}
    pending = list(joints)
    edges: list[Edge] = []
    while pending:
        ready = [joint for joint in pending if _end(path, joint, "parent") in placed]
        if not ready:
            raise ValueError(f"{path.name}: joints {', '.join(joint.get('name', '') for joint in pending)} hang on no placed link")
        for joint in ready:
            pending.remove(joint)
            parent, child = _end(path, joint, "parent"), _end(path, joint, "child")
            name, kind = joint.get("name", child), joint.get("type", "")
            if kind not in URDF_TYPES:
                raise ValueError(f"{path.name}: joint {name} has type {kind!r}, known: {', '.join(URDF_TYPES)}")
            if URDF_TYPES[kind] != kind:
                logger.warning("%s: %s joint %s is kept as fixed", path.name, kind, name)
            placed[child] = placed[parent] @ _origin(joint.find("origin"))
            limit_element = joint.find("limit")
            limit = None
            kind = URDF_TYPES[kind]
            if kind in ("revolute", "prismatic"):
                if limit_element is None or limit_element.get("lower") is None or limit_element.get("upper") is None:
                    if kind == "prismatic":
                        raise ValueError(f"{path.name}: prismatic joint {name} has no limit, an articulation needs one")
                    kind = "continuous"
                else:
                    limit = (float(limit_element.get("lower", "0")), float(limit_element.get("upper", "0")))
            axis = joint.find("axis")
            local_axis = np.array([float(value) for value in ((axis.get("xyz") if axis is not None else None) or "1 0 0").split()])
            frame_of = placed[child]
            scale = float(np.linalg.norm(frame_of[:3, :3] @ (local_axis / np.linalg.norm(local_axis))))
            if limit is not None and kind == "prismatic":
                limit = (limit[0] * scale, limit[1] * scale)
            edges.append(
                Edge(
                    name=name,
                    type=kind,
                    parent=parent,
                    child=child,
                    origin=typing.cast(Vector, tuple(float(component) for component in frame_of[:3, 3])),
                    axis=_unit(frame_of[:3, :3] @ local_axis),
                    limit=limit,
                )
            )
    visuals = []
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                logger.warning("%s: link %s shows a visual that is not a mesh, it is left out", path.name, link.get("name"))
                continue
            scale = np.diag([*(float(value) for value in (mesh.get("scale") or "1 1 1").split()), 1.0])
            visuals.append(Visual(link=link.get("name", ""), mesh=mesh.get("filename", ""), transform=placed[link.get("name", "")] @ _origin(visual.find("origin")) @ scale))
    return Urdf(root=roots[0], links=placed, edges=edges, visuals=visuals)


def _end(path: Path, joint: ET.Element, tag: str) -> str:
    """Link a URDF joint names as its parent or child."""
    element = joint.find(tag)
    link = element.get("link") if element is not None else None
    if not link:
        raise ValueError(f"{path.name}: joint {joint.get('name')} names no {tag} link")
    return link


def urdf_articulation(urdf: Urdf, nodes: Mapping[str, Sequence[str]], labels: Mapping[str, str] | None = None, lower_closed: bool = False) -> Articulation | None:
    """The articulation of a read URDF whose link meshes carry the given node names, None when no joint moves."""
    bodies = {name: Body(link=name, nodes=tuple(nodes.get(name, ())), label=(labels or {}).get(name, "")) for name in urdf.links}
    return assemble(bodies, urdf.edges, urdf.root, lower_closed)


def _index(token: str, count: int) -> int:
    value = int(token)
    return value + count + 1 if value < 0 else value


def _texture_line(line: str, source: Path, target: Path) -> str:
    """An MTL map line with its file path made relative to the target directory."""
    words = line.split()
    if not words or not words[0].lower().startswith(MAPS) or len(words) < 2:
        return line
    path = Path(words[-1].replace("\\", "/"))
    resolved = path if path.is_absolute() else source / path
    words[-1] = Path(os.path.relpath(resolved, target)).as_posix()
    return " ".join(words) + "\n"


def merge_obj(parts: Sequence[tuple[str, Path, np.ndarray]], out: Path) -> None:
    """Write OBJ parts as one Y-up OBJ and MTL, each an object named by its node and placed by its transform, reflective illum 3 materials as plain illum 2 ones."""
    vertices: list[str] = []
    textures: list[str] = []
    normals: list[str] = []
    body: list[str] = []
    materials: list[str] = []
    named: dict[Path, str] = {}
    for node, mesh, transform in parts:
        if mesh not in named:
            named[mesh] = f"m{len(named)}"
            for library in _libraries(mesh):
                materials.extend(_renamed_mtl(library, named[mesh], out.parent))
        prefix = named[mesh]
        normal_map = np.linalg.inv(transform[:3, :3]).T
        base = (len(vertices), len(textures), len(normals))
        count = [0, 0, 0]
        body.append(f"o {node}\n")
        for line in mesh.read_text(errors="replace").splitlines():
            words = line.split()
            if not words:
                continue
            key = words[0]
            if key == "v":
                point = transform @ np.array([float(words[1]), float(words[2]), float(words[3]), 1.0])
                vertices.append(" ".join(["v", repr(float(point[0])), repr(float(point[2])), repr(float(-point[1])), *words[4:]]) + "\n")
                count[0] += 1
            elif key == "vt":
                textures.append(line + "\n")
                count[1] += 1
            elif key == "vn":
                normal = normal_map @ np.array([float(words[1]), float(words[2]), float(words[3])])
                normal /= np.linalg.norm(normal) or 1.0
                normals.append(" ".join(["vn", repr(float(normal[0])), repr(float(normal[2])), repr(float(-normal[1]))]) + "\n")
                count[2] += 1
            elif key == "f":
                corners = []
                for corner in words[1:]:
                    indices = corner.split("/")
                    corners.append("/".join("" if not token else str(_index(token, count[n]) + base[n]) for n, token in enumerate(indices)))
                body.append("f " + " ".join(corners) + "\n")
            elif key == "usemtl" and len(words) > 1:
                body.append(f"usemtl {prefix}_{' '.join(words[1:])}\n")
    library = out.with_suffix(".mtl")
    library.write_text("".join(materials))
    out.write_text(f"mtllib {library.name}\n" + "".join(vertices) + "".join(textures) + "".join(normals) + "".join(body))


def _libraries(mesh: Path) -> list[Path]:
    found = []
    for line in mesh.read_text(errors="replace").splitlines():
        if line.startswith("mtllib "):
            library = mesh.parent / line[len("mtllib ") :].strip()
            if library.is_file():
                found.append(library)
    return found


def _renamed_mtl(library: Path, prefix: str, target: Path) -> list[str]:
    lines = []
    for line in library.read_text(errors="replace").splitlines(keepends=True):
        words = line.split()
        if words and words[0] == "newmtl" and len(words) > 1:
            lines.append(f"newmtl {prefix}_{' '.join(words[1:])}\n")
        elif words == ["illum", "3"]:
            lines.append("illum 2\n")
        else:
            lines.append(_texture_line(line if line.endswith("\n") else line + "\n", library.parent, target))
    return lines
