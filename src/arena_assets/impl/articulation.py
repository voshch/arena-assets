"""Articulated object bundles: one static object bundle per link, the URDF and the UsdPhysics stage over them, structured by arena_simulation_setup.utils.articulation only when an annotation has an articulation."""

from __future__ import annotations

import math
import shutil
import typing
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

from arena_assets.impl import ANNOTATION_NAME, collider

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.articulation import Articulation, Box, Dynamics, Material
else:
    Articulation = typing.Any
    Dynamics = typing.Any
    Material = typing.Any

LINKS = "links"
URDF_SUFFIX = ".urdf"
USD_SUFFIX = ".articulated.usda"
LINK_MASS = 1.0
EFFORT = 100.0
MIN_INERTIA = 1e-4
OBJ_TO_MODEL_RPY = (math.pi / 2, 0.0, 0.0)
"""Blender writes OBJ files Y up, this roll stands them back up in the Z-up model frame."""


class ArticulationError(Exception):
    """A user-facing reason an articulated object was not built."""


def structure_articulation(value: Articulation | Mapping[str, typing.Any] | None) -> Articulation | None:
    if value is None:
        return None
    from arena_simulation_setup.utils.articulation import structure_articulation as structure

    return structure(value)


def unstructure_articulation(value: Articulation | None) -> dict[str, object] | None:
    return None if value is None else value.unstructure()


def link_dir(bundle: Path, link: str) -> Path:
    return bundle / LINKS / link


def link_formats(formats: Iterable[str]) -> list[str]:
    """Formats a link bundle is exported in: the build's plus obj, which the URDF names."""
    return sorted({*formats, "obj"})


def urdf_mesh(link: str) -> str:
    return f"{LINKS}/{link}/{link}.obj/{link}.obj"


def clear(bundle: Path, name: str) -> None:
    """Delete the link bundles, URDF and articulated USD an earlier build left in bundle."""
    shutil.rmtree(bundle / LINKS, ignore_errors=True)
    (bundle / f"{name}{URDF_SUFFIX}").unlink(missing_ok=True)
    (bundle / f"{name}{USD_SUFFIX}").unlink(missing_ok=True)


def _number(value: float) -> str:
    return f"{round(float(value), 9) + 0.0:.9g}"


def _vector(values: Iterable[float]) -> str:
    return " ".join(_number(value) for value in values)


def link_frames(articulation: Articulation) -> dict[str, np.ndarray]:
    """Model-frame origin of each URDF link frame at rest: the model origin for the root, the parent joint origin for the others."""
    frames = {link: np.zeros(3) for link in articulation.links}
    for joint in articulation.joints:
        frames[joint.child] = np.array(list(joint.origin), dtype=np.float64)
    return frames


def _inertial(link: ET.Element, box: Box | None, frame: np.ndarray) -> None:
    inertial = ET.SubElement(link, "inertial")
    if box is None:
        center = np.zeros(3)
        moments = (MIN_INERTIA,) * 3
    else:
        low, high = np.array(box, dtype=np.float64).T
        center = (low + high) / 2 - frame
        x, y, z = (high - low) ** 2
        moments = tuple(max(LINK_MASS / 12 * (a + b), MIN_INERTIA) for a, b in ((y, z), (x, z), (x, y)))
    ET.SubElement(inertial, "origin", xyz=_vector(center), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=_number(LINK_MASS))
    ET.SubElement(inertial, "inertia", ixx=_number(moments[0]), ixy="0", ixz="0", iyy=_number(moments[1]), iyz="0", izz=_number(moments[2]))


def write_urdf(path: Path, name: str, articulation: Articulation, boxes: Mapping[str, Box]) -> None:
    """URDF of the articulation over the link OBJ meshes, link frames at their parent joints, a 1 kg box inertia per link, authored dry friction and damping as joint dynamics."""
    from arena_simulation_setup.utils.articulation import JointType

    frames = link_frames(articulation)
    robot = ET.Element("robot", name=name)
    for link_name in articulation.links:
        link = ET.SubElement(robot, "link", name=link_name)
        box = boxes.get(link_name)
        _inertial(link, box, frames[link_name])
        if box is None:
            continue
        for tag in ("visual", "collision"):
            element = ET.SubElement(link, tag)
            ET.SubElement(element, "origin", xyz=_vector(-frames[link_name]), rpy=_vector(OBJ_TO_MODEL_RPY))
            ET.SubElement(ET.SubElement(element, "geometry"), "mesh", filename=urdf_mesh(link_name))
    for joint in articulation.joints:
        element = ET.SubElement(robot, "joint", name=joint.name, type=str(joint.type))
        ET.SubElement(element, "origin", xyz=_vector(frames[joint.child] - frames[joint.parent]), rpy="0 0 0")
        ET.SubElement(element, "parent", link=joint.parent)
        ET.SubElement(element, "child", link=joint.child)
        if joint.type is not JointType.FIXED:
            ET.SubElement(element, "axis", xyz=_vector(joint.axis))
        if joint.limit is not None:
            ET.SubElement(element, "limit", lower=_number(joint.limit[0]), upper=_number(joint.limit[1]), effort=_number(EFFORT), velocity=_number(joint.speed))
        dynamics = joint.dynamics
        if dynamics is not None and (values := {key: _number(value) for key, value in (("damping", dynamics.damping), ("friction", dynamics.friction)) if value is not None}):
            ET.SubElement(element, "dynamics", values)
    tree = ET.ElementTree(robot)
    ET.indent(tree, "  ")
    tree.write(path, encoding="unicode", xml_declaration=True)


def _free(name: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    while name in taken:
        name += "_"
    return name


def axis_rotation(axis: Iterable[float]) -> Gf.Quatf:
    """Rotation taking +X onto the axis, the joint frame of a UsdPhysics joint whose axis is X."""
    quat = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(*axis)).GetQuat()
    return Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary()))


def write_usd(path: Path, name: str, articulation: Articulation, references: Mapping[str, str]) -> None:
    """Z-up meters stage with an articulation root over one rigid body per link, the links with meshes referencing their link USD and bound to their physics materials, authored joint dynamics as PhysX joint attributes and drives."""
    from arena_simulation_setup.utils.articulation import JointType

    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, Sdf.Path.absoluteRootPath.AppendChild(Tf.MakeValidIdentifier(name))).GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    bodies: dict[str, Sdf.Path] = {}
    for link in articulation.links:
        prim = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild(Tf.MakeValidIdentifier(link))).GetPrim()
        if link in references:
            prim.GetReferences().AddReference(references[link])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(LINK_MASS)
        bodies[link] = prim.GetPath()
        if (material := articulation.materials.get(link)) is not None:
            _bind_material(stage, prim, material)
    scope = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild(_free("joints", (body.name for body in bodies.values())))).GetPath()
    names = [Tf.MakeValidIdentifier(joint.name) for joint in articulation.joints]
    UsdPhysics.FixedJoint.Define(stage, scope.AppendChild(_free("world", names))).CreateBody1Rel().SetTargets([bodies[articulation.root]])
    for joint, prim_name in zip(articulation.joints, names, strict=True):
        joint_path = scope.AppendChild(prim_name)
        limit = joint.limit
        if joint.type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            usd_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            usd_joint.CreateAxisAttr(UsdPhysics.Tokens.x)
            if limit is not None:
                usd_joint.CreateLowerLimitAttr(math.degrees(limit[0]))
                usd_joint.CreateUpperLimitAttr(math.degrees(limit[1]))
        elif joint.type is JointType.PRISMATIC:
            usd_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            usd_joint.CreateAxisAttr(UsdPhysics.Tokens.x)
            if limit is not None:
                usd_joint.CreateLowerLimitAttr(limit[0])
                usd_joint.CreateUpperLimitAttr(limit[1])
        else:
            usd_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        usd_joint.CreateBody0Rel().SetTargets([bodies[joint.parent]])
        usd_joint.CreateBody1Rel().SetTargets([bodies[joint.child]])
        origin = Gf.Vec3f(*joint.origin)
        rotation = axis_rotation(joint.axis)
        usd_joint.CreateLocalPos0Attr(origin)
        usd_joint.CreateLocalPos1Attr(origin)
        usd_joint.CreateLocalRot0Attr(rotation)
        usd_joint.CreateLocalRot1Attr(rotation)
        if joint.dynamics is not None:
            _author_dynamics(usd_joint.GetPrim(), joint.dynamics, joint.type is not JointType.PRISMATIC)
    stage.GetRootLayer().Save()


def _float(prim: Usd.Prim, name: str, value: float) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))


def _author_dynamics(prim: Usd.Prim, dynamics: Dynamics, angular: bool) -> None:
    """PhysX joint friction and armature, axis friction effort, and a drive carrying damping and spring, angular gains per degree."""
    instance = "angular" if angular else "linear"
    if dynamics.friction_coefficient is not None or dynamics.armature is not None:
        prim.AddAppliedSchema("PhysxJointAPI")
        if dynamics.friction_coefficient is not None:
            _float(prim, "physxJoint:jointFriction", dynamics.friction_coefficient)
        if dynamics.armature is not None:
            _float(prim, "physxJoint:armature", dynamics.armature)
    if dynamics.friction is not None:
        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{instance}")
        _float(prim, f"physxJointAxis:{instance}:staticFrictionEffort", dynamics.friction)
        _float(prim, f"physxJointAxis:{instance}:dynamicFrictionEffort", dynamics.friction)
    if dynamics.damping is None and dynamics.spring is None:
        return
    per = math.pi / 180.0 if angular else 1.0
    drive = UsdPhysics.DriveAPI.Apply(prim, instance)
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateDampingAttr(0.0 if dynamics.damping is None else dynamics.damping * per)
    drive.CreateStiffnessAttr(0.0 if dynamics.spring is None else dynamics.spring.stiffness * per)
    if dynamics.spring is not None:
        drive.CreateTargetPositionAttr(math.degrees(dynamics.spring.rest) if angular else dynamics.spring.rest)


def _bind_material(stage: Usd.Stage, link: Usd.Prim, material: Material) -> None:
    """A physics material under the link carrying its friction and restitution, bound to the link for physics."""
    surface = UsdShade.Material.Define(stage, link.GetPath().AppendChild("physics_material"))
    api = UsdPhysics.MaterialAPI.Apply(surface.GetPrim())
    if material.static_friction is not None:
        api.CreateStaticFrictionAttr(material.static_friction)
    if material.dynamic_friction is not None:
        api.CreateDynamicFrictionAttr(material.dynamic_friction)
    if material.restitution is not None:
        api.CreateRestitutionAttr(material.restitution)
    UsdShade.MaterialBindingAPI.Apply(link).Bind(surface, UsdShade.Tokens.strongerThanDescendants, "physics")


def write_link_annotation(bundle: Path, link: str, path: str, box: Box) -> None:
    (bundle / ANNOTATION_NAME).write_text(yaml.safe_dump({"name": link, "path": path, "bounding_box": [list(side) for side in box]}))


def finish(bundle: Path, name: str, path: str, articulation: Articulation, boxes: Mapping[str, Box]) -> list[Path]:
    """Annotate and collide the exported link bundles, then write the URDF and, when the links have a USD model, the articulated USD; return the files written."""
    references: dict[str, str] = {}
    for link, box in boxes.items():
        directory = link_dir(bundle, link)
        write_link_annotation(directory, link, f"{path}/{LINKS}/{link}", box)
        usds = collider.usd_models(directory, link)
        for usd in usds:
            collider.collide_visuals(usd)
        if usds:
            references[link] = f"./{usds[0].relative_to(bundle).as_posix()}"
    written = [bundle / f"{name}{URDF_SUFFIX}"]
    write_urdf(written[0], name, articulation, boxes)
    if references:
        written.append(bundle / f"{name}{USD_SUFFIX}")
        write_usd(written[1], name, articulation, references)
    return written
