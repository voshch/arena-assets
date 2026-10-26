from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import logging
import math
import os
import types
import typing
import xml.sax.saxutils
from pathlib import Path

import bpy
import mathutils
import numpy as np
from PIL import UnidentifiedImageError

from . import ModelFormat, sdf_model
from .CoordinateSystem import CoordinateSystem
from .geom import BoundingBox
from .io_utils import capture_all_output
from .lights import SourceLight, SourceMaterial, UsdLight, matching_prim
from .links import LinkBox, LinkObjects, Side, base_name, link_of, node_links
from .textures import TextureLimit, limited

logger = logging.getLogger("ModelConverter")

GLOW_MATERIALS = "glow_materials"


def _principled(material: bpy.types.Material) -> bpy.types.Node | None:
    if not material.use_nodes or material.node_tree is None:
        return None
    return next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)


def _is_emissive(material: bpy.types.Material) -> bool:
    principled = _principled(material)
    if principled is None:
        return False
    color, strength = principled.inputs["Emission Color"], principled.inputs["Emission Strength"]
    return not color.is_linked and not strength.is_linked and any(color.default_value[:3]) and strength.default_value > 0.0


def _material_indices(mesh: bpy.types.Mesh) -> np.ndarray:
    indices = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", indices)
    return indices


def _material_faces(obj: bpy.types.Object) -> dict[str, np.ndarray]:
    """Face masks of a mesh object by name of the material the faces use."""
    indices = _material_indices(obj.data)
    slots = obj.material_slots
    faces: dict[str, np.ndarray] = {}
    for index in np.unique(indices):
        material = slots[min(int(index), len(slots) - 1)].material if slots else None
        if material is not None:
            faces[material.name] = faces.get(material.name, False) | (indices == index)
    return faces


def _face_points(obj: bpy.types.Object, faces: np.ndarray) -> np.ndarray:
    """World-space corner positions of the masked faces of a mesh object."""
    mesh = obj.data
    totals = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vertices)
    points = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", points)
    matrix = np.array(obj.matrix_world)
    return points.reshape(-1, 3)[vertices[np.repeat(faces, totals)]] @ matrix[:3, :3].T + matrix[:3, 3]


def _keep_faces(mesh: bpy.types.Mesh, faces: np.ndarray) -> None:
    """Delete every face of a mesh outside the mask and every material slot left without faces."""
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.delete(bm, geom=[face for face, keep in zip(bm.faces, faces, strict=True) if not keep], context="FACES")
        bm.to_mesh(mesh)
    finally:
        bm.free()
    used = set(np.minimum(_material_indices(mesh), len(mesh.materials) - 1).tolist())
    for index in reversed(range(len(mesh.materials))):
        if index not in used:
            mesh.materials.pop(index=index)


def usd_lights(path: str) -> list[UsdLight]:
    """Rect, disk and sphere UsdLux lights of a USD stage."""
    bpy.utils.expose_bundled_modules()
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(path)
    found = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.IsA(UsdLux.RectLight):
            rect = UsdLux.RectLight(prim)
            size = {"width": rect.GetWidthAttr().Get(), "height": rect.GetHeightAttr().Get()}
        elif prim.IsA(UsdLux.DiskLight):
            size = {"radius": UsdLux.DiskLight(prim).GetRadiusAttr().Get()}
        elif prim.IsA(UsdLux.SphereLight):
            size = {"radius": UsdLux.SphereLight(prim).GetRadiusAttr().Get()}
        else:
            continue
        light = UsdLux.LightAPI(prim)
        cone = UsdLux.ShapingAPI(prim).GetShapingConeAngleAttr().Get()
        found.append(
            UsdLight(
                names=tuple(str(prim.GetPath()).strip("/").split("/")),
                kind=prim.GetTypeName(),
                **{key: float(value) for key, value in size.items() if value is not None},
                intensity=float(light.GetIntensityAttr().Get() or 0.0),
                exposure=float(light.GetExposureAttr().Get() or 0.0),
                normalize=bool(light.GetNormalizeAttr().Get()),
                cct_K=float(light.GetColorTemperatureAttr().Get() or 6500.0) if light.GetEnableColorTemperatureAttr().Get() else None,
                cone_deg=2.0 * float(cone) if cone is not None and cone < 90.0 else None,
            )
        )
    return found


def _usd_light(light: SourceLight, prim: UsdLight, scale: mathutils.Vector) -> SourceLight:
    """A light with the photometry, shape and cone of the UsdLux prim it was imported from, sized by its world scale."""
    sx, sy, sz = (abs(component) for component in scale)
    if prim.kind == "RectLight":
        area = face = prim.width * sx * prim.height * sy
        kind, shape = "AREA", "RECTANGLE"
    elif prim.kind == "DiskLight":
        area = face = math.pi * prim.radius**2 * sx * sy
        kind, shape = "AREA", "DISK"
    else:
        radius = prim.radius * (sx * sy * sz) ** (1.0 / 3.0)
        area, face = 4.0 * math.pi * radius**2, math.pi * radius**2
        kind, shape = "POINT", ""
    radiance = prim.intensity * 2.0**prim.exposure / (area if prim.normalize and area > 0.0 else 1.0)
    return dataclasses.replace(
        light,
        type=kind if prim.cone_deg is None else "SPOT",
        shape=shape,
        spot_size=0.0 if prim.cone_deg is None else math.radians(prim.cone_deg),
        radiance=radiance,
        area=area,
        face_area=face,
        cct_K=prim.cct_K,
    )


def _extent(points: list[mathutils.Vector]) -> tuple[Side, Side, Side]:
    """Axis-aligned low and high of points per axis."""
    xs, ys, zs = zip(*points, strict=True)
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _unparent(obj: bpy.types.Object) -> None:
    matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = matrix


def _glow_objects(materials: typing.Iterable[str]) -> dict[str, list[bpy.types.Object]]:
    """Mesh objects of the scene whose faces all use one of the named materials, by material name."""
    objects: dict[str, list[bpy.types.Object]] = {name: [] for name in materials}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for name, faces in _material_faces(obj).items():
            if name in objects and faces.all():
                objects[name].append(obj)
    return objects


class _ModelConverterExt(typing.Protocol):
    @classmethod
    def coordinates(cls) -> CoordinateSystem: ...

    @classmethod
    def load(cls, path: str, **options: typing.Any) -> None: ...

    @classmethod
    def save(cls, path: str) -> None: ...

    @classmethod
    def inline(cls, coordinates: CoordinateSystem, load: typing.Callable, save: typing.Callable) -> type[_ModelConverterExt]:
        class Impl(cls):
            @classmethod
            def coordinates(cls) -> CoordinateSystem:
                return coordinates

            @classmethod
            def load(cls, path: str, **options: typing.Any) -> None:
                load(filepath=path, **options)

            @classmethod
            def save(cls, path: str) -> None:
                save(filepath=path)

        return Impl


class ModelConverter:
    _reset: bool
    _coordinates: CoordinateSystem

    def __enter__(self) -> typing.Self:
        # Create temporary files to act as a bridge
        self.__ctx = capture_all_output(self._stdout, self._stderr)
        self.__ctx.__enter__()
        if self._reset:
            bpy.ops.wm.read_factory_settings(use_empty=True)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None):
        self.__ctx.__exit__(exc_type, exc_val, exc_tb)

    __exts: typing.ClassVar[dict[str, type[_ModelConverterExt]]] = {}

    @classmethod
    def register(cls, *ext: ModelFormat) -> typing.Callable[[type[_ModelConverterExt]], None]:
        def decorator(subcls: type[_ModelConverterExt]):
            for e in ext:
                cls.__exts[e.lower()] = subcls

        return decorator

    @classmethod
    def exts(cls) -> tuple[str, ...]:
        return tuple(cls.__exts.keys())

    def transform_coordinates(self, coords: CoordinateSystem) -> None:
        x, y, z = self._coordinates.get_transformation_to(coords)
        bpy.ops.object.select_all(action="SELECT")
        try:
            bpy.ops.transform.rotate(value=x, orient_axis="X", center_override=(0.0, 0.0, 0.0))
            bpy.ops.transform.rotate(value=y, orient_axis="Y", center_override=(0.0, 0.0, 0.0))
            bpy.ops.transform.rotate(value=z, orient_axis="Z", center_override=(0.0, 0.0, 0.0))
            self._coordinates = coords
        finally:
            bpy.ops.object.select_all(action="DESELECT")

    def __init__(self, *, reset: bool = True):
        self._reset = reset
        self._coordinates = CoordinateSystem.default()
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._lineages: dict[str, list[str]] = {}
        self._links: dict[str, str] = {}

    @property
    def stdout(self) -> str:
        return self._stdout.getvalue()

    @property
    def stderr(self) -> str:
        return self._stderr.getvalue()

    def load(self, path: str, **options: typing.Any) -> None:
        """Load a model, options going to the format's importer."""
        ext = os.path.splitext(path)[-1].lower().strip(".")
        if ext in self.__exts:
            ext_cls = self.__exts[ext]
            ext_cls.load(path, **options)
            self._coordinates = ext_cls.coordinates()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def rectify(self) -> np.ndarray:
        """Rectify the model's orientation and position (make it upright and floor-touching), return the applied 4x4 transform."""

        anchor = next((obj for obj in bpy.context.scene.objects if obj.type == "MESH"), None)
        bpy.context.view_layer.update()
        before = anchor.matrix_world.copy() if anchor is not None else None

        self.transform_coordinates(CoordinateSystem.default())
        bbox = self.bounding_box()

        # Check if model is too large in x or y and scale down if needed
        size_x = bbox.max_x - bbox.min_x
        size_y = bbox.max_y - bbox.min_y
        if size_x > 5 or size_y > 5:
            bpy.ops.object.select_all(action="SELECT")
            try:
                bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
            finally:
                bpy.ops.object.select_all(action="DESELECT")
            # Recalculate bounding box after scaling
            bbox = self.bounding_box()

        translation = mathutils.Vector(
            (
                0.0,
                0.0,
                -bbox.min_z,
            )
        )
        bpy.ops.object.select_all(action="SELECT")
        try:
            bpy.ops.transform.translate(value=translation)
        finally:
            bpy.ops.object.select_all(action="DESELECT")

        if anchor is None:
            return np.eye(4)
        assert before is not None
        bpy.context.view_layer.update()
        return np.array(anchor.matrix_world @ before.inverted())

    def bounding_box(self) -> BoundingBox:
        coords = []
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ mathutils.Vector(corner)
                    coords.append(world_corner)
        if not coords:
            return BoundingBox.empty()
        min_corner = mathutils.Vector(map(min, zip(*coords, strict=True)))
        max_corner = mathutils.Vector(map(max, zip(*coords, strict=True)))
        return BoundingBox(
            (
                (min_corner.x, max_corner.x),
                (min_corner.y, max_corner.y),
                (min_corner.z, max_corner.z),
            )
        )

    def limit_textures(self, max_px: int, directory: str) -> list[TextureLimit]:
        """Fit every image texture into max_px with 8 bits per channel as files in directory, repacking packed ones."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        changed: list[TextureLimit] = []
        for index, image in enumerate(bpy.data.images):
            if image.source != "FILE":
                continue
            size = tuple(image.size)
            if max(size) == 0 or (max(size) <= max_px and not image.is_float):
                continue
            if image.packed_file is not None:
                data = bytes(image.packed_file.data)
            else:
                source = Path(bpy.path.abspath(image.filepath, library=image.library))
                if not source.is_file():
                    continue
                data = source.read_bytes()
            try:
                encoded, image_format, (width, height), (new_width, new_height) = limited(data, max_px)
            except UnidentifiedImageError:
                continue
            packed = image.packed_file is not None
            target = out / f"{index}_{Path(image.filepath or image.name).stem}.{'png' if image_format == 'PNG' else 'jpg'}"
            target.write_bytes(encoded)
            image.filepath = str(target)
            if packed:
                image.unpack(method="REMOVE")
            image.file_format = image_format
            image.reload()
            if packed:
                image.pack()
            changed.append(TextureLimit(image.name, width, height, new_width, new_height))
        return changed

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space vertices and triangle indices of every mesh in the scene."""
        bpy.context.view_layer.update()
        points: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        offset = 0
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            mesh = obj.data
            mesh.calc_loop_triangles()
            vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vertices)
            indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", indices)
            matrix = np.array(obj.matrix_world)
            points.append(vertices.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3])
            triangles.append(indices.reshape(-1, 3).astype(np.int64) + offset)
            offset += len(mesh.vertices)
        if not points:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        return np.concatenate(points), np.concatenate(triangles)

    def lights(self, usd: str | None = None, links: dict[str, list[str]] | None = None, root: str = "") -> list[SourceLight]:
        """Light objects of the scene in world coordinates, sorted by object name, with the photometry of the UsdLux prims of usd and on the articulation links when given."""
        bpy.context.view_layer.update()
        prims = usd_lights(usd) if usd else []
        owners = node_links(links or {})
        found = []
        for obj in sorted(bpy.context.scene.objects, key=lambda obj: obj.name):
            if obj.type != "LIGHT":
                continue
            lineage = self._lineage(obj)
            light = SourceLight(
                name=obj.name,
                type=obj.data.type,
                shape=obj.data.shape if obj.data.type == "AREA" else "",
                spot_size=obj.data.spot_size if obj.data.type == "SPOT" else 0.0,
                power=obj.data.energy,
                color=tuple(obj.data.color),
                location=tuple(obj.matrix_world.translation),
                rotation=tuple(obj.matrix_world.decompose()[1]),
                link=link_of(lineage, owners, root) if links is not None else "",
            )
            prim = matching_prim([base_name(name) for name in lineage], prims)
            if prim is not None:
                prims.remove(prim)
                light = _usd_light(light, prim, obj.matrix_world.to_scale())
            found.append(light)
        return found

    def remove_lights(self) -> None:
        """Delete every light object, keeping the world transform of its children."""
        for obj in [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]:
            for child in obj.children:
                self._detach(child)
            light = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if light.users == 0:
                bpy.data.lights.remove(light)
        bpy.context.view_layer.update()

    def materials(self) -> list[str]:
        """Names of the materials used by mesh faces of the scene, sorted."""
        return sorted({name for obj in bpy.context.scene.objects if obj.type == "MESH" for name in _material_faces(obj)})

    def emissive_materials(self) -> list[SourceMaterial]:
        """Materials with a constant emission used by mesh faces of the scene, sorted by name."""
        bpy.context.view_layer.update()
        corners: dict[str, list[np.ndarray]] = {}
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or not any(slot.material is not None and _is_emissive(slot.material) for slot in obj.material_slots):
                continue
            for name, faces in _material_faces(obj).items():
                if _is_emissive(bpy.data.materials[name]):
                    points = _face_points(obj, faces)
                    corners.setdefault(name, []).extend((points.min(axis=0), points.max(axis=0)))
        return [SourceMaterial(name=name, center=tuple(((np.min(corners[name], axis=0) + np.max(corners[name], axis=0)) / 2).tolist())) for name in sorted(corners)]

    def prepare_glow(self, materials: typing.Iterable[str]) -> None:
        """Clear the emission of the named materials, move their faces into root objects of their own and record the names on the scene."""
        glow = sorted(set(materials))
        if not glow:
            return
        existing = self.materials()
        if missing := [name for name in glow if name not in existing]:
            raise ValueError(f"Glow material {', '.join(missing)} is not used by the model, its materials are: {', '.join(existing)}")

        for name in glow:
            material = bpy.data.materials[name]
            principled = _principled(material)
            if principled is None:
                continue
            color, strength = principled.inputs["Emission Color"], principled.inputs["Emission Strength"]
            for link in (*color.links, *strength.links):
                material.node_tree.links.remove(link)
            color.default_value = (0.0, 0.0, 0.0, 1.0)
            strength.default_value = 0.0

        for obj in [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]:
            faces = _material_faces(obj)
            names = [name for name in glow if name in faces]
            if names and np.logical_or.reduce([faces[name] for name in names]).all():
                names.pop()
            if not names:
                continue
            if obj.data.users > 1:
                obj.data = obj.data.copy()
            for name in names:
                piece = obj.copy()
                piece.data = obj.data.copy()
                piece.name = f"{obj.name}_{name}"
                self._lineages[piece.name] = self._lineage(obj)
                for collection in obj.users_collection:
                    collection.objects.link(piece)
                _keep_faces(piece.data, faces[name])
            _keep_faces(obj.data, ~np.logical_or.reduce([faces[name] for name in names]))

        bpy.context.view_layer.update()
        for objects in _glow_objects(glow).values():
            for obj in objects:
                for child in obj.children:
                    self._detach(child)
                self._detach(obj)
        bpy.context.view_layer.update()
        bpy.context.scene[GLOW_MATERIALS] = glow

    def _lineage(self, obj: bpy.types.Object) -> list[str]:
        """Names of an object and its ancestors as loaded, nearest first, kept across unparenting."""
        if obj.name in self._lineages:
            return self._lineages[obj.name]
        return [obj.name, *(self._lineage(obj.parent) if obj.parent is not None else ())]

    def _detach(self, obj: bpy.types.Object) -> None:
        self._lineages[obj.name] = self._lineage(obj)
        _unparent(obj)

    def split_links(self, links: dict[str, list[str]], root: str) -> list[LinkObjects]:
        """Assign every mesh object to the link naming it or its nearest listed ancestor, root taking the rest, and unparent the meshes."""
        bpy.context.view_layer.update()
        owners = node_links(links)
        known = {base_name(name) for obj in bpy.context.scene.objects for name in self._lineage(obj)}
        if missing := sorted(set(owners) - known):
            raise ValueError(f"articulation nodes {', '.join(missing)} match no object of the model, its objects are: {', '.join(sorted(obj.name for obj in bpy.context.scene.objects))}")
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        self._links = {obj.name: link_of(self._lineage(obj), owners, root) for obj in meshes}
        for obj in meshes:
            self._detach(obj)
        bpy.context.view_layer.update()
        return [LinkObjects(link=link, objects=sorted(name for name, owner in self._links.items() if owner == link)) for link in dict.fromkeys([*links, root])]

    def link_bounding_boxes(self) -> list[LinkBox]:
        """World-space bounding box of each link's meshes, links without meshes left out."""
        bpy.context.view_layer.update()
        corners: dict[str, list[mathutils.Vector]] = {}
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and obj.name in self._links:
                corners.setdefault(self._links[obj.name], []).extend(obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box)
        return [LinkBox(link=link, bounding_box=_extent(points)) for link, points in corners.items()]

    def save_link(self, link: str, path: str) -> None:
        """Save only the meshes of one link, every other object out of the scene for the export and back afterwards."""
        keep = {name for name, owner in self._links.items() if owner == link}
        if not any(obj.name in keep for obj in bpy.context.scene.objects):
            raise ValueError(f"link {link} has no meshes, links with meshes: {', '.join(sorted(set(self._links.values()))) or 'none, run split_links first'}")
        aside = [(obj, list(obj.users_collection)) for obj in bpy.context.scene.objects if obj.name not in keep]
        for obj, collections in aside:
            for collection in collections:
                collection.objects.unlink(obj)
        bpy.context.view_layer.update()
        try:
            self.save(path)
        finally:
            for obj, collections in aside:
                for collection in collections:
                    collection.objects.link(obj)
            bpy.context.view_layer.update()

    def bind_uv_maps(self) -> None:
        """Bind each material's image textures to the UV layer holding its UVs."""
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or not obj.data.uv_layers:
                continue
            mesh = obj.data
            for slot, material in enumerate(mesh.materials):
                if material is None or not material.use_nodes:
                    continue
                loops = [li for p in mesh.polygons if p.material_index == slot for li in p.loop_indices]
                if not loops:
                    continue

                best, best_span = None, 0.0
                for uv in mesh.uv_layers:
                    xs = [uv.data[li].uv.x for li in loops]
                    ys = [uv.data[li].uv.y for li in loops]
                    span = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    if span > best_span:
                        best, best_span = uv.name, span
                if best is None:
                    continue

                tree = material.node_tree
                uv_node = tree.nodes.new("ShaderNodeUVMap")
                uv_node.uv_map = best
                for node in tree.nodes:
                    if node.type == "TEX_IMAGE" and not node.inputs["Vector"].is_linked:
                        tree.links.new(uv_node.outputs["UV"], node.inputs["Vector"])

    @contextlib.contextmanager
    def ambient_context(self, strength: float):
        """White world lighting, kept out of the image by film_transparent."""
        scene = bpy.context.scene
        prev_world = scene.world
        world = bpy.data.worlds.new("RenderWorld")
        try:
            world.use_nodes = True
            background = world.node_tree.nodes["Background"]
            background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            background.inputs["Strength"].default_value = strength
            scene.world = world
            yield
        finally:
            scene.world = prev_world
            bpy.data.worlds.remove(world, do_unlink=True)

    @contextlib.contextmanager
    def camera_lighting_context(self):
        """Context manager that creates and manages a camera with sunlight.

        Args:
            center: Center point for the camera and lighting setup
            camera_height: Height at which to position the camera

        Yields:
            camera_object: The created camera object for configuration and rendering
        """
        scene = bpy.context.scene
        prev_camera = scene.camera
        created_objects: list[bpy.types.Object] = []
        created_data: list[bpy.types.ID] = []

        try:
            # Setup camera
            camera_data = bpy.data.cameras.new(name="RenderCamera")
            created_data.append(camera_data)
            camera_object = bpy.data.objects.new("RenderCamera", camera_data)
            created_objects.append(camera_object)
            scene.collection.objects.link(camera_object)
            scene.camera = camera_object

            # Key light - main sunlight angled toward the object
            sun_data = bpy.data.lights.new(name="RenderSun", type="SUN")
            created_data.append(sun_data)
            sun_data.energy = 6.0
            sun_object = bpy.data.objects.new("RenderSun", sun_data)
            created_objects.append(sun_object)
            scene.collection.objects.link(sun_object)

            # Fill light - soft light from opposite side for shadow fill
            fill_data = bpy.data.lights.new(name="RenderFill", type="AREA")
            created_data.append(fill_data)
            fill_data.energy = 2.0
            fill_data.size = 5.0
            fill_object = bpy.data.objects.new("RenderFill", fill_data)
            created_objects.append(fill_object)
            scene.collection.objects.link(fill_object)

            yield camera_object, sun_object, fill_object
        finally:
            scene.camera = prev_camera
            for obj in created_objects:
                if obj.name in scene.collection.objects:
                    scene.collection.objects.unlink(obj)
                bpy.data.objects.remove(obj, do_unlink=True)
            for data_block in created_data:
                if isinstance(data_block, bpy.types.Camera):
                    bpy.data.cameras.remove(data_block, do_unlink=True)
                elif isinstance(data_block, bpy.types.Light):
                    bpy.data.lights.remove(data_block, do_unlink=True)

    def _render(self, output_path: str):
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 32
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = output_path

        # Configure GPU rendering with CUDA
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            if not any(device.type == "CUDA" for device in prefs.devices):
                raise RuntimeError("No CUDA devices found")
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = "GPU"
        except Exception as e:
            logger.warning("CUDA unavailable (%s), rendering on CPU", e)
            prefs.compute_device_type = "NONE"
            scene.cycles.device = "CPU"

        bpy.ops.render.render(write_still=True)

    def _infer_resolution(self, bounding_box: BoundingBox) -> tuple[int, int]:
        size_x = bounding_box.max_x - bounding_box.min_x
        size_y = bounding_box.max_y - bounding_box.min_y

        min_short_side = 1024
        if size_x == 0.0 and size_y == 0.0:
            return (min_short_side, min_short_side)
        elif size_x >= size_y:
            res_y = min_short_side
            res_x = max(
                int(round(min_short_side * (size_x / max(size_y, 1e-6)))),
                min_short_side,
            )
            return (res_x, res_y)
        else:
            res_x = min_short_side
            res_y = max(
                int(round(min_short_side * (size_y / max(size_x, 1e-6)))),
                min_short_side,
            )
            return (res_x, res_y)

    def render_perspective(
        self,
        output_path: str,
        *,
        resolution: tuple[int, int] | None = None,
        theta: float = 0,
        elevation: float = math.pi / 4,
        ambient: float = 0.0,
    ) -> None:
        """Render the current scene to a perspective image, viewing the face at azimuth theta."""
        scene = bpy.context.scene

        bbox = self.bounding_box()
        center = mathutils.Vector(
            (
                (bbox.min_x + bbox.max_x) / 2,
                (bbox.min_y + bbox.max_y) / 2,
                (bbox.min_z + bbox.max_z) / 2,
            )
        )

        if resolution is None:
            resolution = self._infer_resolution(bbox)
        scene.render.resolution_x = max(*resolution)
        scene.render.resolution_y = max(*resolution)

        with self.camera_lighting_context() as (camera_object, sun_object, fill_object):
            camera_data = camera_object.data
            camera_data.type = "PERSP"

            size = max(
                bbox.max_x - bbox.min_x,
                bbox.max_y - bbox.min_y,
                bbox.max_z - bbox.min_z,
            )
            fov = camera_object.data.angle
            camera_distance = (size / 2.0) / math.tan(fov / 2.0) * 1.3
            camera_object.location = center + mathutils.Vector(
                (
                    -camera_distance * math.cos(theta) * math.sin(elevation),
                    -camera_distance * math.sin(theta) * math.sin(elevation),
                    camera_distance * math.cos(elevation),
                )
            )
            camera_object.rotation_euler = (center - camera_object.location).to_track_quat("-Z", "Y").to_euler()

            sun_object.location = camera_object.location + mathutils.Vector((0.0, 0.0, 1.0))
            sun_object.rotation_euler = (center - sun_object.location).to_track_quat("-Z", "Y").to_euler()

            fill_object.location = camera_object.location + mathutils.Vector((0.0, 0.0, -1.0))
            fill_object.rotation_euler = (center - fill_object.location).to_track_quat("-Z", "Y").to_euler()

            with self.ambient_context(ambient) if ambient > 0 else contextlib.nullcontext():
                self._render(output_path)

    def render_topdown(self, output_path: str, *, resolution: tuple[int, int] | None = None) -> None:
        """Render an orthographic top-down preview that snugly fits the XY bounds.

        Args:
            output_path: Path where the rendered image will be saved
        """
        scene = bpy.context.scene

        bbox = self.bounding_box()
        if resolution is None:
            resolution = self._infer_resolution(bbox)

        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.resolution_percentage = 100

        size_x = bbox.max_x - bbox.min_x
        size_y = bbox.max_y - bbox.min_y
        ortho_scale = max(size_x, size_y)

        center = mathutils.Vector(
            (
                (bbox.min_x + bbox.max_x) / 2,
                (bbox.min_y + bbox.max_y) / 2,
                (bbox.min_z + bbox.max_z) / 2,
            )
        )

        camera_height = max(bbox.max_z - bbox.min_z, 1.0) * 2.0

        with self.camera_lighting_context() as (camera_object, sun_object, fill_object):
            camera_object.data.type = "ORTHO"
            camera_object.location = center + mathutils.Vector((0.0, 0.0, camera_height))
            camera_object.rotation_euler = (0.0, 0.0, 0.0)
            camera_object.data.ortho_scale = max(ortho_scale, 1e-6)

            sun_object.location = camera_object.location + mathutils.Vector((0.0, 0.0, 1.0))
            sun_object.rotation_euler = (center - sun_object.location).to_track_quat("-Z", "Y").to_euler()

            fill_object.location = camera_object.location + mathutils.Vector((0.0, 0.0, -1.0))
            fill_object.rotation_euler = (center - fill_object.location).to_track_quat("-Z", "Y").to_euler()

            self._render(output_path)

    def save(self, path: str, *, ext: ModelFormat | None = None) -> None:
        ext = ModelFormat(os.path.splitext(path)[-1].lower().strip("."))
        if ext in self.__exts:
            ext_cls = self.__exts[ext]
            # self.transform_coordinates(ext_cls.coordinates())
            ext_cls.save(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


ModelConverter.register(ModelFormat.USD, ModelFormat.USDA, ModelFormat.USDC, ModelFormat.USDZ)(
    _ModelConverterExt.inline(
        CoordinateSystem.default(),
        functools.partial(bpy.ops.wm.usd_import, support_scene_instancing=False),
        functools.partial(
            bpy.ops.wm.usd_export,
            export_materials=True,
            export_normals=True,
            export_uvmaps=True,
            export_animation=False,
            selected_objects_only=False,
            export_textures_mode="NEW",
            overwrite_textures=True,
            export_lights=False,
            export_cameras=False,
            export_mesh_colors=False,
        ),
    )
)


_PACKED_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "OPEN_EXR": ".exr", "TARGA": ".tga", "BMP": ".bmp", "WEBP": ".webp"}


def obj_export(filepath: str):
    """Export an OBJ folder, the packed images without a file (as a glTF brings them) written into it so the MTL can name them."""
    base_path = Path(filepath)
    base_path.mkdir(parents=True, exist_ok=True)
    pathless = [image for image in bpy.data.images if image.packed_file is not None and not image.filepath]
    for n, image in enumerate(pathless):
        target = base_path / f"{base_path.stem}_texture_{n}{_PACKED_SUFFIX.get(image.file_format, '.png')}"
        target.write_bytes(image.packed_file.data)
        image.filepath = str(target)
    try:
        bpy.ops.wm.obj_export(
            filepath=str(base_path / f"{base_path.stem}.obj"),
            export_materials=True,
            path_mode="COPY",
        )
    finally:
        for image in pathless:
            image.filepath = ""


ModelConverter.register(ModelFormat.OBJ)(_ModelConverterExt.inline(CoordinateSystem.default(), bpy.ops.wm.obj_import, obj_export))


def fbx_import(filepath: str):
    bpy.ops.import_scene.fbx(filepath=filepath)
    # FBX importer limitation for BSDF: Only changes Metallic to 0.0 if it's unlinked and exactly 1.0.
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            principled = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)

            if principled:
                metallic_socket = principled.inputs.get("Metallic")

                if metallic_socket:
                    if not metallic_socket.is_linked and metallic_socket.default_value >= 0.99:
                        metallic_socket.default_value = 0.0


ModelConverter.register(ModelFormat.FBX)(
    _ModelConverterExt.inline(
        CoordinateSystem.default(),
        fbx_import,
        functools.partial(
            bpy.ops.export_scene.fbx,
            object_types={"ARMATURE", "EMPTY", "MESH", "OTHER"},
            embed_textures=True,
            path_mode="COPY",
        ),
    )
)

ModelConverter.register(ModelFormat.DAE)(_ModelConverterExt.inline(CoordinateSystem.default(), bpy.ops.wm.collada_import, bpy.ops.wm.collada_export))


def _collada_export(path: Path, objects: typing.Iterable[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    try:
        for obj in objects:
            obj.select_set(True)
        bpy.ops.wm.collada_export(filepath=str(path), selected=True)
    finally:
        bpy.ops.object.select_all(action="DESELECT")


def _sdf_material(material: bpy.types.Material) -> str:
    principled = _principled(material)
    if principled is None or principled.inputs["Base Color"].is_linked:
        return ""
    color = " ".join(f"{round(channel, 4):g}" for channel in principled.inputs["Base Color"].default_value[:3])
    return f"""
        <material>
          <ambient>{color} 1</ambient>
          <diffuse>{color} 1</diffuse>
          <specular>0 0 0 1</specular>
          <emissive>0 0 0 1</emissive>
        </material>"""


def sdf_export(filepath: str):
    base_path = Path(filepath)
    stem = base_path.stem
    glow = {name: objects for name, objects in _glow_objects(bpy.context.scene.get(GLOW_MATERIALS, ())).items() if objects}
    plain = True
    visuals = []
    if glow:
        apart = {obj.name for objects in glow.values() for obj in objects}
        rest = [obj for obj in bpy.context.view_layer.objects if obj.name not in apart]
        plain = any(obj.type == "MESH" for obj in rest)
        if plain:
            _collada_export(base_path / f"{stem}.dae", rest)
        for index, (name, objects) in enumerate(glow.items()):
            uri = f"{stem}.glow_{index}.dae"
            _collada_export(base_path / uri, objects)
            visuals.append((xml.sax.saxutils.escape(f"glow_{name}", {'"': "&quot;"}), uri, _sdf_material(bpy.data.materials[name])))
    else:
        bpy.ops.wm.collada_export(filepath=str(base_path / f"{stem}.dae"))
    (base_path / f"{stem}.sdf").write_text(sdf_model(stem, visuals, plain=plain))


ModelConverter.register(ModelFormat.SDF)(
    _ModelConverterExt.inline(
        CoordinateSystem.default(),
        lambda filepath: NotImplementedError("SDF import is not implemented yet."),
        sdf_export,
    )
)


def gltf_export(filepath: str):
    """Export glTF and clear typing's alias caches, which otherwise keep the exporter and bpy alive at exit."""
    bpy.ops.export_scene.gltf(filepath=filepath)
    for cleanup in typing._cleanups:  # pyright: ignore[reportAttributeAccessIssue]
        cleanup()


ModelConverter.register(ModelFormat.GLB, ModelFormat.GLTF)(_ModelConverterExt.inline(CoordinateSystem.default(), bpy.ops.import_scene.gltf, gltf_export))


__all__ = ["ModelConverter"]
