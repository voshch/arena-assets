from __future__ import annotations

import contextlib
import functools
import io
import math
import os
import types
import typing
from pathlib import Path

import bpy
import mathutils

from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.logging import get_logger

from ..CoordinateSystem import CoordinateSystem
from ..io_utils import capture_all_output
from . import ModelFormat

logger = get_logger("ModelConverter")


class _ModelConverterExt(typing.Protocol):
    @classmethod
    def coordinates(cls) -> CoordinateSystem: ...

    @classmethod
    def load(cls, path: str) -> None: ...

    @classmethod
    def save(cls, path: str) -> None: ...

    @classmethod
    def inline(cls, coordinates: CoordinateSystem, load: typing.Callable, save: typing.Callable) -> type[_ModelConverterExt]:
        class Impl(cls):
            @classmethod
            def coordinates(cls) -> CoordinateSystem:
                return coordinates

            @classmethod
            def load(cls, path: str) -> None:
                load(filepath=path)

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

    __exts: typing.ClassVar[dict[ModelFormat, _ModelConverterExt]] = {}

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
            bpy.ops.transform.rotate(value=x, orient_axis="X")
            bpy.ops.transform.rotate(value=y, orient_axis="Y")
            bpy.ops.transform.rotate(value=z, orient_axis="Z")
            self._coordinates = coords
        finally:
            bpy.ops.object.select_all(action="DESELECT")

    def __init__(self, *, reset: bool = True):
        self._reset: bool = reset
        self._coordinates: CoordinateSystem = CoordinateSystem.default()
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()

    @property
    def stdout(self) -> str:
        return self._stdout.getvalue()

    @property
    def stderr(self) -> str:
        return self._stderr.getvalue()

    def load(self, path: str) -> None:
        ext = os.path.splitext(path)[-1].lower().strip(".")
        if ext in self.__exts:
            ext_cls = self.__exts[ext]
            ext_cls.load(path)
            self._coordinates = ext_cls.coordinates()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def rectify(self):
        """Rectify the model's orientation and position (make it upright and floor-touching)."""

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
        CoordinateSystem("X+", "Y+", "Z+"),
        bpy.ops.wm.usd_import,
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


def obj_export(filepath: str):
    base_path = Path(filepath)
    base_path.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(
        filepath=str(base_path / f"{base_path.stem}.obj"),
        export_materials=True,
        path_mode="COPY",
    )


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


def sdf_export(filepath: str):
    base_path = Path(filepath)
    dae_path = base_path / f"{base_path.stem}.dae"
    bpy.ops.wm.collada_export(filepath=str(dae_path))

    sdf_path = base_path / f"{base_path.stem}.sdf"
    with open(sdf_path, "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{base_path.stem}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{base_path.stem}.dae</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>""")


ModelConverter.register(ModelFormat.SDF)(
    _ModelConverterExt.inline(
        CoordinateSystem.default(),
        lambda filepath: NotImplementedError("SDF import is not implemented yet."),
        sdf_export,
    )
)


ModelConverter.register(ModelFormat.GLB, ModelFormat.GLTF)(_ModelConverterExt.inline(CoordinateSystem.default(), bpy.ops.import_scene.gltf, bpy.ops.export_scene.gltf))


__all__ = ["ModelConverter"]
