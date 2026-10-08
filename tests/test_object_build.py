import subprocess
import sys

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.CoordinateSystem import CoordinateSystem
from arena_assets.utils.geom import BoundingBox

BOX_FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
UPRIGHT_BOX = BoundingBox(((-0.5, 0.5), (-1.0, 1.0), (0.0, 3.0)))
BOX_EXTENTS = {
    "Z": ((-0.5, 0.5), (-1.0, 1.0), (0.0, 3.0)),
    "Y": ((-0.5, 0.5), (0.0, 3.0), (-1.0, 1.0)),
}
OFF_CENTER = "double3 xformOp:translate = (4, 6, 2)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]"
EXTRAS = {
    "LIGHT": f"""
def SphereLight "Lamp"
{{
    float inputs:intensity = 1000
    float inputs:radius = 0.1
    {OFF_CENTER}
}}
""",
    "EMPTY": f"""
def Xform "Marker"
{{
    {OFF_CENTER}
}}
""",
}


def _usda_box(up_axis, extra=""):
    xs, ys, zs = BOX_EXTENTS[up_axis]
    points = ", ".join(f"({x}, {y}, {z})" for x in xs for y in ys for z in zs)
    return f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "{up_axis}"
)

def Xform "Root"
{{
    def Mesh "Box"
    {{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = {BOX_FACES}
        point3f[] points = [{points}]
    }}
}}
{extra}"""


def _write_box_source(directory, up_axis="Z", extra=""):
    directory.mkdir(parents=True)
    (directory / "box.usda").write_text(_usda_box(up_axis, extra))
    (directory / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "box"}))
    return directory / "box.usda"


def _flat(bounding_box):
    return [bound for side in bounding_box for bound in side]


def _loaded_box(path):
    from arena_assets.utils.ModelConverter.converter import ModelConverter

    with ModelConverter() as model_converter:
        model_converter.load(str(path))
        return model_converter.bounding_box().round(4)


@pytest.mark.parametrize("up_axis", ["Z", "Y"])
def test_build_keeps_usd_source_upright(tmp_path, up_axis):
    pytest.importorskip("bpy")
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    _write_box_source(input_path / "Object" / "box", up_axis)

    builder = ObjectDatabaseBuilder(input_path, output_path)
    builder.enable("formats", "fbx")
    dest = output_path / "Object" / "box"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(input_path / "Object" / "box"), dest)

    assert built is not None
    assert built.bounding_box == UPRIGHT_BOX
    assert _flat(_loaded_box(dest / "box.fbx")) == pytest.approx(_flat(UPRIGHT_BOX), abs=1e-3)


def _transformed(path, extra_type=None):
    import bpy

    from arena_assets.utils.ModelConverter.converter import ModelConverter

    with ModelConverter() as model_converter:
        model_converter.load(str(path))
        bpy.context.view_layer.update()
        objects = {obj.type: obj for obj in bpy.context.scene.objects if obj.name != "Root"}
        mesh = objects["MESH"]
        anchor = None if extra_type is None else mesh.matrix_world.inverted() @ objects[extra_type].matrix_world.translation

        model_converter.transform_coordinates(CoordinateSystem("X+", "Y+", "Z+"))
        bpy.context.view_layer.update()

        offset = None if extra_type is None else tuple(objects[extra_type].matrix_world.translation - mesh.matrix_world @ anchor)
        return model_converter.bounding_box().round(4), offset


@pytest.mark.parametrize("extra_type", ["LIGHT", "EMPTY"])
def test_coordinate_transform_pivot_ignores_lights_and_empties(tmp_path, extra_type):
    pytest.importorskip("bpy")

    plain, _ = _transformed(_write_box_source(tmp_path / "plain"))
    crowded, offset = _transformed(_write_box_source(tmp_path / "crowded", extra=EXTRAS[extra_type]), extra_type)

    assert plain == BoundingBox(((-0.5, 0.5), (-3.0, 0.0), (-1.0, 1.0)))
    assert crowded == plain
    assert offset == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)


GLB_BUILD_SCRIPT = """
import sys
from pathlib import Path

from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

input_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
builder = ObjectDatabaseBuilder(input_path, output_path)
builder.enable("formats", "glb")
dest = output_path / "Object" / "box"
dest.mkdir(parents=True)
built = builder.process_entity(builder.read_annotation_file(input_path / "Object" / "box"), dest)
assert built is not None
assert (dest / "box.glb").stat().st_size > 0
"""


def test_process_exits_after_gltf_build(tmp_path):
    pytest.importorskip("bpy")

    _write_box_source(tmp_path / "input" / "Object" / "box")
    result = subprocess.run([sys.executable, "-c", GLB_BUILD_SCRIPT, str(tmp_path / "input"), str(tmp_path / "output")], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
