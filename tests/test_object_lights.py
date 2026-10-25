import json
import logging
import math
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME, Face, converter
from arena_assets.impl.build.ObjectDatabaseBuilder import (
    ISAAC_EXPOSURE_GAIN,
    LightFixture,
    ObjectAnnotation,
    ObjectLight,
    light_lumens,
    object_lights,
)
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.ModelConverter import environment
from arena_assets.utils.ModelConverter.lights import SourceLight, SourceMaterial, color_temperature, nearest_material

BULB = {"fixture": "bulb", "pose": [[0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0]], "lumens": 800.0}


def emission(light):
    """Unit vector a light emits along: the local -z axis of its pose quaternion."""
    w, x, y, z = light.pose[1]
    return (-2.0 * (x * z + w * y), -2.0 * (y * z - w * x), -(1.0 - 2.0 * (x * x + y * y)))


def structure(*lights):
    return converter.structure(
        {"name": "lamp", "path": "Object/lamp", "lights": list(lights)},
        ObjectAnnotation,
    )


def source(**fields):
    return SourceLight(
        **{
            "name": "Light",
            "type": "POINT",
            "power": 10.0,
            "color": (1.0, 1.0, 1.0),
            "location": (0.0, 0.0, 2.0),
            **fields,
        }
    )


def test_light_defaults_fill_name_and_cct():
    (light,) = structure(BULB).lights
    assert light == ObjectLight(
        name="light",
        fixture=LightFixture.BULB,
        pose=((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0)),
        lumens=800.0,
        cct_K=6500.0,
        cone_deg=None,
    )


def test_light_structures_every_field():
    (light,) = structure(
        {
            "name": "reading_1",
            "fixture": "spot",
            "pose": [[1, 2, 3], [0, 1.5708, 0]],
            "lumens": 450,
            "cct_K": 2700,
            "cone_deg": 60,
        }
    ).lights
    assert light.name == "reading_1"
    assert light.fixture is LightFixture.SPOT
    assert light.pose == ((1.0, 2.0, 3.0), (0.0, 1.5708, 0.0))
    assert light.lumens == 450.0
    assert light.cct_K == 2700.0
    assert light.cone_deg == 60.0


def test_light_null_cone_is_absent():
    (light,) = structure({**BULB, "cone_deg": None}).lights
    assert light.cone_deg is None


def test_annotation_without_lights_defaults_to_empty_list():
    assert structure().lights == []
    assert converter.structure({"name": "lamp", "path": "Object/lamp"}, ObjectAnnotation).lights == []


def test_annotation_yaml_without_lights_has_no_lights_key():
    annotation = ObjectAnnotation(
        name="Chair",
        path="Object/Chair",
        desc="office chair",
        tags=["chair", "color::black"],
        face=Face.POS_X,
        note="n",
        bounding_box=BoundingBox(((-0.5, 0.5), (-0.4, 0.4), (0.0, 1.2))),
    )
    assert yaml.safe_dump(converter.unstructure(annotation)) == ("asa: 0\nbounding_box:\n- - -0.5\n  - 0.5\n- - -0.4\n  - 0.4\n- - 0.0\n  - 1.2\ndesc: office chair\nface: +x\nname: Chair\nnote: n\npath: Object/Chair\ntags:\n- chair\n- color::black\n")


def test_lights_roundtrip_through_yaml():
    annotation = structure(BULB, {"name": "reading", "fixture": "spot", "pose": [[0.2, 0.0, 1.0], [0.0, -0.7854, 0.0]], "lumens": 300.0, "cone_deg": 45.0})
    data = yaml.safe_load(yaml.safe_dump(converter.unstructure(annotation)))
    assert data["lights"] == [
        {"name": "light", "fixture": "bulb", "pose": [[0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0]], "lumens": 800.0, "cct_K": 6500.0},
        {"name": "reading", "fixture": "spot", "pose": [[0.2, 0.0, 1.0], [0.0, -0.7854, 0.0]], "lumens": 300.0, "cct_K": 6500.0, "cone_deg": 45.0},
    ]
    assert converter.structure(data, ObjectAnnotation) == annotation


def test_glow_roundtrips_through_yaml_and_metadata():
    annotation = structure({**BULB, "glow": "Shade"}, {**BULB, "name": "plain"})
    data = yaml.safe_load(yaml.safe_dump(converter.unstructure(annotation)))
    assert [light.get("glow") for light in data["lights"]] == ["Shade", None]
    assert "glow" not in data["lights"][1]
    assert converter.structure(data, ObjectAnnotation) == annotation
    assert ObjectAnnotation.from_metadata(annotation.as_metadata) == annotation
    assert [light.glow for light in annotation.lights] == ["Shade", None]


def test_lights_roundtrip_through_metadata():
    annotation = structure({**BULB, "cct_K": 3000.0}, {"name": "spot", "fixture": "spot", "pose": [[0, 0, 1], [0, 0, 0]], "lumens": 10, "cone_deg": 30})
    metadata = annotation.as_metadata
    assert json.loads(metadata["lights"])[0]["cct_K"] == 3000.0
    assert ObjectAnnotation.from_metadata(metadata) == annotation


def test_metadata_without_lights_has_no_lights_column():
    assert "lights" not in structure().as_metadata


@pytest.mark.parametrize(
    "entry",
    [
        {**BULB, "fixture": "chandelier"},
        {**BULB, "name": "Main Light"},
        {**BULB, "name": ""},
        {**BULB, "lumens": 0.0},
        {**BULB, "lumens": -5.0},
        {**BULB, "cct_K": 0.0},
        {**BULB, "pose": [[0.0, 0.0, 1.5], [0.0, 0.0, 0.0, 0.0]]},
        {**BULB, "pose": [[0.0, 0.0, 1.5], [0.0, 1.0]]},
        {**BULB, "pose": [[0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, 0.0]]},
        {**BULB, "pose": [0.0, 0.0, 1.5]},
        {**BULB, "cone_deg": 0.0},
        {**BULB, "cone_deg": 180.0},
        {"pose": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], "lumens": 800.0},
        {"fixture": "bulb", "lumens": 800.0},
    ],
    ids=[
        "unknown_fixture",
        "uppercase_name",
        "empty_name",
        "zero_lumens",
        "negative_lumens",
        "zero_cct",
        "zero_quaternion",
        "short_orientation",
        "long_position",
        "flat_pose",
        "zero_cone",
        "straight_cone",
        "missing_fixture",
        "missing_pose",
    ],
)
def test_invalid_light_raises_value_error(entry):
    with pytest.raises(ValueError):
        structure(entry)


def test_duplicate_light_names_raise_value_error():
    with pytest.raises(ValueError, match="light"):
        structure(BULB, {**BULB, "pose": [[0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]]})


@pytest.mark.parametrize(
    ("blender_type", "shape", "fixture"),
    [
        ("POINT", "", LightFixture.BULB),
        ("SPOT", "", LightFixture.SPOT),
        ("AREA", "SQUARE", LightFixture.PANEL),
        ("AREA", "RECTANGLE", LightFixture.PANEL),
        ("AREA", "DISK", LightFixture.DOWNLIGHT),
        ("AREA", "ELLIPSE", LightFixture.DOWNLIGHT),
    ],
)
def test_blender_light_type_maps_to_fixture(blender_type, shape, fixture):
    (light,) = object_lights([source(type=blender_type, shape=shape, spot_size=math.radians(40.0))])
    assert light.fixture is fixture
    assert light.cone_deg == (40.0 if fixture is LightFixture.SPOT else None)


def test_sun_is_skipped_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        lights = object_lights([source(name="Sun", type="SUN"), source(name="Bulb")])
    assert [light.fixture for light in lights] == [LightFixture.BULB]
    assert lights[0].name == "light"
    assert "Sun" in caplog.text


def test_powerless_light_is_skipped_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert object_lights([source(power=0.0)]) == []
    assert "Light" in caplog.text


def test_power_converts_to_lumens_at_683_per_watt():
    (light,) = object_lights([source(power=1.5)])
    assert light.lumens == 1024.5


def test_blender_spot_keeps_the_cone_share_of_its_point_equivalent_power():
    (light,) = object_lights([source(type="SPOT", power=5.0, spot_size=math.radians(60.0))])
    assert light.lumens == pytest.approx(683.0 * 5.0 * 0.25 / 4.0, abs=0.05)


def test_usd_panel_lumens_carry_its_radiance_over_its_world_area_to_isaac():
    (light,) = object_lights([source(type="AREA", shape="RECTANGLE", radiance=20000.0, area=0.0341, face_area=0.0341, power=62832.0)])
    assert light.lumens == pytest.approx(math.pi * 20000.0 * 0.0341 / ISAAC_EXPOSURE_GAIN, abs=0.05)


def test_usd_spot_lumens_keep_the_on_axis_intensity_of_its_face():
    spot = source(type="SPOT", spot_size=math.radians(40.0), radiance=1000.0, area=4.0, face_area=1.0)
    assert light_lumens(spot, LightFixture.SPOT) == pytest.approx(math.pi * 1000.0 * math.sin(math.radians(20.0)) ** 2 / ISAAC_EXPOSURE_GAIN)


def test_usd_color_temperature_overrides_the_light_color():
    (light,) = object_lights([source(color=(1.0, 0.6, 0.3), cct_K=6200.0, radiance=1.0, area=1.0)])
    assert light.cct_K == 6200.0


def test_light_link_is_kept_and_written_only_when_set():
    lights = object_lights([source(name="a", link="door"), source(name="b")])
    assert [light.link for light in lights] == ["door", None]
    written = converter.unstructure(lights[0]), converter.unstructure(lights[1])
    assert written[0]["link"] == "door"
    assert "link" not in written[1]
    assert converter.structure(written[0], ObjectLight) == lights[0]


def test_straight_spot_cone_stays_below_180():
    (light,) = object_lights([source(type="SPOT", spot_size=math.pi)])
    assert 179.0 < light.cone_deg < 180.0


def test_lights_are_named_in_source_name_order():
    lights = object_lights(
        [
            source(name="c", location=(0.0, 0.0, 3.0)),
            source(name="a", location=(0.0, 0.0, 1.0)),
            source(name="b", location=(0.0, 0.0, 2.0)),
        ]
    )
    assert [(light.name, light.pose[0][2]) for light in lights] == [("light", 1.0), ("light_1", 2.0), ("light_2", 3.0)]


def test_pose_keeps_the_light_rotation_and_rounds_its_position():
    tilt = math.atan2(3.0, 4.0)
    rotation = (math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)
    (light,) = object_lights([source(type="SPOT", spot_size=1.0, location=(0.123456, -0.00001, 2.0), rotation=rotation)])
    assert light.pose[0] == (0.1235, 0.0, 2.0)
    assert math.copysign(1.0, light.pose[0][1]) == 1.0
    assert light.pose[1] == pytest.approx(rotation, abs=1e-6)
    assert emission(light) == pytest.approx((0.0, 0.6, -0.8), abs=1e-5)


def test_bulb_pose_drops_the_light_rotation():
    (light,) = object_lights([source(rotation=(0.0, 1.0, 0.0, 0.0))])
    assert light.pose == ((0.0, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0))


def test_white_color_is_daylight():
    assert color_temperature((1.0, 1.0, 1.0)) == pytest.approx(6500.0, abs=10.0)


def test_warm_color_is_below_4000_kelvin():
    warm = color_temperature((1.0, 0.6, 0.3))
    assert 2500.0 < warm < 4000.0
    assert warm % 10.0 == 0.0


def test_cool_color_is_above_daylight():
    assert color_temperature((0.6, 0.8, 1.0)) > 7000.0


def test_black_color_falls_back_to_daylight():
    assert color_temperature((0.0, 0.0, 0.0)) == 6500.0


def test_color_temperature_is_clamped():
    for color in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]:
        assert 1000.0 <= color_temperature(color) <= 40000.0


def material(name, *center):
    return SourceMaterial(name=name, center=center)


def test_nearest_material_is_none_without_materials():
    assert nearest_material((0.0, 0.0, 0.0), []) is None


def test_nearest_material_picks_smallest_distance():
    materials = [material("Far", 0.0, 0.0, 5.0), material("Near", 0.0, 1.0, 0.0), material("Mid", 2.0, 0.0, 0.0)]
    assert nearest_material((0.0, 0.0, 0.0), materials) == "Near"
    assert nearest_material((0.0, 0.0, 4.0), materials) == "Far"


def test_nearest_material_breaks_ties_by_name():
    materials = [material("b", 1.0, 0.0, 0.0), material("a", -1.0, 0.0, 0.0)]
    assert nearest_material((0.0, 0.0, 0.0), materials) == "a"


def test_extracted_lights_glow_their_nearest_material():
    lights = object_lights(
        [source(name="a", location=(0.0, 0.0, 1.0)), source(name="b", location=(0.0, 0.0, 1.2)), source(name="c", location=(3.0, 0.0, 0.0))],
        [material("Shade", 0.0, 0.0, 1.1), material("Screen", 2.0, 0.0, 0.0), material("Ember", -9.0, 0.0, 0.0)],
    )
    assert [light.glow for light in lights] == ["Shade", "Shade", "Screen"]


def test_extracted_lights_have_no_glow_without_emissive_materials():
    assert [light.glow for light in object_lights([source()])] == [None]


LAMP_SOURCE_SCRIPT = """
import math
import sys

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5))
bpy.context.active_object.name = "Body"


def add_light(name, kind, location, rotation, power, color):
    data = bpy.data.lights.new(name, kind)
    data.energy = power
    data.color = color
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    obj.rotation_euler = rotation
    scene.collection.objects.link(obj)
    return obj


bulb = add_light("Bulb", "POINT", (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), 10.0, (1.0, 1.0, 1.0))
spot = add_light("Spot", "SPOT", (0.5, 0.0, 1.5), (math.radians(30.0), 0.0, 0.0), 5.0, (1.0, 0.6, 0.3))
spot.data.spot_size = math.radians(60.0)
add_light("Sun", "SUN", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), 3.0, (1.0, 1.0, 1.0))

bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0.0, 0.0, 0.0))
glass = bpy.context.active_object
glass.name = "Glass"
glass.parent = bulb
glass.location = (0.0, 0.0, -0.2)

bpy.ops.export_scene.gltf(filepath=sys.argv[1], export_lights=True)
"""


GLOW_LAMP_SOURCE_SCRIPT = """
import sys

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def add_material(name, base, emission=None, strength=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*base, 1.0)
    if emission is not None:
        shader.inputs["Emission Color"].default_value = (*emission, 1.0)
        shader.inputs["Emission Strength"].default_value = strength
    return material


def add_cube(name, size, location, *materials):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    cube = bpy.context.active_object
    cube.name = name
    for material in materials:
        cube.data.materials.append(material)
    return cube


def add_light(name, kind, location):
    data = bpy.data.lights.new(name, kind)
    data.energy = 10.0
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    scene.collection.objects.link(obj)


base = add_material("Base", (0.5, 0.5, 0.5))
shade = add_material("Shade", (1.0, 0.5, 0.25), (1.0, 0.8, 0.6), 1.0)
screen = add_material("Screen", (0.0, 0.25, 1.0), (0.2, 0.4, 1.0), 1.0)
ember = add_material("Ember", (0.25, 0.0, 0.0), (1.0, 0.2, 0.0), 1.0)

image = bpy.data.images.new("Pixels", 2, 2)
image.pixels = [0.0, 0.25, 1.0, 1.0] * 4
texture = screen.node_tree.nodes.new("ShaderNodeTexImage")
texture.image = image
shader = next(node for node in screen.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
screen.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])

rig = bpy.data.objects.new("Rig", None)
rig.location = (0.0, 0.0, 0.5)
scene.collection.objects.link(rig)
body = add_cube("Body", 1.0, (0.0, 0.0, 0.0), base, shade)
body.parent = rig
for polygon in body.data.polygons:
    polygon.material_index = 1 if polygon.normal.z > 0.5 else 0
panel = add_cube("Panel", 0.2, (1.0, 0.0, 0.1), screen)
coal = add_cube("Coal", 0.2, (-2.0, 0.0, 0.0), ember)
coal.parent = panel

add_light("Bulb", "POINT", (0.0, 0.0, 1.2))
add_light("Spot", "SPOT", (1.0, 0.0, 0.6))

bpy.ops.export_scene.gltf(filepath=sys.argv[1], export_lights=True)
"""


SCENE_TYPES_SCRIPT = """
import json
import sys

import bpy

from arena_assets_blender.converter import ModelConverter

with ModelConverter() as model_converter:
    model_converter.load(sys.argv[1])
    bpy.context.view_layer.update()
    types = sorted(obj.type for obj in bpy.context.scene.objects)
    bounding_box = model_converter.bounding_box().round(4)
print(json.dumps([types, bounding_box]))
"""

MATERIAL_FACES_SCRIPT = """
import json
import sys

import bpy

from arena_assets_blender.converter import ModelConverter

with ModelConverter() as model_converter:
    model_converter.load(sys.argv[1])
    bpy.context.view_layer.update()
    faces = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for polygon in obj.data.polygons:
            name = obj.material_slots[polygon.material_index].material.name
            corners = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
            faces.setdefault(name, []).extend(tuple(round(component, 3) + 0.0 for component in corner) for corner in corners)
print(json.dumps({name: [list(map(min, zip(*corners, strict=True))), list(map(max, zip(*corners, strict=True)))] for name, corners in faces.items()}))
"""

EMISSIVE_SCRIPT = """
import json
import sys

from arena_assets_blender.converter import ModelConverter

with ModelConverter() as model_converter:
    model_converter.load(sys.argv[1])
    names = [material.name for material in model_converter.emissive_materials()]
print(json.dumps(names))
"""


def _in_blender(python, script, *args):
    """Last stdout line of a script run in the Blender worker's interpreter, as JSON."""
    result = subprocess.run([python, "-c", script, *map(str, args)], capture_output=True, text=True, timeout=120, env=environment())
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _write_lamp_source(python, directory, annotation, script=LAMP_SOURCE_SCRIPT):
    directory.mkdir(parents=True)
    subprocess.run([python, "-c", script, str(directory / "lamp.glb")], check=True, capture_output=True, timeout=120, env=environment())
    (directory / ANNOTATION_NAME).write_text(yaml.safe_dump(annotation))


def _scene_types(python, path):
    types, bounding_box = _in_blender(python, SCENE_TYPES_SCRIPT, path)
    return types, BoundingBox(bounding_box)


def _build_lamp(python, tmp_path, annotation, script=LAMP_SOURCE_SCRIPT):
    _write_lamp_source(python, tmp_path / "input" / "Object" / "lamp", annotation, script)
    return _build_written_lamp(tmp_path)


def _build_written_lamp(tmp_path):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    builder = ObjectDatabaseBuilder(input_path, output_path)
    builder.enable("formats", "usdz,sdf,fbx")
    dest = output_path / "Object" / "lamp"
    dest.mkdir(parents=True)
    source_annotation = builder.read_annotation_file(input_path / "Object" / "lamp")
    assert isinstance(source_annotation, ObjectAnnotation)
    return builder.process_entity(source_annotation, dest), dest


def _process_lamp(python, tmp_path, annotation, script=LAMP_SOURCE_SCRIPT):
    built, dest = _build_lamp(python, tmp_path, annotation, script)
    assert built is not None
    return built, dest


EXPECTED_BOX = BoundingBox(((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.9)))


def test_build_extracts_source_lights_and_strips_them_from_every_model(blender_python, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        built, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp"})

    assert "Sun" in caplog.text

    assert built.bounding_box == EXPECTED_BOX
    assert [light.name for light in built.lights] == ["light", "light_1"]
    bulb, spot = built.lights
    assert bulb.fixture is LightFixture.BULB
    assert bulb.pose[0] == pytest.approx((0.0, 0.0, 2.0), abs=1e-3)
    assert emission(bulb) == pytest.approx((0.0, 0.0, -1.0), abs=1e-3)
    assert bulb.lumens == pytest.approx(6830.0, rel=1e-3)
    assert bulb.cct_K == pytest.approx(6500.0, abs=10.0)
    assert bulb.cone_deg is None
    assert spot.fixture is LightFixture.SPOT
    assert spot.pose[0] == pytest.approx((0.5, 0.0, 1.5), abs=1e-3)
    assert emission(spot) == pytest.approx((0.0, 0.5, -0.866), abs=1e-3)
    assert spot.lumens == pytest.approx(213.4, rel=1e-3)
    assert spot.cct_K < 4000.0
    assert spot.cone_deg == pytest.approx(60.0, abs=0.1)

    written = yaml.safe_load(yaml.safe_dump(converter.unstructure(built)))
    assert [light["fixture"] for light in written["lights"]] == ["bulb", "spot"]

    for model in (dest / "lamp.usdz", dest / "lamp.sdf" / "lamp.dae", dest / "lamp.fbx"):
        types, bounding_box = _scene_types(blender_python, model)
        assert "MESH" in types, model
        assert "LIGHT" not in types, model
        assert [c for side in bounding_box for c in side] == pytest.approx([c for side in EXPECTED_BOX for c in side], abs=1e-3), model


def test_build_keeps_annotated_lights_over_source_lights(blender_python, tmp_path):
    annotated = {"name": "shade", "fixture": "downlight", "pose": [[0.0, 0.0, 1.8], [1.0, 0.0, 0.0, 0.0]], "lumens": 900.0, "cct_K": 2700.0}
    built, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp", "lights": [annotated]})

    assert built.lights == [converter.structure(annotated, ObjectLight)]
    types, _ = _scene_types(blender_python, dest / "lamp.sdf" / "lamp.dae")
    assert "LIGHT" not in types


def _material_faces(python, path):
    return {name: (tuple(low), tuple(high)) for name, (low, high) in _in_blender(python, MATERIAL_FACES_SCRIPT, path).items()}


def _emissive(python, path):
    return _in_blender(python, EMISSIVE_SCRIPT, path)


def _collada_emissions(path):
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    return {effect.get("id").removesuffix("-effect"): [float(channel) for channel in effect.find(".//c:emission/c:color", namespace).text.split()] for effect in ET.parse(path).getroot().iterfind(".//c:effect", namespace)}


def _link(path):
    """Visuals and collisions of a model SDF as (tag, name, uri, material diffuse)."""
    link = ET.parse(path).getroot().find("model/link")
    return [(element.tag, element.get("name"), element.findtext("geometry/mesh/uri"), element.findtext("material/diffuse")) for element in link]


def _plain():
    return [("visual", "visual", "lamp.dae", None), ("collision", "collision", "lamp.dae", None)]


def _glow(index, name, color=None):
    uri = f"lamp.glow_{index}.dae"
    return [("visual", f"glow_{name}", uri, None if color is None else f"{color} 1"), ("collision", f"collision_glow_{index}", uri, None)]


SHADE_FACE = ((-0.5, -0.5, 1.0), (0.5, 0.5, 1.0))
BODY_BOX = ((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))
PANEL_BOX = ((0.9, -0.1, 0.0), (1.1, 0.1, 0.2))
COAL_BOX = ((-1.1, -0.1, 0.0), (-0.9, 0.1, 0.2))


def test_build_assigns_nearest_emissive_material_as_glow(blender_python, tmp_path):
    built, _ = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp"}, GLOW_LAMP_SOURCE_SCRIPT)

    assert [(light.name, light.fixture, light.glow) for light in built.lights] == [("light", LightFixture.BULB, "Shade"), ("light_1", LightFixture.SPOT, "Screen")]
    written = yaml.safe_load(yaml.safe_dump(converter.unstructure(built)))
    assert [light["glow"] for light in written["lights"]] == ["Shade", "Screen"]


def test_build_exports_each_glow_material_as_own_sdf_visual(blender_python, tmp_path):
    _, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp"}, GLOW_LAMP_SOURCE_SCRIPT)
    sdf = dest / "lamp.sdf"

    assert _link(sdf / "lamp.sdf") == (_plain() + _glow(0, "Screen") + _glow(1, "Shade", "1 0.5 0.25"))
    assert _material_faces(blender_python, sdf / "lamp.dae") == {"Base": BODY_BOX, "Ember": COAL_BOX}
    assert _material_faces(blender_python, sdf / "lamp.glow_0.dae") == {"Screen": PANEL_BOX}
    assert _material_faces(blender_python, sdf / "lamp.glow_1.dae") == {"Shade": SHADE_FACE}


def test_build_clears_glow_emission_in_every_model(blender_python, tmp_path):
    _, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp"}, GLOW_LAMP_SOURCE_SCRIPT)
    sdf = dest / "lamp.sdf"

    for model in (dest / "lamp.usdz", dest / "lamp.fbx"):
        assert _emissive(blender_python, model) == ["Ember"], model
        assert _material_faces(blender_python, model) == {"Base": BODY_BOX, "Ember": COAL_BOX, "Screen": PANEL_BOX, "Shade": SHADE_FACE}, model
    assert _collada_emissions(sdf / "lamp.dae") == {"Base": [0.0, 0.0, 0.0, 1.0], "Ember": pytest.approx([1.0, 0.2, 0.0, 1.0], abs=1e-3)}
    assert _collada_emissions(sdf / "lamp.glow_0.dae") == {"Screen": [0.0, 0.0, 0.0, 1.0]}
    assert _collada_emissions(sdf / "lamp.glow_1.dae") == {"Shade": [0.0, 0.0, 0.0, 1.0]}


def test_build_glows_only_annotated_material(blender_python, tmp_path):
    annotated = {"name": "shade", "fixture": "bulb", "pose": [[0.0, 0.0, 1.2], [1.0, 0.0, 0.0, 0.0]], "lumens": 900.0, "glow": "Shade"}
    built, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp", "lights": [annotated]}, GLOW_LAMP_SOURCE_SCRIPT)
    sdf = dest / "lamp.sdf"

    assert built.lights == [converter.structure(annotated, ObjectLight)]
    assert _link(sdf / "lamp.sdf") == _plain() + _glow(0, "Shade", "1 0.5 0.25")
    assert not (sdf / "lamp.glow_1.dae").exists()
    assert _material_faces(blender_python, sdf / "lamp.dae") == {"Base": BODY_BOX, "Ember": COAL_BOX, "Screen": PANEL_BOX}
    assert _material_faces(blender_python, sdf / "lamp.glow_0.dae") == {"Shade": SHADE_FACE}
    assert _emissive(blender_python, dest / "lamp.usdz") == ["Ember", "Screen"]


def test_missing_glow_material_raises_value_error(blender_python, tmp_path):
    from arena_assets.utils.ModelConverter import BlenderError, BlenderWorker

    _write_lamp_source(blender_python, tmp_path / "lamp", {}, GLOW_LAMP_SOURCE_SCRIPT)
    source = str(tmp_path / "lamp" / "lamp.glb")
    worker = BlenderWorker()
    try:
        with pytest.raises(BlenderError, match="Glass.*Base, Ember, Screen, Shade"):
            worker.run([("load", source), ("prepare_glow", ["Shade", "Glass"])])
        assert [material["name"] for material in worker.run([("load", source), ("emissive_materials",)]).values[1]] == ["Ember", "Screen", "Shade"]
    finally:
        worker.close()


def test_build_fails_on_annotated_glow_naming_missing_material(blender_python, tmp_path, caplog):
    annotated = {"fixture": "bulb", "pose": [[0.0, 0.0, 1.2], [1.0, 0.0, 0.0, 0.0]], "lumens": 900.0, "glow": "Glass"}
    with caplog.at_level(logging.ERROR):
        built, dest = _build_lamp(blender_python, tmp_path, {"desc": "floor lamp", "lights": [annotated]}, GLOW_LAMP_SOURCE_SCRIPT)

    assert built is None
    assert "Glass" in caplog.text
    assert "Base, Ember, Screen, Shade" in caplog.text
    assert list(dest.iterdir()) == []


def test_build_without_glow_materials_writes_single_visual_sdf(blender_python, tmp_path):
    built, dest = _process_lamp(blender_python, tmp_path, {"desc": "floor lamp"})

    assert [light.glow for light in built.lights] == [None, None]
    assert sorted(path.name for path in (dest / "lamp.sdf").iterdir()) == ["lamp.dae", "lamp.sdf"]
    assert _link(dest / "lamp.sdf" / "lamp.sdf") == _plain()


def _box_mesh(name, size):
    half = size / 2.0
    points = ", ".join(f"({x}, {y}, {z})" for x in (-half, half) for y in (-half, half) for z in (-half, half))
    return f'def Mesh "{name}"\n{{\nint[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\nint[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]\npoint3f[] points = [{points}]\n}}\n'


USD_LAMP = f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 0.01
    upAxis = "Y"
)
def Xform "Root"
{{
    {_box_mesh("Body", 50.0)}
    def Xform "Shade"
    {{
        double3 xformOp:translate = (0, 100, 0)
        float3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        {_box_mesh("Glass", 5.0)}
        def RectLight "Panel"
        {{
            float inputs:width = 10
            float inputs:height = 5
            float inputs:intensity = 20000
            float inputs:exposure = 1
            bool inputs:enableColorTemperature = 1
            float inputs:colorTemperature = 3000
        }}
        def DiskLight "Disk"
        {{
            float inputs:radius = 3
            float inputs:intensity = 100
            bool inputs:normalize = 1
        }}
    }}
    def SphereLight "Spot" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {{
        float inputs:radius = 2
        float inputs:intensity = 500000
        float inputs:shaping:cone:angle = 30
        double3 xformOp:translate = (0, 80, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def Xform "Socket"
    {{
        double3 xformOp:translate = (20, 60, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def SphereLight "Bulb"
        {{
            float inputs:radius = 5
            float inputs:intensity = 10000
        }}
    }}
}}
"""


def _authored(intensity, area):
    return math.pi * intensity * area / ISAAC_EXPOSURE_GAIN


def test_build_reads_usd_light_photometry_from_usdlux(blender_python, tmp_path):
    source = tmp_path / "input" / "Object" / "lamp"
    source.mkdir(parents=True)
    (source / "lamp.usda").write_text(USD_LAMP)
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "desk lamp"}))
    built, _ = _process_lamp_written(tmp_path)

    bulb, disk, panel, spot = built.lights
    assert (bulb.fixture, disk.fixture, panel.fixture, spot.fixture) == (LightFixture.BULB, LightFixture.DOWNLIGHT, LightFixture.PANEL, LightFixture.SPOT)
    assert bulb.lumens == pytest.approx(_authored(10000.0, 4.0 * math.pi * 0.05**2), abs=0.1)
    assert disk.lumens == pytest.approx(_authored(100.0, 1.0), abs=0.1)
    assert panel.lumens == pytest.approx(_authored(20000.0 * 2.0, 0.2 * 0.1), abs=0.1)
    assert spot.lumens == pytest.approx(_authored(500000.0, math.pi * 0.02**2) * 0.25, abs=0.1)
    assert spot.cone_deg == pytest.approx(60.0, abs=0.01)
    assert (panel.cct_K, bulb.cct_K) == (3000.0, 6500.0)
    assert panel.pose[0] == pytest.approx((0.0, 0.0, 1.25), abs=1e-3)
    assert bulb.pose[0] == pytest.approx((0.2, 0.0, 0.85), abs=1e-3)
    assert all(light.link is None for light in built.lights)


def _process_lamp_written(tmp_path):
    built, dest = _build_written_lamp(tmp_path)
    assert built is not None
    return built, dest
