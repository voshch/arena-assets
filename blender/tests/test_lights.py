import xml.etree.ElementTree as ET

import pytest

from arena_assets_blender.lights import UsdLight, matching_prim


def _link(path):
    """Visuals and collisions of a model SDF as (tag, name, uri, material diffuse)."""
    link = ET.parse(path).getroot().find("model/link")
    return [(element.tag, element.get("name"), element.findtext("geometry/mesh/uri"), element.findtext("material/diffuse")) for element in link]


def _plain():
    return [("visual", "visual", "lamp.dae", None), ("collision", "collision", "lamp.dae", None)]


def _glow(index, name, color=None):
    uri = f"lamp.glow_{index}.dae"
    return [("visual", f"glow_{name}", uri, None if color is None else f"{color} 1"), ("collision", f"collision_glow_{index}", uri, None)]


def _material_faces(path):
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as model_converter:
        model_converter.load(str(path))
        bpy.context.view_layer.update()
        faces = {}
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            for polygon in obj.data.polygons:
                name = obj.material_slots[polygon.material_index].material.name
                corners = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
                faces.setdefault(name, []).extend(tuple(round(component, 3) + 0.0 for component in corner) for corner in corners)
    return {name: (tuple(map(min, zip(*corners, strict=True))), tuple(map(max, zip(*corners, strict=True)))) for name, corners in faces.items()}


def _add_cube(location, *materials):
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    cube = bpy.context.active_object
    for material in materials:
        cube.data.materials.append(material)
    for polygon in cube.data.polygons:
        polygon.material_index = len(materials) - 1 if polygon.normal.z > 0.5 else 0
    return cube


def test_prepare_glow_splits_every_instance_of_shared_mesh(tmp_path):
    pytest.importorskip("bpy")
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as model_converter:
        cube = _add_cube((0.0, 0.0, 0.5), bpy.data.materials.new("Base"), bpy.data.materials.new("Shade"))
        twin = bpy.data.objects.new("Twin", cube.data)
        twin.location = (2.0, 0.0, 0.5)
        bpy.context.scene.collection.objects.link(twin)
        model_converter.prepare_glow(["Shade"])
        model_converter.save(str(tmp_path / "lamp.sdf"))

    assert _link(tmp_path / "lamp.sdf" / "lamp.sdf") == _plain() + _glow(0, "Shade")
    assert _material_faces(tmp_path / "lamp.sdf" / "lamp.dae") == {"Base": ((-0.5, -0.5, 0.0), (2.5, 0.5, 1.0))}
    assert _material_faces(tmp_path / "lamp.sdf" / "lamp.glow_0.dae") == {"Shade": ((-0.5, -0.5, 1.0), (2.5, 0.5, 1.0))}


def test_sdf_of_fully_glowing_model_has_only_glow_visual(tmp_path):
    pytest.importorskip("bpy")
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as model_converter:
        shade = bpy.data.materials.new("Shade")
        shade.use_nodes = True
        _add_cube((0.0, 0.0, 0.5), shade)
        model_converter.prepare_glow(["Shade"])
        model_converter.save(str(tmp_path / "lamp.sdf"))

    assert _link(tmp_path / "lamp.sdf" / "lamp.sdf") == _glow(0, "Shade", "0.8 0.8 0.8")
    assert sorted(path.name for path in (tmp_path / "lamp.sdf").iterdir()) == ["lamp.glow_0.dae", "lamp.sdf"]


def _prim(*names):
    return UsdLight(names=names, kind="RectLight", intensity=1.0)


def test_light_object_matches_the_prim_of_its_full_lineage():
    prims = [_prim("root", "door", "RectLight"), _prim("root", "RectLight")]
    assert matching_prim(["RectLight", "door", "root"], prims) == prims[0]
    assert matching_prim(["RectLight", "root"], prims) == prims[1]


def test_light_object_matches_its_prim_across_a_merged_xform():
    prims = [_prim("root", "socket", "bulb")]
    assert matching_prim(["bulb", "root"], prims) == prims[0]


def test_light_object_matches_no_prim_of_another_name_or_branch():
    prims = [_prim("root", "door", "lamp")]
    assert matching_prim(["glow", "door", "root"], prims) is None
    assert matching_prim(["lamp", "drawer", "root"], prims) is None
