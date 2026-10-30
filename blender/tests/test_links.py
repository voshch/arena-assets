import json
import re
import subprocess
import sys

import pytest

WORKER = [sys.executable, "-m", "arena_assets_blender"]
LINKS = {"body": ["base"], "door": ["door"], "handle": ["handle"]}
FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]


def _cube(name, location, parent=None, size=0.2):
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    cube = bpy.context.active_object
    cube.name = name
    if parent is not None:
        bpy.context.view_layer.update()
        cube.parent = parent
        cube.matrix_parent_inverse = parent.matrix_world.inverted()
    return cube


def _empty(name, location, parent=None):
    import bpy

    empty = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = location
    if parent is not None:
        empty.parent = parent
    bpy.context.view_layer.update()
    return empty


def _cabinet():
    """Root empty holding base, a stray mesh, a door empty with its panel, a handle.001 on the panel and a knob on the handle."""
    import bpy

    root = _empty("Root", (0.0, 0.0, 1.0))
    _cube("base", (0.0, 0.0, 1.0), root)
    _cube("stray", (0.0, 1.0, 1.0), root)
    door = _empty("door", (1.0, 0.0, 0.0), root)
    panel = _cube("panel", (1.0, 0.0, 1.0), door)
    handle = _cube("handle.001", (1.2, 0.0, 1.0), panel, size=0.1)
    _cube("knob", (1.3, 0.0, 1.0), handle, size=0.05)
    bpy.context.view_layer.update()


def _world(name):
    import bpy

    return [value for row in bpy.data.objects[name].matrix_world for value in row]


def _scene():
    import bpy

    bpy.context.view_layer.update()
    return {obj.name: (obj.type, [[round(value, 6) for value in row] for row in obj.matrix_world], sorted(collection.name for collection in obj.users_collection)) for obj in bpy.context.scene.objects}


def _obj_names(path):
    return sorted(line.split(maxsplit=1)[1] for line in path.read_text().splitlines() if line.startswith("o "))


def test_split_links_assigns_meshes_by_own_name_and_nearest_listed_ancestor():
    pytest.importorskip("bpy")
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        before = {name: _world(name) for name in ("base", "stray", "panel", "handle.001", "knob")}
        records = converter.split_links(LINKS, "body")

        assert [(record.link, record.objects) for record in records] == [("body", ["base", "stray"]), ("door", ["panel"]), ("handle", ["handle.001", "knob"])]
        for name, world in before.items():
            assert bpy.data.objects[name].parent is None
            assert _world(name) == pytest.approx(world, abs=1e-6)


def test_split_links_names_the_missing_nodes_and_the_objects():
    pytest.importorskip("bpy")
    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        with pytest.raises(ValueError, match=r"articulation nodes drawer, lid match no object of the model, its objects are: Root, base, door, handle\.001, knob, panel, stray"):
            converter.split_links({"body": [], "door": ["door", "lid"], "drawer": ["drawer"]}, "body")


def test_link_bounding_boxes_cover_each_links_meshes():
    pytest.importorskip("bpy")
    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        converter.split_links(LINKS, "body")
        boxes = {record.link: [bound for side in record.bounding_box for bound in side] for record in converter.link_bounding_boxes()}

    assert boxes["body"] == pytest.approx([-0.1, 0.1, -0.1, 1.1, 0.9, 1.1], abs=1e-6)
    assert boxes["door"] == pytest.approx([0.9, 1.1, -0.1, 0.1, 0.9, 1.1], abs=1e-6)
    assert boxes["handle"] == pytest.approx([1.15, 1.325, -0.05, 0.05, 0.95, 1.05], abs=1e-6)


def test_save_link_exports_only_its_meshes_and_leaves_the_scene_as_it_was(tmp_path):
    pytest.importorskip("bpy")
    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        converter.split_links(LINKS, "body")
        scene = _scene()
        converter.save_link("handle", str(tmp_path / "handle.obj"))
        converter.save_link("door", str(tmp_path / "door.sdf"))
        converter.save_link("body", str(tmp_path / "body.usda"))
        assert _scene() == scene
        with pytest.raises(ValueError, match="link lid has no meshes, links with meshes: body, door, handle"):
            converter.save_link("lid", str(tmp_path / "lid.obj"))

    assert _obj_names(tmp_path / "handle.obj" / "handle.obj") == ["handle.001", "knob"]
    assert (tmp_path / "door.sdf" / "door.sdf").is_file()
    door_dae = (tmp_path / "door.sdf" / "door.dae").read_text()
    assert 'name="panel"' in door_dae and 'name="base"' not in door_dae
    prims = set(re.findall(r'def \w+ "([^"]+)"', (tmp_path / "body.usda").read_text()))
    assert {"base", "stray"} <= prims
    assert not prims & {"panel", "handle_001", "knob", "door", "Root"}


def test_glow_pieces_follow_the_link_of_their_source_object():
    pytest.importorskip("bpy")
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        lamp = _cube("lamp", (1.0, 0.0, 1.5), bpy.data.objects["door"])
        lamp.data.materials.append(bpy.data.materials.new("Body"))
        lamp.data.materials.append(bpy.data.materials.new("Shade"))
        for polygon in lamp.data.polygons:
            polygon.material_index = 1 if polygon.normal.z > 0.5 else 0
        converter.prepare_glow(["Shade"])
        records = {record.link: record.objects for record in converter.split_links(LINKS, "body")}

    assert records["door"] == ["lamp", "lamp_Shade", "panel"]


def test_a_glowing_link_saves_as_sdf_without_the_other_links(tmp_path):
    pytest.importorskip("bpy")
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as converter:
        _cabinet()
        lamp = _cube("lamp", (1.0, 0.0, 1.5), bpy.data.objects["door"])
        lamp.data.materials.append(bpy.data.materials.new("Body"))
        lamp.data.materials.append(bpy.data.materials.new("Shade"))
        for polygon in lamp.data.polygons:
            polygon.material_index = 1 if polygon.normal.z > 0.5 else 0
        converter.prepare_glow(["Shade"])
        converter.split_links(LINKS, "body")
        converter.save_link("door", str(tmp_path / "door.sdf"))
        names = sorted(obj.name for obj in bpy.context.view_layer.objects)

    assert sorted(path.name for path in (tmp_path / "door.sdf").iterdir()) == ["door.dae", "door.glow_0.dae", "door.sdf"]
    assert {"base", "stray", "knob"} <= set(names)


def _usda(path):
    def mesh(name, x):
        points = ", ".join(f"({px}, {py}, {pz})" for px in x for py in (0, 1) for pz in (0, 1))
        return f'def Mesh "{name}"\n{{\nint[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\nint[] faceVertexIndices = {FACES}\npoint3f[] points = [{points}]\n}}\n'

    path.write_text(f'#usda 1.0\n(\ndefaultPrim = "Root"\nmetersPerUnit = 1\nupAxis = "Z"\n)\ndef Xform "Root"\n{{\n{mesh("base", (0, 1))}def Xform "lid"\n{{\n{mesh("lidmesh", (1, 2))}}}\n}}\n')
    return path


def test_the_worker_splits_a_usd_model_by_the_xform_it_merges_by_default(tmp_path):
    model = _usda(tmp_path / "box.usda")
    steps = [["split_links", {"body": [], "lid": ["lid"]}, "body"], ["link_bounding_boxes"], ["save_link", "lid", str(tmp_path / "lid.obj")]]
    request = [{"steps": [["load", str(model)], *steps]}, {"steps": [["load", str(model), {"merge_parent_xform": False}], *steps]}]
    proc = subprocess.run([*WORKER, "--serve"], input="".join(json.dumps(line) + "\n" for line in request), capture_output=True, text=True, check=True)
    merged, kept = (json.loads(line) for line in proc.stdout.splitlines())

    assert not merged["ok"] and "articulation nodes lid match no object" in merged["error"]
    assert kept["ok"], kept.get("error")
    assert kept["values"][1] == [{"link": "body", "objects": ["base"]}, {"link": "lid", "objects": ["lidmesh"]}]
    assert {record["link"]: record["bounding_box"] for record in kept["values"][2]} == {"body": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], "lid": [[1.0, 2.0], [0.0, 1.0], [0.0, 1.0]]}
    assert kept["values"][3] is None
    assert _obj_names(tmp_path / "lid.obj" / "lid.obj") == ["lidmesh"]
