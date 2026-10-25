import json
import subprocess
import sys

import pytest

WORKER = [sys.executable, "-m", "arena_assets_blender"]

CUBE = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


@pytest.fixture()
def cube(tmp_path):
    path = tmp_path / "cube.obj"
    path.write_text(CUBE)
    return path


def serve(*requests: dict) -> list[dict]:
    proc = subprocess.run([*WORKER, "--serve"], input="".join(json.dumps(request) + "\n" for request in requests), capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_exts_prints_the_registered_formats():
    proc = subprocess.run([*WORKER, "--exts"], capture_output=True, text=True, check=True)
    assert {"usd", "usda", "usdc", "usdz", "obj", "fbx", "dae", "sdf", "glb", "gltf"} == set(json.loads(proc.stdout.splitlines()[-1]))


def test_serve_replies_once_per_request_on_a_clean_channel(cube, tmp_path):
    first, second = serve(
        {"steps": [["load", str(cube)], ["rectify"], ["bounding_box"], ["save", str(tmp_path / "cube.sdf")]]},
        {"steps": [["load", str(cube)], ["bounding_box"]]},
    )
    assert first["ok"] and second["ok"]
    assert first["values"][2] == [pytest.approx([0.0, 1.0], abs=1e-6), pytest.approx([-1.0, 0.0], abs=1e-6), pytest.approx([0.0, 1.0], abs=1e-6)]
    assert second["values"][1] == [pytest.approx(side, abs=1e-6) for side in first["values"][2]]
    assert (tmp_path / "cube.sdf" / "cube.sdf").is_file()
    assert (tmp_path / "cube.sdf" / "cube.dae").is_file()


def test_serve_reports_a_failed_step_and_continues(cube, tmp_path):
    failed, unknown, malformed, fine = serve(
        {"steps": [["load", str(tmp_path / "missing.xyz")]]},
        {"steps": [["quit_blender"]]},
        {"nothing": []},
        {"steps": [["load", str(cube)], ["bounding_box"]]},
    )
    assert not failed["ok"] and "Unsupported file extension" in failed["error"]
    assert not unknown["ok"] and "unknown operation" in unknown["error"]
    assert not malformed["ok"]
    assert fine["ok"]


def test_serve_passes_keyword_arguments_to_the_renderer(cube, tmp_path):
    (reply,) = serve({"steps": [["load", str(cube)], ["rectify"], ["render_topdown", str(tmp_path / "top.png"), {"resolution": [48, 48]}]]})
    assert reply["ok"], reply.get("error")
    assert (tmp_path / "top.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_convert_command_writes_the_output(cube, tmp_path):
    subprocess.run([*WORKER, str(cube), str(tmp_path / "cube.usdz")], capture_output=True, text=True, check=True)
    assert (tmp_path / "cube.usdz").stat().st_size > 0


def test_a_kept_scene_carries_into_the_next_request(cube, tmp_path):
    kept, carried, fresh = serve(
        {"steps": [["load", str(cube)], ["rectify"]], "keep": True},
        {"steps": [["bounding_box"], ["save", str(tmp_path / "cube.usdz")]]},
        {"steps": [["bounding_box"]]},
    )
    assert kept["ok"] and carried["ok"] and fresh["ok"]
    assert carried["values"][0] == [pytest.approx([0.0, 1.0], abs=1e-6), pytest.approx([-1.0, 0.0], abs=1e-6), pytest.approx([0.0, 1.0], abs=1e-6)]
    assert (tmp_path / "cube.usdz").stat().st_size > 0
    assert fresh["values"][0] != carried["values"][0]


def test_lights_come_back_as_records(tmp_path):
    script = "import sys, bpy\nbpy.ops.wm.read_factory_settings(use_empty=True)\nbpy.ops.mesh.primitive_cube_add(size=1.0)\ndata = bpy.data.lights.new('Bulb', 'POINT')\ndata.energy = 10.0\nobj = bpy.data.objects.new('Bulb', data)\nobj.location = (0.0, 0.0, 2.0)\nbpy.context.scene.collection.objects.link(obj)\nbpy.ops.export_scene.gltf(filepath=sys.argv[1], export_lights=True)\n"
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "lamp.glb")], check=True, capture_output=True, timeout=120)
    (reply,) = serve({"steps": [["load", str(tmp_path / "lamp.glb")], ["lights"], ["remove_lights"], ["lights"], ["emissive_materials"]]})
    assert reply["ok"], reply.get("error")
    _, (light,), removed, after, materials = reply["values"]
    assert (light["name"], light["type"], light["shape"], light["spot_size"]) == ("Bulb", "POINT", "", 0.0)
    assert set(light) == {"name", "type", "shape", "spot_size", "power", "color", "location", "rotation", "radiance", "area", "face_area", "cct_K", "link"}
    assert (light["radiance"], light["area"], light["face_area"], light["cct_K"], light["link"]) == (None, 0.0, 0.0, None, "")
    assert light["location"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-3)
    assert (removed, after, materials) == (None, [], [])


def test_triangles_come_back_as_world_space_vertices_and_indices(cube):
    (reply,) = serve({"steps": [["load", str(cube)], ["rectify"], ["triangles"]]})
    assert reply["ok"], reply.get("error")
    points, triangles = reply["values"][2]
    assert len(points) == 8
    assert len(triangles) == 12
    assert {len(triangle) for triangle in triangles} == {3}
    assert [min(column) for column in zip(*points, strict=True)] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
    assert [max(column) for column in zip(*points, strict=True)] == pytest.approx([1.0, 0.0, 1.0], abs=1e-6)
