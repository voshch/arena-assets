import io
import json
import tarfile
import urllib.error
import urllib.parse
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, License
from arena_assets.importers.artiverse import GLB, JOINTS, MANIFEST, RELEASES, REPO, REVISION, Artiverse, artiverse_articulation, extract, read_glb, release_chunks, stage, write_glb
from arena_assets.importers.providers import HFHub
from blender_scene import owners, scene, unmatched
from http_host import Host

ITEM = "a1b2c3"
CORNERS = [(x, y, z) for x in (0.0, 0.1) for y in (0.0, 0.1) for z in (0.0, 0.1)]
TRIANGLES = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
PARTS = {1: ("base", (0.0, 0.0, 0.0)), 2: ("door", (0.5, 0.0, 0.25)), 3: ("drawer", (0.0, 0.2, 0.0)), 5: ("knob", (0.0, 1.0, 0.0)), 6: ("lamp arm", (0.0, 0.5, 0.0)), 7: ("cushion", (0.0, 0.6, 0.0))}
JOINT_ENTRIES = [
    {"pid": 2, "type": "rotation", "origin": [0.5, 0.0, 0.25], "axis": [0, 1, 0], "base": [1], "rangeMin": 0, "rangeMax": 1.5, "motionStates": {"closed": {"percent": 0, "value": 0}, "ajar": {"percent": 27, "value": 0.4}}},
    {"pid": 3, "type": "translation", "origin": [0, 0, 0], "axis": [0, 0, 1], "rangeMin": 0, "rangeMax": 0.3},
    {"pid": 5, "type": "cylindrical", "origin": [0, 1, 0], "axis": [0, 1, 0], "prismatic_rangeMin": 0, "prismatic_rangeMax": 0.1},
    {"pid": 6, "type": "universal", "origin1": [0, 0.5, 0], "axis1": [1, 0, 0], "axis2": [0, 0, 1], "rangeMin": -0.5, "rangeMax": 0.5, "rot2RangeMin": -0.2, "rot2RangeMax": 0.2},
    {"pid": 7, "type": "free"},
]


def segmented_glb(path: Path) -> dict:
    """A part-segmented glTF as Artiverse ships it: a root node whose children are parts with id and label extras, one part holding a child node, the part tree in the document's flattened_graph."""
    binary = np.array(CORNERS, dtype=np.float32).tobytes() + np.array(TRIANGLES, dtype=np.uint16).tobytes()
    nodes: list[dict] = [{"name": "model", "children": [n + 1 for n in range(len(PARTS))]}]
    for pid, (label, at) in PARTS.items():
        nodes.append({"name": f"Part {label}", "mesh": 0, "translation": list(at), "extras": {"id": pid, "label": label}})
    nodes[2]["children"] = [len(nodes)]
    nodes.append({"name": "handle", "mesh": 0, "translation": [0.0, 0.05, 0.0]})
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(CORNERS), "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.1, 0.1, 0.1]},
            {"bufferView": 1, "componentType": 5123, "count": len(TRIANGLES), "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12 * len(CORNERS)}, {"buffer": 0, "byteOffset": 12 * len(CORNERS), "byteLength": 2 * len(TRIANGLES)}],
        "buffers": [{"byteLength": len(binary) + (-len(binary) % 4)}],
        "extras": {"flattened_graph": {str(pid): {"parent": -1 if pid == 1 else 1} for pid in PARTS}},
    }
    write_glb(path, document, binary)
    return document


def item_files(folder: Path, entries: list[dict] = JOINT_ENTRIES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    segmented_glb(folder / f"{ITEM}{GLB}")
    (folder / f"{ITEM}{JOINTS}").write_text(json.dumps({"articulations": entries}), encoding="utf-8-sig")
    return folder / f"{ITEM}{GLB}"


def by_name(articulation) -> dict:
    return {joint.name: joint for joint in articulation.joints}


def test_registry_marks_artiverse_non_commercial_and_bulk():
    assert IMPORTERS.load("object", "artiverse") is Artiverse
    meta = IMPORTERS.meta("object", "artiverse")
    assert (meta.license, meta.on_demand.value) == (License.NC, "bulk")


def test_glb_round_trip_keeps_document_and_buffer(tmp_path):
    document = segmented_glb(tmp_path / "in.glb")
    read, binary = read_glb(tmp_path / "in.glb")
    assert read == document
    write_glb(tmp_path / "out.glb", read, binary)
    assert read_glb(tmp_path / "out.glb") == (document, binary)


def test_parts_become_links_with_their_joints_in_the_z_up_model_frame(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    document = segmented_glb(tmp_path / "item.glb")
    articulation = artiverse_articulation(document, {"articulations": JOINT_ENTRIES})
    assert articulation is not None
    assert articulation.links == {"base_1": ["base_1"], "door_2": ["door_2"], "drawer_3": ["drawer_3"], "knob_5": ["knob_5"], "lamp_arm_6": ["lamp_arm_6"], "cushion_7": ["cushion_7"], "knob_5_slide": [], "lamp_arm_6_axis1": []}
    joints = by_name(articulation)
    assert {name: (str(joint.type), joint.parent, joint.child) for name, joint in joints.items()} == {
        "door_2": ("revolute", "base_1", "door_2"),
        "drawer_3": ("prismatic", "base_1", "drawer_3"),
        "knob_5_slide": ("prismatic", "base_1", "knob_5_slide"),
        "knob_5_turn": ("continuous", "knob_5_slide", "knob_5"),
        "lamp_arm_6_axis1": ("revolute", "base_1", "lamp_arm_6_axis1"),
        "lamp_arm_6_axis2": ("revolute", "lamp_arm_6_axis1", "lamp_arm_6"),
        "cushion_7": ("fixed", "base_1", "cushion_7"),
    }
    door, drawer = joints["door_2"], joints["drawer_3"]
    assert (list(door.origin), list(door.axis), door.limit) == ([0.5, -0.25, 0.0], [0.0, 0.0, 1.0], (0.0, 1.5))
    assert (list(drawer.axis), drawer.limit) == ([0.0, -1.0, 0.0], (0.0, 0.3))
    assert (list(joints["knob_5_slide"].axis), joints["knob_5_slide"].limit, list(joints["knob_5_turn"].origin)) == ([0.0, 0.0, 1.0], (0.0, 0.1), [0.0, 0.0, 1.0])
    first, second = joints["lamp_arm_6_axis1"], joints["lamp_arm_6_axis2"]
    assert (list(first.axis), list(first.origin), first.limit) == ([1.0, 0.0, 0.0], [0.0, 0.0, 0.5], (-0.5, 0.5))
    assert (list(second.axis), list(second.origin), second.limit) == ([0.0, -1.0, 0.0], [0.0, 0.0, 0.5], (-0.2, 0.2))
    assert articulation.states == {"closed": {"door_2": 0.0, "drawer_3": 0.0}, "open": {"door_2": 1.5, "drawer_3": 0.3}}
    assert articulation.detents == {"door_2": {"ajar": 0.4}}


def test_motion_states_besides_closed_and_open_become_detents_unless_reserved(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    entries = [{**entry, "motionStates": {**entry["motionStates"], "Setting": {"value": 0.2}, "Half Open": {"value": 0.7}}} if entry["pid"] == 2 else entry for entry in JOINT_ENTRIES]
    articulation = artiverse_articulation(segmented_glb(tmp_path / "item.glb"), {"articulations": entries})
    assert articulation is not None
    assert articulation.detents == {"door_2": {"ajar": 0.4, "half_open": 0.7}}


def test_without_motion_states_doors_and_drawers_open_at_their_limits(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    entries = [{key: value for key, value in entry.items() if key != "motionStates"} for entry in JOINT_ENTRIES]
    articulation = artiverse_articulation(segmented_glb(tmp_path / "item.glb"), {"annotation": {"articulations": entries}})
    assert articulation is not None
    assert articulation.states == {"closed": {"door_2": 0.0, "drawer_3": 0.0}, "open": {"door_2": 1.5, "drawer_3": 0.3}}
    assert articulation.detents == {}


def test_moving_top_part_hangs_on_a_base_link_and_unknown_types_are_refused(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    document = segmented_glb(tmp_path / "item.glb")
    document["extras"]["flattened_graph"] = {str(pid): {"parent": -1} for pid in PARTS}
    articulation = artiverse_articulation(document, {"articulations": JOINT_ENTRIES})
    assert articulation is not None
    assert articulation.root == "base"
    assert (str(by_name(articulation)["base_1"].type), by_name(articulation)["base_1"].parent) == ("fixed", "base")
    with pytest.raises(ValueError, match="part door_2 has joint type 'spherical'"):
        artiverse_articulation(document, {"articulations": [{"pid": 2, "type": "spherical"}]})
    assert artiverse_articulation(document, {"articulations": [{"pid": 2, "type": "fixed"}]}) is None


def test_stage_renames_the_part_nodes_and_writes_the_articulation(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    source = stage(ITEM, item_files(tmp_path / "src"), tmp_path / "dataset")
    assert sorted(path.name for path in source.iterdir()) == [ANNOTATION_NAME, f"artiverse_{ITEM}.glb"]
    document, _ = read_glb(source / f"artiverse_{ITEM}.glb")
    assert [node["name"] for node in document["nodes"]] == ["node_0", "base_1", "door_2", "drawer_3", "knob_5", "lamp_arm_6", "cushion_7", "node_7"]
    annotation = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert annotation["desc"] == "Artiverse object with base, cushion, door, drawer, knob, lamp arm"
    assert len(annotation["articulation"]["joints"]) == 7


def test_stage_describes_a_categorized_item_by_its_category(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    source = stage(f"wall_cabinet/{ITEM}", item_files(tmp_path / "src"), tmp_path / "dataset")
    assert source.name == f"artiverse_{ITEM}"
    annotation = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert annotation["desc"] == "wall cabinet with base, cushion, door, drawer, knob, lamp arm"
    assert "origin::artiverse" in annotation["tags"]


def test_staged_glb_loads_with_every_link_node_in_blender(blender_python, tmp_path):
    source = stage(ITEM, item_files(tmp_path / "src"), tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    objects = scene(blender_python, source / f"artiverse_{ITEM}.glb")
    assert unmatched(objects, articulation) == []
    assert owners(objects, articulation) == {"base_1": "base_1", "door_2": "door_2", "node_7": "door_2", "drawer_3": "drawer_3", "knob_5": "knob_5", "lamp_arm_6": "lamp_arm_6", "cushion_7": "cushion_7"}
    door = next(joint for joint in articulation["joints"] if joint["name"] == "door_2")
    assert objects["door_2"]["location"] == pytest.approx(door["origin"], abs=1e-6)


ROOT = f"data/armoire/3dw/{ITEM}"
OTHER = "data/desk/abo/zz9"


class Counting(io.BytesIO):
    """A byte stream that counts what its reader took."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.taken = 0

    def read(self, size: int | None = -1) -> bytes:
        chunk = super().read(size)
        self.taken += len(chunk)
        return chunk


def chunk(members: dict[str, bytes]) -> bytes:
    """A release chunk: a tar.gz of the given files with their folders, in the given order."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            folder = tarfile.TarInfo(name.rsplit("/", 1)[0])
            folder.type = tarfile.DIRTYPE
            tar.addfile(folder)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_release_chunks_follow_the_release_order():
    releases = {"versions": [{"id": "v0.1", "chunks_dir": "dataset_chunks"}, {"id": "v1.2", "chunks_dir": "dataset_chunks_v1.2"}]}
    manifests = {
        "dataset_chunks": {"chunks": [{"archive": "a-1.tar.gz", "roots": [ROOT]}, {"archive": "a-2.tar.gz", "roots": [OTHER]}]},
        "dataset_chunks_v1.2": {"chunks": [{"archive": "fix.tar.gz", "roots": [ROOT, OTHER]}]},
    }
    assert release_chunks(releases, manifests) == [("dataset_chunks/a-1.tar.gz", [ROOT]), ("dataset_chunks/a-2.tar.gz", [OTHER]), ("dataset_chunks_v1.2/fix.tar.gz", [ROOT, OTHER])]


def test_extract_takes_one_folder_and_reads_no_further(tmp_path):
    data = chunk({f"{ROOT}/{ITEM}{GLB}": b"glb", f"{ROOT}/{ITEM}{JOINTS}": b"{}", f"{ROOT}/imgs/000.png": b"png", f"{OTHER}/zz9{GLB}": np.random.default_rng(0).bytes(1 << 20)})
    stream = Counting(data)
    assert extract(stream, ROOT, tmp_path / "out") == [GLB, JOINTS]
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [f"{ITEM}{JOINTS}", f"{ITEM}{GLB}"]
    assert stream.taken < len(data) // 2


def importer(host: Host, tmp_path: Path) -> Artiverse:
    return Artiverse(cache=tmp_path / "cache", provider=HFHub(REPO, revision=REVISION, host=host.base))


def resolve(relpath: str) -> str:
    return f"/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(relpath)}"


def serve_releases(host: Host, folder: Path) -> None:
    """A base release holding the item and another model, and a correction that replaces the item's articulations."""
    joints = (folder / f"{ITEM}{JOINTS}").read_bytes()
    host.pages[resolve(RELEASES)] = json.dumps({"versions": [{"id": "v0.1", "chunks_dir": "dataset_chunks"}, {"id": "v1.2", "chunks_dir": "dataset_chunks_v1.2"}]}).encode()
    host.pages[resolve(f"dataset_chunks/{MANIFEST}")] = json.dumps({"chunks": [{"archive": "base.tar.gz", "roots": [ROOT, OTHER]}]}).encode()
    host.pages[resolve(f"dataset_chunks_v1.2/{MANIFEST}")] = json.dumps({"chunks": [{"archive": "fix.tar.gz", "roots": [ROOT]}]}).encode()
    host.pages[resolve("dataset_chunks/base.tar.gz")] = chunk({f"{ROOT}/{ITEM}{GLB}": (folder / f"{ITEM}{GLB}").read_bytes(), f"{ROOT}/{ITEM}{JOINTS}": b'{"articulations": []}', f"{OTHER}/zz9{GLB}": b"glb"})
    host.pages[resolve("dataset_chunks_v1.2/fix.tar.gz")] = chunk({f"{ROOT}/{ITEM}{JOINTS}": joints, f"{ROOT}/mass_physics_v1.2.json": b"{}"})


def test_import_lists_by_category_and_fetches_through_the_release_layers(host, tmp_path):
    item_files(tmp_path / "src")
    serve_releases(host, tmp_path / "src")
    imp = importer(host, tmp_path)
    assert list(imp.items()) == [f"armoire/{ITEM}", "desk/zz9"]
    glb = imp.fetch(f"armoire/{ITEM}")
    assert glb.read_bytes() == (tmp_path / "src" / f"{ITEM}{GLB}").read_bytes()
    assert json.loads(glb.with_name(f"{ITEM}{JOINTS}").read_text(encoding="utf-8-sig"))["articulations"][0]["pid"] == 2
    served = len(host.served)
    assert imp.fetch(ITEM) == glb
    assert len(host.served) == served
    with pytest.raises(KeyError, match="no Artiverse object 'nope'.*import object artiverse"):
        imp.fetch("nope")


def test_a_model_without_segmented_parts_is_refused_by_name(host, tmp_path):
    item_files(tmp_path / "src")
    serve_releases(host, tmp_path / "src")
    with pytest.raises(ValueError, match=r"Artiverse object desk/zz9 ships no \.segmented\.glb with \.articulations\.json"):
        importer(host, tmp_path).fetch("desk/zz9")


def test_gated_releases_end_with_the_request_access_line(host, tmp_path):
    host.statuses[resolve(RELEASES)] = 403
    with pytest.raises(PermissionError, match=rf"Artiverse is gated .* request access at https://huggingface.co/datasets/{REPO} "):
        list(importer(host, tmp_path).items())


@pytest.mark.network
def test_live_releases_list_every_model_by_category(tmp_path):
    imp = Artiverse(cache=tmp_path / "cache")
    try:
        items = list(imp.items())
    except PermissionError as error:
        pytest.skip(f"no access to {REPO}: {error}")
    assert len(items) > 5000
    assert "armoire/7402940980c4a422cb8fd9b8e3886ff7" in items
