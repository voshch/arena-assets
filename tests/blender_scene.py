import json
import re
import subprocess
from pathlib import Path

from arena_assets.utils.ModelConverter import environment

SCRIPT = """
import json, sys
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
path = sys.argv[1]
ext = path.rsplit(".", 1)[-1].lower()
if ext.startswith("usd"):
    bpy.ops.wm.usd_import(filepath=path, support_scene_instancing=False, merge_parent_xform=False)
elif ext == "obj":
    bpy.ops.wm.obj_import(filepath=path)
else:
    bpy.ops.import_scene.gltf(filepath=path)
bpy.context.view_layer.update()
objects = {}
for obj in bpy.context.scene.objects:
    lineage, up = [], obj
    while up is not None:
        lineage.append(up.name)
        up = up.parent
    entry = {"type": obj.type, "lineage": lineage, "location": list(obj.matrix_world.translation)}
    if obj.type == "MESH" and obj.data.vertices:
        points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        entry["box"] = [[min(point[n] for point in points), max(point[n] for point in points)] for n in range(3)]
    entry["metallic"] = [node.inputs["Metallic"].default_value for slot in obj.material_slots if slot.material and slot.material.node_tree for node in slot.material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"]
    objects[obj.name] = entry
print("OBJECTS" + json.dumps(objects))
"""

DUPLICATE = re.compile(r"\.\d{3,}$")


def scene(interpreter: str, path: Path) -> dict[str, dict]:
    """Objects of a model as the Blender worker loads it for an articulated build, by name: type, names up the parent chain, world location, mesh box and material metallic values."""
    result = subprocess.run([interpreter, "-c", SCRIPT, str(path)], capture_output=True, text=True, env=environment(), check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith("OBJECTS"))
    return json.loads(line[len("OBJECTS") :])


def owners(objects: dict[str, dict], articulation: dict) -> dict[str, str]:
    """Link of every mesh object by the builder's rule: the link naming the object or its nearest listed ancestor, else the root link."""
    by_node = {node: link for link, nodes in articulation["links"].items() for node in nodes}
    children = {joint["child"] for joint in articulation["joints"]}
    root = next(link for link in articulation["links"] if link not in children)
    return {name: next((by_node[base] for up in entry["lineage"] if (base := DUPLICATE.sub("", up)) in by_node), root) for name, entry in objects.items() if entry["type"] == "MESH"}


def unmatched(objects: dict[str, dict], articulation: dict) -> list[str]:
    """Articulation nodes that name no object of the model nor an ancestor of one."""
    known = {DUPLICATE.sub("", up) for entry in objects.values() for up in entry["lineage"]}
    return sorted(node for nodes in articulation["links"].values() for node in nodes if node not in known)
