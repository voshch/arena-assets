import io
import json
import math
import urllib.parse
from pathlib import Path

import pytest
import yaml
from PIL import Image

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS
from arena_assets.importers.artvip import REPO, REVISION, ArtVIP, link_name, model_paths
from arena_assets.importers.providers import HFHub
from blender_scene import owners, scene, unmatched
from http_host import Host

pytest.importorskip("pxr")

FOLDER = "Articulated_objects/small_furniture/cabinet/cabinet_1"
ITEM = "cabinet_1"
POINTS = ", ".join(f"({x}, {y}, {z})" for x in (0, 0.3) for y in (0, 0.02) for z in (0, 0.35))
FACES = "int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n            int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]"
MODEL = """#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    subLayers = [@./resource/link_0.usda@]
    upAxis = "Z"
)

over "root"
{
    over "E_body_2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
    )
    {
        def PhysicsFixedJoint "FixedJoint"
        {
            rel physics:body1 = </root/E_body_2>
        }
    }

    over "E_door_3" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def PhysicsRevoluteJoint "RevoluteJoint_middle"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </root/E_body_2>
            rel physics:body1 = </root/E_door_3>
            point3f physics:localPos0 = (0.3, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            float physics:lowerLimit = -90
            float physics:upperLimit = 0
        }
    }
}
"""
GEOMETRY = f"""#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "root"
{{
    def Xform "E_body_2"
    {{
        def Mesh "P_9" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            {FACES}
            point3f[] points = [{POINTS}]
            rel material:binding = </root/Looks/Wood>
        }}
    }}

    def Xform "E_door_3"
    {{
        double3 xformOp:translate = (0.3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "P_14"
        {{
            {FACES}
            point3f[] points = [{POINTS}]
        }}
    }}

    def Scope "Looks"
    {{
        def Material "Wood"
        {{
            token outputs:surface.connect = </root/Looks/Wood/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </root/Looks/Wood/Albedo.outputs:rgb>
                token outputs:surface
            }}

            def Shader "Albedo"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./img/wood.png@
                float3 outputs:rgb
            }}
        }}
    }}
}}
"""


def _png() -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (8, 8), (150, 110, 60)).save(data, format="PNG")
    return data.getvalue()


def repository(host: Host, geometry: str = GEOMETRY) -> dict[str, bytes]:
    """Serve an ArtVIP-shaped repository: one cabinet whose model stage sublayers its geometry, which binds a texture, beside a base variant, a thumbnail and a second object."""
    files = {
        f"{FOLDER}/model_cabinet_1.usd": MODEL.encode(),
        f"{FOLDER}/resource/link_0.usda": geometry.encode(),
        f"{FOLDER}/resource/img/wood.png": _png(),
        f"{FOLDER}/origin/model_cabinet_1_base.usd": MODEL.encode(),
        f"{FOLDER}/.thumbs/256x256/model_cabinet_1.usd.png": _png(),
        "Articulated_objects/Ikea_furniture/EKET_Cabinet_35x35cm/model_EKET_Cabinet_35x35cm.usd": MODEL.encode(),
        "Articulated_objects/README.md": b"# Articulated Objects\n",
    }
    tree = f"/api/datasets/{REPO}/tree/{REVISION}"
    for prefix in ("Articulated_objects", FOLDER):
        host.pages[f"{tree}/{urllib.parse.quote(prefix)}?recursive=true"] = json.dumps([{"type": "file", "path": path} for path in files if path.startswith(f"{prefix}/")]).encode()
    for path, body in files.items():
        host.pages[f"/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(path)}"] = body
    return files


def importer(host: Host, tmp_path: Path) -> ArtVIP:
    return ArtVIP(cache=tmp_path / "cache", provider=HFHub(REPO, revision=REVISION, host=host.base))


def served(host: Host) -> set[str]:
    return {urllib.parse.unquote(path).split(f"{FOLDER}/", 1)[1] for path in host.served if "/resolve/" in path and FOLDER in urllib.parse.unquote(path)}


def test_registry_resolves_artvip_to_its_pinned_repository():
    assert IMPORTERS.load("object", "artvip") is ArtVIP
    assert IMPORTERS.meta("object", "artvip").license.value == "yes"
    assert ArtVIP.default_provider().url("README.md") == f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"


def test_items_are_the_object_folders_with_a_model_stage():
    paths = model_paths(
        [
            "Articulated_objects/small_furniture/chair/office chair005/model_office_chair002.usd",
            "Articulated_objects/small_furniture/chair/office chair005/origin/model_office_chair002_base.usd",
            "Articulated_objects/small_furniture/chair/office chair005/.thumbs/256x256/model_office_chair002.usd.png",
            "Articulated_objects/household_items/basket/basket_1/model_basket_1.usda",
            "Articulated_objects/README.md",
        ]
    )
    assert paths == {"basket_1": "Articulated_objects/household_items/basket/basket_1/model_basket_1.usda", "office_chair005": "Articulated_objects/small_furniture/chair/office chair005/model_office_chair002.usd"}
    with pytest.raises(ValueError, match="both name the ArtVIP item basket_1"):
        model_paths(["Articulated_objects/a/basket_1/model_a.usd", "Articulated_objects/b/basket_1/model_b.usd"])


def test_listing_names_both_objects(host, tmp_path):
    repository(host)
    assert list(importer(host, tmp_path).items()) == [ITEM, "eket_cabinet_35x35cm"]


def test_fetch_follows_the_stage_to_its_layers_and_textures_only(host, tmp_path):
    repository(host)
    stage = importer(host, tmp_path).fetch(ITEM)
    assert stage.name == "model_cabinet_1.usd"
    assert served(host) == {"model_cabinet_1.usd", "resource/link_0.usda", "resource/img/wood.png"}


def test_fetch_finds_a_texture_named_by_a_search_path(host, tmp_path):
    repository(host, GEOMETRY.replace("@./img/wood.png@", "@img/wood.png@"))
    importer(host, tmp_path).fetch(ITEM)
    assert served(host) == {"model_cabinet_1.usd", "resource/link_0.usda", "resource/img/wood.png"}


def test_fetch_of_an_unknown_item_names_the_listing_command(host, tmp_path):
    repository(host)
    with pytest.raises(KeyError, match="no ArtVIP object 'sofa_1'.*import object artvip"):
        importer(host, tmp_path).fetch("sofa_1")


def test_stage_writes_the_door_joint_with_its_open_and_closed_states(host, tmp_path):
    repository(host)
    imp = importer(host, tmp_path)
    source = imp.stage(ITEM, imp.fetch(ITEM), tmp_path / "dataset")
    annotation = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert annotation["desc"] == "cabinet 1"
    assert "origin::artvip" in annotation["tags"]
    articulation = annotation["articulation"]
    assert articulation["links"] == {"body_2": ["E_body_2"], "door_3": ["E_door_3"]}
    assert articulation["joints"] == [{"name": "revolute_joint_middle", "type": "revolute", "parent": "body_2", "child": "door_3", "origin": [0.3, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "limit": [-1.570796, 0.0], "role": "door"}]
    assert articulation["states"] == {"closed": {"revolute_joint_middle": 0.0}, "open": {"revolute_joint_middle": pytest.approx(-math.pi / 2, abs=1e-6)}}
    assert imp.attribution(ITEM).url == f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{FOLDER}"


def test_staged_cabinet_loads_with_every_link_node_in_blender(blender_python, host, tmp_path):
    repository(host)
    imp = importer(host, tmp_path)
    source = imp.stage(ITEM, imp.fetch(ITEM), tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    objects = scene(blender_python, source / "model_cabinet_1.usd")
    assert unmatched(objects, articulation) == []
    assert owners(objects, articulation) == {"P_9": "body_2", "P_14": "door_3"}


def test_link_names_drop_the_element_prefix():
    assert [link_name(prim) for prim in ("E_door_3", "E_EKET_Cabinet_2", "base")] == ["door_3", "eket_cabinet_2", "base"]


@pytest.mark.network
def test_live_eket_cabinet_door_opens_and_loads_in_blender(blender_python, tmp_path):
    item = "eket_cabinet_with_door_brown_walnut_effect_35x35x35cm"
    imp = ArtVIP(cache=tmp_path / "cache")
    source = imp.stage(item, imp.fetch(item), tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    assert [(joint["type"], joint["child"], joint["limit"]) for joint in articulation["joints"]] == [("revolute", "door_3", [-1.570796, 0.0])]
    assert sorted(articulation["states"]) == ["closed", "open"]
    objects = scene(blender_python, next(source.glob("model_*.usd")))
    assert unmatched(objects, articulation) == []
    assert set(owners(objects, articulation).values()) == set(articulation["links"])
