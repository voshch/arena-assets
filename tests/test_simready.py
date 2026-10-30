import io
import json
import urllib.parse
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.providers import HFHub
from arena_assets.importers.simready import KEPT, SimReady, SimReadyDSX, description, joint_name, kind, link_name
from blender_scene import owners, scene, unmatched
from http_host import Host

ITEM = "bin_waste_a01"
ROOT = f"simready_usd/sm_{ITEM}_01.usd"
SMALL_ITEM = "bandage_medical_a01"


def _png(size: tuple[int, int]) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(data, format="PNG")
    return data.getvalue()


def _stage(edge_m: float) -> str:
    points = ", ".join(f"({x}, {y}, {z})" for x in (0, edge_m) for y in (0, edge_m) for z in (0, 0.3))
    return f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root"
{{
    def Mesh "Body" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
        point3f[] points = [{points}]
        rel material:binding = </Root/Looks/Plastic>
    }}

    def Scope "Looks"
    {{
        def Material "Plastic"
        {{
            token outputs:surface.connect = </Root/Looks/Plastic/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Looks/Plastic/Albedo.outputs:rgb>
                float inputs:roughness.connect = </Root/Looks/Plastic/Orm.outputs:g>
                token outputs:surface
            }}

            def Shader "Albedo"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./Textures/t_plastic_alb.png@
                float3 outputs:rgb
            }}

            def Shader "Orm"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./source_assets/t_plastic_orm.png@
                float outputs:g
            }}
        }}
    }}
}}
"""


def package(host: Host, item: str = ITEM, repo: str = SimReady.REPO, revision: str = SimReady.REVISION, license: str = "CC-BY-4.0", edge_m: float = 0.25) -> dict[str, bytes]:
    """Serve one SimReady package the way the Hugging Face tree and resolve endpoints do."""
    files = {
        "com.nvidia.simready.packaging.json": json.dumps({"format_version": "1.0", "license": license}).encode(),
        ".metadata/com.nvidia.simready.root_usds.json": json.dumps({"format_version": "1.0", "entries": [ROOT.replace(ITEM, item)]}).encode(),
        ".metadata/com.nvidia.simready.conformance.Package@1.0.0.json": json.dumps({"profile": "Package"}).encode(),
        ".metadata/LICENSE.txt": b"Attribution 4.0 International\n",
        ROOT.replace(ITEM, item): _stage(edge_m).encode(),
        "simready_usd/payloads/Physics/physics.usda": b'#usda 1.0\n\nover "Root"\n{\n    float physics:density = 900\n}\n',
        "simready_usd/Textures/t_plastic_alb.png": _png((4100, 4)),
        "simready_usd/source_assets/t_plastic_orm.png": _png((16, 16)),
        "simready_usd/source_assets/t_plastic_unused.png": _png((16, 16)),
        "simready_usd/.thumbs/256x256/preview.png": _png((4, 4)),
    }
    tree = f"/api/datasets/{repo}/tree/{revision}"
    host.pages[f"{tree}?recursive=false"] = json.dumps([{"type": "file", "path": ".gitattributes"}, {"type": "directory", "path": item}]).encode()
    host.pages[f"{tree}/{item}?recursive=true"] = json.dumps([{"type": "file", "path": f"{item}/{path}"} for path in files]).encode()
    for path, body in files.items():
        host.pages[f"/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(f'{item}/{path}')}"] = body
    return files


ARTICULATED = "bed_test_a01"
BODIES = {"base_01": (0.0, 0.0, 0.0), "wheelPin_01": (0.9, -0.3, 0.15), "wheel_01": (0.95, -0.3, 0.07), "brake_01": (-0.98, 0.0, 0.12), "baseGlass_01": (0.0, 0.0, 0.0)}


def _body(name: str, at: tuple[float, float, float]) -> str:
    points = ", ".join(f"({x}, {y}, {z})" for x in (0, 0.05) for y in (0, 0.05) for z in (0, 0.05))
    return f"""
        def Xform "{name}_obj_00"
        {{
            double3 xformOp:translate = ({at[0]}, {at[1]}, {at[2]})
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Mesh "{name}_mesh_00"
            {{
                int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
                int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
                point3f[] points = [{points}]
            }}

            def Scope "VisualMaterials"
            {{
            }}
        }}
"""


def _joint(kind: str, name: str, body0: str, body1: str, position: tuple[float, ...], axis: str = "Z", limits: tuple[str, str] | None = None) -> str:
    fields = [
        f"rel physics:body0 = </RootNode/Geometry/{body0}_obj_00/{body0}_mesh_00>",
        f"rel physics:body1 = </RootNode/Geometry/{body1}_obj_00/{body1}_mesh_00>",
        f"point3f physics:localPos0 = {position}",
        "point3f physics:localPos1 = (0, 0, 0)",
        "quatf physics:localRot0 = (1, 0, 0, 0)",
        *([f'uniform token physics:axis = "{axis}"'] if kind != "PhysicsFixedJoint" else []),
        *([f"float physics:lowerLimit = {limits[0]}", f"float physics:upperLimit = {limits[1]}"] if limits else []),
    ]
    body = "".join(f"            {field}\n" for field in fields)
    return f'        def {kind} "joint_{name}_joint_00"\n        {{\n{body}        }}\n'


def articulated_package(host: Host) -> dict[str, bytes]:
    """Serve a SimReady package laid out like bed_hospital_standard_a01: instanced body Xforms holding one mesh each, joints in the physics payload naming the meshes."""
    files = package(host, item=ARTICULATED)
    root = ROOT.replace(ITEM, ARTICULATED)
    bodies = "".join(_body(name, at) for name, at in BODIES.items())
    files[root] = f"""#usda 1.0
(
    defaultPrim = "RootNode"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RootNode" (
    prepend payload = @./payloads/Physics/physics.usda@
)
{{
    def Xform "Geometry"
    {{{bodies}    }}
}}
""".encode()
    pin, wheel, brake = (BODIES[name] for name in ("wheelPin_01", "wheel_01", "brake_01"))
    overs = "".join(f'        over "{name}_obj_00" (\n            prepend apiSchemas = ["PhysicsRigidBodyAPI"]\n        )\n        {{\n        }}\n' for name in BODIES)
    joints = "".join(
        [
            _joint("PhysicsRevoluteJoint", "wheelPin_01", "base_01", "wheelPin_01", pin, "Z", ("-inf", "inf")),
            _joint("PhysicsRevoluteJoint", "wheel_01", "wheelPin_01", "wheel_01", tuple(round(w - p, 6) for w, p in zip(wheel, pin, strict=True)), "Y", ("-inf", "inf")),
            _joint("PhysicsRevoluteJoint", "brake_01", "base_01", "brake_01", brake, "X", ("0", "30")),
            _joint("PhysicsFixedJoint", "baseGlass_01", "base_01", "baseGlass_01", (0.0, 0.0, 0.0)),
        ]
    )
    files["simready_usd/payloads/Physics/physics.usda"] = f"""#usda 1.0
(
    defaultPrim = "RootNode"
    metersPerUnit = 1
    upAxis = "Z"
)

over "RootNode"
{{
    over "Geometry"
    {{
{overs}    }}

    def Scope "Joints"
    {{
{joints}    }}
}}
""".encode()
    for path, body in files.items():
        host.pages[f"/datasets/{SimReady.REPO}/resolve/{SimReady.REVISION}/{urllib.parse.quote(f'{ARTICULATED}/{path}')}"] = body
    return files


def importer(host: Host, tmp_path: Path) -> SimReady:
    return SimReady(cache=tmp_path / "cache", provider=HFHub(SimReady.REPO, revision=SimReady.REVISION, host=host.base))


def served(host: Host) -> set[str]:
    return {urllib.parse.unquote(path).split(f"{ITEM}/", 1)[-1] for path in host.served if "/resolve/" in path}


def test_registry_resolves_both_sources_to_their_pinned_repositories():
    assert IMPORTERS.load("object", "simready") is SimReady
    assert IMPORTERS.load("object", "simready_dsx") is SimReadyDSX
    assert (SimReady.REPO, SimReadyDSX.REPO) == ("nvidia/simready-assets", "nvidia/simready-dsx")
    assert IMPORTERS.meta("object", "simready").listing_ttl_s is None


@pytest.mark.parametrize(
    ("item", "words", "category"), [("bed_hospital_standard_a01", "bed hospital standard", "bed"), ("IV_pole_a01", "IV pole", "iv"), ("appliance_handDryer_automatic_b01", "appliance hand dryer automatic", "appliance"), ("handTool_wrench_a03", "hand tool wrench", "hand_tool"), ("GB300", "GB300", "gb300")]
)
def test_description_and_kind_come_from_the_folder_name(item, words, category):
    assert (description(item), kind(item)) == (words, category)


def test_items_are_the_asset_folders(host, tmp_path):
    package(host)
    assert list(importer(host, tmp_path).items()) == [ITEM]


def test_fetch_takes_the_layers_and_only_the_files_the_stage_refers_to(host, tmp_path):
    package(host)
    dest = importer(host, tmp_path).fetch(ITEM)
    assert served(host) == {
        "com.nvidia.simready.packaging.json",
        ".metadata/com.nvidia.simready.root_usds.json",
        ".metadata/com.nvidia.simready.conformance.Package@1.0.0.json",
        ".metadata/LICENSE.txt",
        ROOT,
        "simready_usd/payloads/Physics/physics.usda",
        "simready_usd/Textures/t_plastic_alb.png",
        "simready_usd/source_assets/t_plastic_orm.png",
    }
    assert (dest / "simready_usd" / "source_assets" / "t_plastic_orm.png").is_file()


def test_fetch_refuses_a_package_under_another_license_before_its_textures(host, tmp_path):
    package(host, license="CC-BY-NC-4.0")
    with pytest.raises(ValueError, match="bin_waste_a01 is under CC-BY-NC-4.0, only CC-BY-4.0 SimReady assets are imported"):
        importer(host, tmp_path).fetch(ITEM)
    assert not any(path.endswith(".png") for path in served(host))


def test_fetch_of_an_unknown_folder_names_the_listing_command(host, tmp_path):
    package(host)
    host.pages[f"/api/datasets/{SimReady.REPO}/tree/{SimReady.REVISION}/nothing_a01?recursive=true"] = b"[]"
    with pytest.raises(KeyError, match="no SimReady asset 'nothing_a01'.*import object simready"):
        importer(host, tmp_path).fetch("nothing_a01")


def test_import_refuses_a_footprint_the_builder_would_rescale_and_drops_the_download(host, tmp_path):
    package(host, edge_m=6.0)
    imp = importer(host, tmp_path)
    with pytest.raises(ValueError, match=r"bin_waste_a01 is 6.00 x 6.00 m, the object builder rescales a footprint over 5 m"):
        imp.import_item(ITEM, tmp_path / "out")
    assert not imp.package(ITEM).exists()


def test_import_builds_the_object_and_keeps_the_simready_metadata_and_physics(blender, host, tmp_path):
    package(host)
    imp = importer(host, tmp_path)
    target = imp.import_item(ITEM, tmp_path / "out")
    assert target == tmp_path / "out" / "Common" / "Object" / ITEM
    assert {f"{ITEM}.usdz", f"{ITEM}.fbx", f"{ITEM}.sdf"} <= {path.name for path in target.iterdir()}
    annotation = yaml.safe_load((target / ANNOTATION_NAME).read_text())
    assert "origin::simready" in annotation["tags"]
    assert annotation["desc"] == "bin waste"
    kept = target / KEPT
    assert sorted(path.relative_to(kept).as_posix() for path in kept.rglob("*") if path.is_file()) == ["com.nvidia.simready.packaging.json", "metadata/LICENSE.txt", "metadata/com.nvidia.simready.conformance.Package@1.0.0.json", "metadata/com.nvidia.simready.root_usds.json", "payloads/Physics/physics.usda"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.authors) == ("CC-BY-4.0", "NVIDIA Corporation")
    assert attribution.url == f"https://huggingface.co/datasets/{SimReady.REPO}/tree/{SimReady.REVISION}/{ITEM}"
    with zipfile.ZipFile(target / f"{ITEM}.usdz") as usdz:
        sizes = [Image.open(io.BytesIO(usdz.read(name))).size for name in usdz.namelist() if name.endswith((".png", ".jpg"))]
    assert sizes
    assert max(max(size) for size in sizes) == 2048
    assert not imp.package(ITEM).exists()


def test_link_and_joint_names_drop_the_simready_suffixes():
    assert [link_name(prim) for prim in ("wheelPin_01_obj_00", "base_01_obj_00", "lid")] == ["wheel_pin_01", "base_01", "lid"]
    assert [joint_name(prim) for prim in ("joint_wheelPin_01_joint_00", "joint_brake_01_joint_00", "hinge")] == ["wheel_pin_01", "brake_01", "hinge"]


def test_stage_writes_no_articulation_for_a_rigid_package(host, tmp_path):
    package(host)
    imp = importer(host, tmp_path)
    imp.fetch(ITEM)
    source = imp.stage(ITEM, tmp_path / "dataset")
    assert "articulation" not in yaml.safe_load((source / ANNOTATION_NAME).read_text())


def test_stage_writes_the_physics_payload_joints_as_articulation(host, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    articulated_package(host)
    imp = importer(host, tmp_path)
    imp.fetch(ARTICULATED)
    source = imp.stage(ARTICULATED, tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    assert articulation["links"] == {"base_01": ["base_01_obj_00"], "wheel_pin_01": ["wheelPin_01_obj_00"], "wheel_01": ["wheel_01_obj_00"], "brake_01": ["brake_01_obj_00"], "base_glass_01": ["baseGlass_01_obj_00"]}
    joints = {joint["name"]: joint for joint in articulation["joints"]}
    assert {name: (joint["type"], joint["parent"], joint["child"]) for name, joint in joints.items()} == {
        "wheel_pin_01": ("continuous", "base_01", "wheel_pin_01"),
        "wheel_01": ("continuous", "wheel_pin_01", "wheel_01"),
        "brake_01": ("revolute", "base_01", "brake_01"),
        "base_glass_01": ("fixed", "base_01", "base_glass_01"),
    }
    assert joints["wheel_01"]["origin"] == pytest.approx(list(BODIES["wheel_01"]))
    assert joints["wheel_01"]["axis"] == [0.0, 1.0, 0.0]
    assert joints["brake_01"]["limit"] == pytest.approx([0.0, 0.523599])
    assert "states" not in articulation


def test_staged_package_loads_with_every_link_node_in_blender(blender_python, host, tmp_path):
    articulated_package(host)
    imp = importer(host, tmp_path)
    imp.fetch(ARTICULATED)
    source = imp.stage(ARTICULATED, tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    objects = scene(blender_python, source / Path(ROOT).name.replace(ITEM, ARTICULATED))
    assert unmatched(objects, articulation) == []
    assert owners(objects, articulation) == {f"{name}_mesh_00": link_name(f"{name}_obj_00") for name in BODIES}


@pytest.mark.network
def test_live_hospital_bed_keeps_its_ten_joints_and_loads_them_in_blender(blender_python, tmp_path):
    item = "bed_hospital_standard_a01"
    imp = SimReady(cache=tmp_path / "cache")
    imp.fetch(item)
    source = imp.stage(item, tmp_path / "dataset")
    articulation = yaml.safe_load((source / ANNOTATION_NAME).read_text())["articulation"]
    kinds = sorted(joint["type"] for joint in articulation["joints"])
    assert kinds == ["continuous"] * 8 + ["fixed", "revolute"]
    assert sorted(articulation["links"]) == ["base_01", "base_glass_01", "brake_01", *(f"wheel_0{n}" for n in range(1, 5)), *(f"wheel_pin_0{n}" for n in range(1, 5))]
    objects = scene(blender_python, source / imp.root(item).name)
    assert unmatched(objects, articulation) == []
    assert set(owners(objects, articulation).values()) == set(articulation["links"])


@pytest.mark.network
def test_live_sources_list_their_assets(tmp_path):
    assets = list(SimReady(cache=tmp_path / "assets").items())
    assert len(assets) > 2500
    assert {"bed_hospital_standard_a01", "IV_pole_a01", ITEM} <= set(assets)
    assert list(SimReadyDSX(cache=tmp_path / "dsx").items()) == ["GB300"]


@pytest.mark.network
def test_live_package_fetches_its_license_root_and_textures(tmp_path):
    imp = SimReady(cache=tmp_path / "cache")
    dest = imp.fetch(SMALL_ITEM)
    assert json.loads((dest / "com.nvidia.simready.packaging.json").read_text())["license"] == "CC-BY-4.0"
    assert imp.root(SMALL_ITEM).is_file()
    assert any((dest / "simready_usd" / "Textures").iterdir())
