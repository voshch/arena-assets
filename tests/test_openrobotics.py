import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation, ObjectDatabaseBuilder
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution, write_attribution
from arena_assets.importers.gso import FuelObjects
from arena_assets.importers.openrobotics import OpenRobotics, asset_name, wanted
from arena_assets.semantic import CURRENT_ASA
from arena_assets.utils.Database import Database
from cli_output import plain

ITEM = "aws_robomaker_warehouse_Lamp_01"
FIXTURE = Path(__file__).parent / "fixtures" / "openrobotics" / f"{ITEM}.zip"
RELPATH = Path("Common") / "Object" / ITEM
VISUAL = f"meshes/{ITEM}_visual.dae"
TEXTURE = f"materials/textures/{ITEM}.png"
COLLADA = "{http://www.collada.org/2005/11/COLLADASchema}"
EXTENTS = (1.0222, 1.0067, 1.5179)
SPAN_X = (-0.5108, 0.5114)
CC_BY = "Creative Commons Attribution 4.0 International"
CC0 = "Creative Commons Zero v1.0 Universal"
CC_BY_SA = "Creative Commons Attribution Share Alike 4.0 International"
CC_BY_NC_ND = "Creative Commons Attribution Non Commercial No Derivatives 4.0 International"
METADATA = {"name": ITEM, "owner": "OpenRobotics", "license_name": CC_BY, "tags": ["warehouse"]}
QUARTER = repr(math.pi / 2)
OBJ_ITEM = "OfficeChairBlack"
OBJ_EXTENTS = (0.7355, 0.7531, 1.2462)
DAE_ITEM = "foldable_table"
DAE_EXTENTS = (1.829, 0.762, 0.6724)

CRATE = """mtllib crate.mtl
o crate
v 0 0 0
v 1 0 0
v 1 2 0
v 0 2 0
v 0 0 3
v 1 0 3
v 1 2 3
v 0 2 3
vn 0 0 1
usemtl crate
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""
CRATE_SDF = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="Crate">
    <link name="body">
      <pose>0 0 0 0 0 {QUARTER}</pose>
      <visual name="visual">
        <pose>0.25 0 0 0 0 0</pose>
        <geometry><mesh><uri>model://Crate/meshes/crate.obj</uri><scale>0.5 0.5 0.5</scale></mesh></geometry>
      </visual>
      <collision name="collision">
        <geometry><mesh><uri>model://Crate/meshes/crate_Col.obj</uri></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
CRATE_CONFIG = """<?xml version="1.0"?>
<model>
  <name>Crate</name>
  <author><name>Jane Roe</name><email>jane.roe@example.org</email></author>
  <author><name>someone@example.org</name></author>
  <author><name>name</name><email>name@example.org</email></author>
  <description>A crate, questions to crates@example.org</description>
</model>
"""


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PIXEL = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff")) + chunk(b"IEND", b"")
CRATE_MEMBERS = {
    "model.sdf": CRATE_SDF,
    "model.config": CRATE_CONFIG,
    "meshes/crate.obj": CRATE,
    "meshes/crate.mtl": "newmtl crate\nmap_Kd crate.png\n",
    "meshes/crate.png": PIXEL,
    "meshes/crate_Col.obj": CRATE,
    "meshes/crate_Col.mtl": "newmtl None\n",
    "thumbnails/1.png": PIXEL,
}


def importer(tmp_path: Path) -> OpenRobotics:
    return IMPORTERS.load("object", "openrobotics")(cache=tmp_path / "cache")


def archive(path: Path, members: dict[str, str | bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    return path


def refitted(path: Path, members: dict[str, str | bytes]) -> Path:
    with zipfile.ZipFile(FIXTURE) as fixture:
        return archive(path, {name: fixture.read(name) for name in fixture.namelist() if not name.endswith("/")} | members)


def seed(tmp_path: Path, item: str = ITEM, metadata: dict | None = None, src: Path = FIXTURE) -> Path:
    cached = tmp_path / "cache" / f"{item}.zip"
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, cached)
    cached.with_suffix(".json").write_text(json.dumps(METADATA if metadata is None else metadata))
    return cached


def unpainted(tmp_path: Path) -> Path:
    with zipfile.ZipFile(FIXTURE) as fixture:
        mesh = fixture.read(f"meshes/{ITEM}_visual.DAE").decode()
        sdf = fixture.read("model.sdf").decode()
    mesh = re.sub(r"<library_images>.*?</library_images>", "", mesh, flags=re.DOTALL)
    mesh = re.sub(r"<diffuse>.*?</diffuse>", "<diffuse><color>1 1 1 1</color></diffuse>", mesh, flags=re.DOTALL)
    sdf = sdf.replace("</geometry>\n        <meta>", f"</geometry><material><pbr><metal><albedo_map>{TEXTURE}</albedo_map></metal></pbr></material>\n        <meta>")
    assert "albedo_map" in sdf and "<texture" not in mesh
    return refitted(tmp_path / "upstream" / f"{ITEM}.zip", {"model.sdf": sdf, f"meshes/{ITEM}_visual.DAE": mesh})


def staged(tmp_path: Path) -> Path:
    return OpenRobotics.stage(ITEM, seed(tmp_path), tmp_path / "dataset")


def crate(tmp_path: Path, members: dict[str, str | bytes] | None = None) -> Path:
    src = archive(tmp_path / "upstream" / "Crate.zip", CRATE_MEMBERS | (members or {}))
    return seed(tmp_path, "Crate", {"name": "Crate", "license_name": CC0, "categories": ["Furniture and Home"]}, src)


def convert(tmp_path: Path, item: str = ITEM) -> Path:
    out = tmp_path / "openrobotics"
    imp = importer(tmp_path)
    assert imp.convert(item, imp.fetch(item), out) == str(Path("Common") / "Object" / item)
    return out


def box(directory: Path):
    annotation = ObjectDatabaseBuilder.read_annotation_file(directory)
    assert isinstance(annotation, ObjectAnnotation)
    return annotation.bounding_box


def extents(directory: Path) -> tuple[float, float, float]:
    bounds = box(directory)
    return (bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y, bounds.max_z - bounds.min_z)


def files(directory: Path) -> list[str]:
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())


def lines(mesh: Path, key: str) -> list[tuple[float, ...]]:
    return [tuple(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith(f"{key} ")]


def test_registry_resolves_openrobotics_on_the_shared_fuel_path():
    cls = IMPORTERS.load("object", "openrobotics")
    assert cls is OpenRobotics
    assert cls.meta.kind == "object"
    assert issubclass(cls, FuelObjects)
    assert cls.stage.__func__ is IMPORTERS.load("object", "gso").stage.__func__
    assert cls.convert is FuelObjects.convert
    assert cls.default_provider().url("Dining Chair/tip/Dining Chair.zip") == "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Dining%20Chair/tip/Dining%20Chair.zip"


@pytest.mark.parametrize(
    ("model", "listed"),
    [
        ({"name": "OfficeChairBlack", "license_name": CC_BY, "categories": ["Furniture and Home"]}, True),
        ({"name": "VendingMachine", "license_name": CC_BY, "categories": ["Appliance"]}, True),
        ({"name": "KitchenSink", "license_name": CC_BY, "categories": ["Furniture and Home", "Kitchen"]}, True),
        ({"name": "CGMClassic", "license_name": CC_BY, "tags": ["hospital", "bed"]}, True),
        ({"name": ITEM, "license_name": CC_BY, "tags": ["Warehouse"]}, True),
        ({"name": "Reflective table", "license_name": CC0, "categories": ["Furniture and Home"], "tags": ["furniture", "PBR"]}, True),
        ({"name": "X1 Config 1", "license_name": CC_BY, "categories": ["Robots"], "tags": ["subt", "ugv"]}, False),
        ({"name": "HospitalRobotCharger", "license_name": CC_BY, "categories": ["Appliance", "Robots"], "tags": ["hospital"]}, False),
        ({"name": "Urban Straight", "license_name": CC_BY, "tags": ["subt", "tile"]}, False),
        ({"name": "Depot", "license_name": CC_BY, "categories": ["Architecture", "Places and Landscapes"], "tags": ["warehouse"]}, False),
        ({"name": "Warehouse", "license_name": CC_BY_NC_ND, "tags": ["warehouse"]}, False),
        ({"name": "Share alike sofa", "license_name": CC_BY_SA, "categories": ["Furniture and Home"]}, False),
        ({"name": "Cessna C-172", "license_name": CC0, "categories": ["Cars and Vehicles"]}, False),
        ({"name": "Coke", "license_name": CC0}, False),
    ],
)
def test_wanted_lists_free_household_hospital_and_warehouse_models_only(model, listed):
    assert wanted(model) is listed


def test_asset_name_has_no_space():
    assert asset_name("Dining Chair") == "Dining_Chair"
    assert asset_name("DRC Practice: Wheel valve wall") == "DRC_Practice_Wheel_valve_wall"
    assert asset_name("Submarine (buoyant)") == "Submarine_buoyant"
    assert asset_name(ITEM) == ITEM


def test_item_id_and_fuel_name_both_name_the_same_object(tmp_path):
    metadata = METADATA | {"name": "Dining Chair"}
    seed(tmp_path, "Dining_Chair", metadata)
    seed(tmp_path, "Dining Chair", metadata)
    imp = importer(tmp_path)
    assert imp.model("Dining_Chair") == imp.model("Dining Chair") == "Dining Chair"
    assert imp.asset("Dining_Chair") == imp.asset("Dining Chair") == "Dining_Chair"
    assert imp.attribution("Dining_Chair").url == "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Dining%20Chair"


@pytest.mark.parametrize("license_name", [CC_BY_SA, CC_BY_NC_ND])
def test_other_licenses_are_refused_before_anything_is_written(tmp_path, license_name):
    seed(tmp_path, metadata=METADATA | {"license_name": license_name})
    out = tmp_path / "openrobotics"
    with pytest.raises(ValueError, match=f"{ITEM} is under {license_name}, only CC0 and CC BY 4.0 models are imported"):
        importer(tmp_path).import_item(ITEM, out)
    assert not out.exists()
    assert not (tmp_path / "cache" / "stage").exists()


@pytest.mark.parametrize(("license_name", "recorded"), [(CC_BY, "CC-BY-4.0"), (CC0, "CC0-1.0")])
def test_attribution_carries_the_license_of_the_model(tmp_path, license_name, recorded):
    seed(tmp_path, metadata=METADATA | {"license_name": license_name})
    attribution = importer(tmp_path).attribution(ITEM)
    assert attribution.license == recorded
    assert attribution.source == "openrobotics"
    assert attribution.url == f"https://fuel.gazebosim.org/1.0/OpenRobotics/models/{ITEM}"
    assert attribution.authors == "Open Robotics"


def test_attribution_names_the_authors_without_an_email_address(tmp_path):
    crate(tmp_path)
    attribution = importer(tmp_path).attribution("Crate")
    assert attribution.authors == "Jane Roe, Open Robotics"
    assert attribution.license == "CC0-1.0"
    write_attribution(tmp_path / ATTRIBUTION_FILE, attribution)
    assert "@" not in (tmp_path / ATTRIBUTION_FILE).read_text()


def test_stage_writes_no_email_address(tmp_path):
    source = OpenRobotics.stage("Crate", crate(tmp_path), tmp_path / "dataset")
    assert yaml.safe_load((source / ANNOTATION_NAME).read_text())["desc"] == "A crate, questions to"
    assert not [path for path in source.rglob("*") if path.suffix in (".yaml", ".obj", ".mtl") and "@" in path.read_text()]


def test_stage_holds_the_annotation_and_the_visual_mesh_only(tmp_path):
    assert files(staged(tmp_path)) == [ANNOTATION_NAME, TEXTURE, VISUAL]


def test_stage_keeps_the_collada_geometry_and_restates_its_material_for_the_builder(tmp_path):
    source = staged(tmp_path)
    root = ET.parse(source / VISUAL).getroot()
    with zipfile.ZipFile(FIXTURE) as fixture:
        upstream = ET.fromstring(fixture.read(f"meshes/{ITEM}_visual.DAE"))
    assert [array.text for array in root.iter(f"{COLLADA}float_array")] == [array.text for array in upstream.iter(f"{COLLADA}float_array")]
    assert [matrix.text for matrix in root.iter(f"{COLLADA}matrix")] == [matrix.text for matrix in upstream.iter(f"{COLLADA}matrix")]
    assert root.findtext(f"{COLLADA}asset/{COLLADA}up_axis") == "Z_UP"
    assert [image.findtext(f"{COLLADA}init_from") for image in root.iter(f"{COLLADA}image")] == [f"../{TEXTURE}"]
    assert (source / "meshes" / f"../{TEXTURE}").is_file()
    image = root.find(f".//{COLLADA}image").get("id")
    assert [texture.get("texture") for texture in root.iter(f"{COLLADA}texture")] == [f"{image}-sampler"]
    assert [parameter.get("sid") for parameter in root.iter(f"{COLLADA}newparam")] == [f"{image}-surface", f"{image}-sampler"]
    assert root.findtext(f".//{COLLADA}surface/{COLLADA}init_from") == image
    assert root.findtext(f".//{COLLADA}sampler2D/{COLLADA}source") == f"{image}-surface"
    assert not list(root.iter(f"{COLLADA}transparent")) + list(root.iter(f"{COLLADA}transparency"))
    ids = [element.get("id") for element in root.iter() if element.get("id") is not None]
    assert not [name for name in ids if re.fullmatch(r"[A-Za-z_][\w.-]*", name) is None]
    links = [value[1:] for element in root.iter() for value in element.attrib.values() if value.startswith("#")]
    assert root.find(f".//{COLLADA}instance_material").get("target")[1:] == root.find(f".//{COLLADA}material").get("id")
    assert not set(links) - set(ids)


@pytest.mark.parametrize(
    ("material", "opacity"),
    [
        ('<transparent opaque="RGB_ZERO"><color>0.5 0.5 0.5 1</color></transparent><transparency><float>1.0</float></transparency>', [0.5]),
        ('<transparent opaque="RGB_ZERO"><color>0 0 0 1</color></transparent><transparency><float>1.0</float></transparency>', []),
        ('<transparent opaque="A_ONE"><color>1 1 1 0.25</color></transparent>', [0.25]),
        ('<transparent><color>1 1 1 1</color></transparent><transparency><float>0.0</float></transparency>', []),
        ("<transparency><float>0.0</float></transparency>", []),
        ("<transparency><float>0.6</float></transparency>", []),
    ],
)
def test_stage_states_the_opacity_gazebo_reads_from_a_collada_material(tmp_path, material, opacity):
    with zipfile.ZipFile(FIXTURE) as fixture:
        mesh = re.sub(r"<transparent.*?</transparency>", material, fixture.read(f"meshes/{ITEM}_visual.DAE").decode(), flags=re.DOTALL)
    assert material in mesh
    src = seed(tmp_path, src=refitted(tmp_path / "upstream" / f"{ITEM}.zip", {f"meshes/{ITEM}_visual.DAE": mesh}))
    root = ET.parse(OpenRobotics.stage(ITEM, src, tmp_path / "dataset") / VISUAL).getroot()
    assert not list(root.iter(f"{COLLADA}transparent"))
    assert [float(element.findtext(f"{COLLADA}float")) for element in root.iter(f"{COLLADA}transparency")] == pytest.approx(opacity)


def test_stage_paints_an_untextured_collada_material_with_the_sdf_albedo_map(tmp_path):
    src = seed(tmp_path, src=unpainted(tmp_path))
    source = OpenRobotics.stage(ITEM, src, tmp_path / "dataset")
    root = ET.parse(source / VISUAL).getroot()
    assert [(image.get("id"), image.findtext(f"{COLLADA}init_from")) for image in root.iter(f"{COLLADA}image")] == [("albedo", f"../{TEXTURE}")]
    assert [texture.get("texture") for texture in root.iterfind(f".//{COLLADA}diffuse/{COLLADA}texture")] == ["albedo-sampler"]
    assert root.findtext(f".//{COLLADA}surface/{COLLADA}init_from") == "albedo"


def test_stage_annotates_with_taxonomy_tags(tmp_path):
    raw = yaml.safe_load((staged(tmp_path) / ANNOTATION_NAME).read_text())
    assert raw["asa"] == CURRENT_ASA
    assert raw["desc"] == "aws robomaker warehouse Lamp 01"
    assert raw["tags"] == ["domain::warehouse", "kind::lamp", "origin::openrobotics"]


def test_stage_takes_the_kind_from_the_first_category_when_the_name_gives_none(tmp_path):
    src = seed(tmp_path, "CArm", {"name": "CArm", "license_name": CC_BY, "categories": ["Electronics", "Appliance"], "tags": ["hospital", "xray"]})
    raw = yaml.safe_load((OpenRobotics.stage("CArm", src, tmp_path / "dataset") / ANNOTATION_NAME).read_text())
    assert raw["tags"] == ["domain::hospital", "kind::electronics", "origin::openrobotics", "xray"]


def test_stage_leaves_the_collision_obj_and_its_material_out(tmp_path):
    source = OpenRobotics.stage("Crate", crate(tmp_path), tmp_path / "dataset")
    assert files(source) == [ANNOTATION_NAME, "meshes/crate.mtl", "meshes/crate.obj", "meshes/crate.png"]


def test_stage_bakes_the_sdf_scale_and_pose_into_the_obj(tmp_path):
    source = OpenRobotics.stage("Crate", crate(tmp_path), tmp_path / "dataset")
    upstream = [tuple(map(float, line.split()[1:4])) for line in CRATE.splitlines() if line.startswith("v ")]
    assert lines(source / "meshes" / "crate.obj", "v") == [pytest.approx((-0.5 * y, 0.5 * z, -(0.5 * x + 0.25))) for x, y, z in upstream]
    assert lines(source / "meshes" / "crate.obj", "vn") == [pytest.approx((0.0, 2.0, 0.0))]


@pytest.mark.parametrize(
    ("sdf", "reason"),
    [
        (CRATE_SDF.replace("<mesh><uri>model://Crate/meshes/crate.obj</uri><scale>0.5 0.5 0.5</scale></mesh>", "<cylinder><radius>1</radius><length>1</length></cylinder>"), "a visual that is not a mesh"),
        (CRATE_SDF.replace("<visual name=", '<visual name="other"><geometry><mesh><uri>meshes/crate_Col.obj</uri></mesh></geometry></visual><visual name='), "shows 2 placed meshes"),
        (CRATE_SDF.replace("crate.obj", "absent.obj"), "which its archive does not hold"),
        (CRATE_SDF.replace("crate.obj", "crate.png"), "staged are .obj and .dae meshes"),
        ('<sdf version="1.6"><model name="Crate"><include><uri>model://other</uri></include></model></sdf>', "shows 0 placed meshes"),
    ],
)
def test_stage_refuses_a_model_that_is_not_one_mesh(tmp_path, sdf, reason):
    with pytest.raises(ValueError, match=reason):
        OpenRobotics.stage("Crate", crate(tmp_path, {"model.sdf": sdf}), tmp_path / "dataset")


def test_convert_builds_a_native_object_item_from_the_collada_visual(blender, tmp_path):
    seed(tmp_path)
    out = convert(tmp_path)
    target = out / RELPATH
    assert (target / f"{ITEM}.usdz").stat().st_size > 0
    assert (target / f"{ITEM}.fbx").stat().st_size > 0
    assert (target / f"{ITEM}.sdf" / f"{ITEM}.sdf").read_text().count(f"<uri>{ITEM}.dae</uri>") == 2
    assert (target / f"{ITEM}.sdf" / f"{ITEM}.dae").stat().st_size > 0
    for preview in (f"{ITEM}_thumb.png", f"{ITEM}_topdown.png"):
        assert (target / preview).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with zipfile.ZipFile(target / f"{ITEM}.usdz") as packed:
        assert f"textures/{ITEM}.png" in packed.namelist()
    assert not (tmp_path / "cache" / "stage" / ITEM).exists()


def test_convert_carries_the_sdf_albedo_map_into_the_object(blender, tmp_path):
    seed(tmp_path, src=unpainted(tmp_path))
    out = convert(tmp_path)
    with zipfile.ZipFile(out / RELPATH / f"{ITEM}.usdz") as packed:
        assert f"textures/{ITEM}.png" in packed.namelist()


def test_builder_takes_the_staged_visual_as_its_model_file(blender, tmp_path):
    source = staged(tmp_path)
    annotation = ObjectDatabaseBuilder.read_annotation_file(source)
    assert isinstance(annotation, ObjectAnnotation)
    assert ObjectDatabaseBuilder.find_mainfile(annotation) == VISUAL


def test_convert_annotation_is_measured_upright_in_meters(blender, tmp_path):
    seed(tmp_path)
    out = convert(tmp_path)
    bounds = box(out / RELPATH)
    assert extents(out / RELPATH) == pytest.approx(EXTENTS, abs=2e-4)
    assert (bounds.min_x, bounds.max_x) == pytest.approx(SPAN_X, abs=2e-4)
    assert bounds.min_z == pytest.approx(0.0, abs=1e-4)
    raw = yaml.safe_load((out / RELPATH / ANNOTATION_NAME).read_text())
    assert raw["path"] == f"Object/{ITEM}"
    assert raw["asa"] == CURRENT_ASA
    assert raw["tags"] == ["domain::warehouse", "kind::lamp", "origin::openrobotics"]


def test_convert_places_a_z_up_collada_mesh_by_the_sdf(blender, tmp_path):
    sdf = f"""<sdf version="1.9"><model name="{ITEM}"><link name="link"><pose>0 0 0 {QUARTER} 0 0</pose>
<visual name="visual"><pose>2 0 0 0 0 0</pose><geometry><mesh><uri>meshes/{ITEM}_visual.DAE</uri><scale>0.5 0.5 0.5</scale></mesh></geometry></visual>
</link></model></sdf>"""
    seed(tmp_path, src=refitted(tmp_path / "upstream" / f"{ITEM}.zip", {"model.sdf": sdf}))
    out = convert(tmp_path)
    bounds = box(out / RELPATH)
    assert extents(out / RELPATH) == pytest.approx((0.5 * EXTENTS[0], 0.5 * EXTENTS[2], 0.5 * EXTENTS[1]), abs=2e-4)
    assert (bounds.min_x, bounds.max_x) == pytest.approx((2 + 0.5 * SPAN_X[0], 2 + 0.5 * SPAN_X[1]), abs=2e-4)
    assert bounds.min_z == pytest.approx(0.0, abs=1e-4)


def test_convert_reads_a_y_up_collada_mesh_in_the_axes_gazebo_does(blender, tmp_path):
    with zipfile.ZipFile(FIXTURE) as fixture:
        mesh = fixture.read(f"meshes/{ITEM}_visual.DAE").replace(b"<up_axis>Z_UP</up_axis>", b"<up_axis>Y_UP</up_axis>")
    assert b"Y_UP" in mesh
    seed(tmp_path, src=refitted(tmp_path / "upstream" / f"{ITEM}.zip", {f"meshes/{ITEM}_visual.DAE": mesh}))
    out = convert(tmp_path)
    assert extents(out / RELPATH) == pytest.approx(EXTENTS, abs=2e-4)


def test_convert_measures_the_placed_obj(blender, tmp_path):
    crate(tmp_path)
    out = convert(tmp_path, "Crate")
    bounds = box(out / "Common" / "Object" / "Crate")
    assert (bounds.min_x, bounds.max_x) == pytest.approx((-1.0, 0.0), abs=1e-4)
    assert (bounds.min_y, bounds.max_y) == pytest.approx((0.25, 0.75), abs=1e-4)
    assert (bounds.min_z, bounds.max_z) == pytest.approx((0.0, 1.5), abs=1e-4)


def test_convert_refuses_a_footprint_the_builder_would_rescale(blender, tmp_path):
    crate(tmp_path, {"model.sdf": CRATE_SDF.replace("<scale>0.5 0.5 0.5</scale>", "<scale>3 3 3</scale>")})
    out = tmp_path / "openrobotics"
    imp = importer(tmp_path)
    with pytest.raises(ValueError, match="Crate measures 6.00 m by 3.00 m, the object builder rescales a footprint of that size to 0.06 m by 0.03 m"):
        imp.convert("Crate", imp.fetch("Crate"), out)
    assert not out.exists()
    assert not (tmp_path / "cache" / "stage" / "Crate").exists()


def test_convert_refuses_a_mesh_the_builder_loads_nothing_from(blender, tmp_path):
    crate(tmp_path, {"meshes/crate.obj": "mtllib crate.mtl\no crate\n"})
    imp = importer(tmp_path)
    with pytest.raises(ValueError, match="Crate: the builder's converter loads no geometry from crate.obj"):
        imp.convert("Crate", imp.fetch("Crate"), tmp_path / "openrobotics")


def test_convert_names_the_object_without_a_space(blender, tmp_path):
    seed(tmp_path, "Lamp 01", METADATA | {"name": "Lamp 01"})
    out = tmp_path / "openrobotics"
    imp = importer(tmp_path)
    assert imp.convert("Lamp 01", imp.fetch("Lamp 01"), out) == "Common/Object/Lamp_01"
    assert (out / "Common" / "Object" / "Lamp_01" / "Lamp_01.usdz").is_file()
    assert Database(out / "Common" / DATABASE_NAME).list_all("object")["ids"] == ["Object/Lamp_01"]


def test_import_item_from_the_cache_is_a_queryable_attributed_object(blender, tmp_path):
    seed(tmp_path)
    out = tmp_path / "openrobotics"
    target = importer(tmp_path).import_item(ITEM, out)
    assert target == out / RELPATH
    assert Database(out / "Common" / DATABASE_NAME).list_all("object")["ids"] == [f"Object/{ITEM}"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.source, attribution.license, attribution.authors) == ("openrobotics", "CC-BY-4.0", "Open Robotics")
    assert attribution.url == f"https://fuel.gazebosim.org/1.0/OpenRobotics/models/{ITEM}"


def test_import_command_writes_the_item_like_a_gso_one(blender, tmp_path):
    seed(tmp_path)
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["-s", "import", "object", "openrobotics", ITEM, "--out", str(out), "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(out / RELPATH)
    assert (out / RELPATH / ATTRIBUTION_FILE).is_file()
    assert (out / RELPATH / f"{ITEM}.usdz").is_file()


def test_import_command_names_the_license_it_refuses(tmp_path):
    seed(tmp_path, metadata=METADATA | {"license_name": CC_BY_SA})
    result = CliRunner().invoke(app, ["-s", "import", "object", "openrobotics", ITEM, "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")])
    assert result.exit_code == 2
    assert f"{ITEM} is under {CC_BY_SA}, only CC0 and CC BY 4.0 models are imported" in plain(result.output)
    assert not (tmp_path / "out").exists()


@pytest.mark.network
def test_items_are_the_wanted_models_of_the_live_listing(tmp_path):
    imp = importer(tmp_path)
    items = list(imp.items())
    assert len(items) == len(set(items))
    assert 100 <= len(items) <= 160
    assert {ITEM, OBJ_ITEM, DAE_ITEM, "Dining_Chair", "KitchenSink", "VendingMachine", "CGMClassic"} <= set(items)
    assert not {"Warehouse", "Depot", "HospitalRobotCharger", "X1", "Dining Chair"} & set(items)
    assert not [item for item in items if asset_name(item) != item]


@pytest.mark.network
def test_item_id_of_a_spaced_name_is_found_through_the_listing(tmp_path):
    imp = importer(tmp_path)
    assert imp.model("Dining_Chair") == "Dining Chair"
    assert imp.license("Dining_Chair") == "CC-BY-4.0"
    with pytest.raises(KeyError, match="no model 'No_Such_Model' of OpenRobotics on Fuel"):
        imp.metadata("No_Such_Model")


@pytest.mark.network
def test_fetch_refuses_a_model_under_another_license(tmp_path):
    with pytest.raises(ValueError, match=f"Warehouse is under {CC_BY_NC_ND}, only CC0 and CC BY 4.0 models are imported"):
        importer(tmp_path).fetch("Warehouse")
    assert files(tmp_path / "cache") == ["Warehouse.json"]


@pytest.mark.network
def test_stage_of_a_live_obj_model_leaves_the_collision_mesh_and_the_email_out(tmp_path):
    imp = importer(tmp_path)
    src = imp.fetch(OBJ_ITEM)
    with zipfile.ZipFile(src) as upstream:
        assert f"meshes/{OBJ_ITEM}_Col.obj" in upstream.namelist()
        assert b"@" in upstream.read("model.config")
    source = OpenRobotics.stage(OBJ_ITEM, src, tmp_path / "dataset")
    assert files(source) == [ANNOTATION_NAME, f"meshes/{OBJ_ITEM}.mtl", f"meshes/{OBJ_ITEM}.obj", f"meshes/{OBJ_ITEM}.png"]
    attribution = imp.attribution(OBJ_ITEM)
    assert attribution.authors == "Wan Yi Seow, Open Robotics"
    assert yaml.safe_load((source / ANNOTATION_NAME).read_text())["tags"] == ["domain::common", "kind::chair", "origin::openrobotics", "furniture_and_home"]


@pytest.mark.network
def test_import_item_end_to_end_from_an_obj_model(blender, tmp_path):
    out = tmp_path / "openrobotics"
    target = importer(tmp_path).import_item(OBJ_ITEM, out)
    assert target == out / "Common" / "Object" / OBJ_ITEM
    assert (target / f"{OBJ_ITEM}.usdz").is_file()
    assert extents(target) == pytest.approx(OBJ_EXTENTS, abs=2e-4)
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert attribution.license == "CC-BY-4.0"
    assert "@" not in (target / ATTRIBUTION_FILE).read_text()
    assert "@" not in (target / ANNOTATION_NAME).read_text()


@pytest.mark.network
def test_import_item_end_to_end_from_a_collada_model(blender, tmp_path):
    out = tmp_path / "openrobotics"
    target = importer(tmp_path).import_item(DAE_ITEM, out)
    assert target == out / "Common" / "Object" / DAE_ITEM
    assert (target / f"{DAE_ITEM}.usdz").is_file()
    assert extents(target) == pytest.approx(DAE_EXTENTS, abs=2e-4)
    assert box(target).min_z == pytest.approx(0.0, abs=1e-4)
    assert read_attribution(target / ATTRIBUTION_FILE).authors == "Rumman Waqar, Open Robotics"
