import xml.etree.ElementTree as ET

from arena_assets.utils.ModelConverter import sdf_model


def test_sdf_model_is_static_and_collides_with_its_visual_mesh():
    model = ET.fromstring(sdf_model("SM_Shelf_06f")).find("model")
    assert model.get("name") == "SM_Shelf_06f"
    assert model.findtext("static") == "true"
    link = model.find("link")
    for element in ("visual", "collision"):
        assert link.findtext(f"{element}/geometry/mesh/uri") == "SM_Shelf_06f.dae"


def test_sdf_model_gives_each_glow_mesh_a_visual_and_a_collision():
    link = ET.fromstring(sdf_model("lamp", [("glow_Shade", "lamp.glow_0.dae", "")])).find("model/link")
    assert [(element.tag, element.get("name"), element.findtext("geometry/mesh/uri")) for element in link] == [
        ("visual", "visual", "lamp.dae"),
        ("collision", "collision", "lamp.dae"),
        ("visual", "glow_Shade", "lamp.glow_0.dae"),
        ("collision", "collision_glow_0", "lamp.glow_0.dae"),
    ]
    only_glow = ET.fromstring(sdf_model("lamp", [("glow_Shade", "lamp.glow_0.dae", "")], plain=False)).find("model/link")
    assert [element.get("name") for element in only_glow] == ["glow_Shade", "collision_glow_0"]
