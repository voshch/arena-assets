import xml.etree.ElementTree as ET

from arena_assets.utils.ModelConverter import sdf_model


def test_sdf_model_is_static_and_collides_with_its_visual_mesh():
    model = ET.fromstring(sdf_model("SM_Shelf_06f")).find("model")
    assert model.get("name") == "SM_Shelf_06f"
    assert model.findtext("static") == "true"
    link = model.find("link")
    for element in ("visual", "collision"):
        assert link.findtext(f"{element}/geometry/mesh/uri") == "SM_Shelf_06f.dae"
