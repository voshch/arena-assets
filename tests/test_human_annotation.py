from arena_assets.impl import AssetType, Face, converter
from arena_assets.impl.build import DatabaseBuilder
from arena_assets.impl.build.HumanDatabaseBuilder import (
    HumanAnnotation,
    HumanDatabaseBuilder,
)
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.human import Legs, PelvisClearance


def test_builder_factory_resolves_human():
    builder_cls = DatabaseBuilder.Builder(AssetType.HUMAN)
    assert builder_cls is HumanDatabaseBuilder
    assert builder_cls._annotation_cls is HumanAnnotation
    assert builder_cls._DISCOVER_PATH == "Human"


def test_human_annotation_structures_from_yaml_dict():
    raw = {
        "name": "Nurse01",
        "path": "Human/Nurse01",
        "desc": "hospital nurse",
        "tags": ["human::role::nurse", "human::gender::female", "domain::hospital"],
        "face": "+x",
        "note": "",
        "bounding_box": [[-0.3, 0.3], [-0.3, 0.3], [0.0, 1.7]],
    }
    annotation = converter.structure(raw, HumanAnnotation)
    assert annotation.name == "Nurse01"
    assert annotation.face is Face.POS_X
    assert annotation.facets["human::role"] == ["nurse"]
    assert annotation.bounding_box.max_z == 1.7


def test_human_annotation_metadata_roundtrip():
    annotation = HumanAnnotation(
        name="Nurse01",
        path="Human/Nurse01",
        tags=["human::role::nurse", "human::gender::female"],
        bounding_box=BoundingBox(((-0.3, 0.3), (-0.3, 0.3), (0.0, 1.7))),
        note="carries a clipboard",
    )
    restored = HumanAnnotation.from_metadata(annotation.as_metadata)
    assert restored == annotation


def test_human_annotation_fields():
    fields = {field.name for field in HumanAnnotation.__attrs_attrs__}
    assert fields == {
        "name",
        "path",
        "desc",
        "tags",
        "asa",
        "bounding_box",
        "face",
        "note",
        "pelvis_clearance",
        "legs",
    }


def test_human_builder_registers_previews():
    assert "previews" in DatabaseBuilder.get_all_options()[AssetType.HUMAN]


def test_body_measurements_roundtrip_and_are_left_out_until_measured():
    annotation = HumanAnnotation(name="Nurse01", path="Human/Nurse01", face=Face.POS_X)
    assert not {"pelvis_clearance", "legs"} & set(converter.unstructure(annotation))
    assert not {"pelvis_clearance", "legs"} & set(annotation.as_metadata)
    annotation.pelvis_clearance = PelvisClearance(sit=0.12, lie=0.11)
    annotation.legs = Legs(hip=0.015, thigh=0.36, shin=0.37, ankle=0.08)
    assert converter.unstructure(annotation)["pelvis_clearance"] == {"sit": 0.12, "lie": 0.11}
    assert converter.unstructure(annotation)["legs"] == {"hip": 0.015, "thigh": 0.36, "shin": 0.37, "ankle": 0.08}
    assert converter.structure({**converter.unstructure(annotation), "name": "Nurse01", "path": "Human/Nurse01"}, HumanAnnotation) == annotation
    assert HumanAnnotation.from_metadata(annotation.as_metadata) == annotation
    annotation.pelvis_clearance = PelvisClearance(sit=0.12, lie=0.11, seated=((0.0, 0.14), (1.5708, 0.12)))
    assert converter.unstructure(annotation)["pelvis_clearance"]["seated"] == [(0.0, 0.14), (1.5708, 0.12)]
    assert converter.structure({**converter.unstructure(annotation), "name": "Nurse01", "path": "Human/Nurse01"}, HumanAnnotation) == annotation
    assert HumanAnnotation.from_metadata(annotation.as_metadata) == annotation
