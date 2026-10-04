from arena_assets.impl import AssetType, Face, converter
from arena_assets.impl.build import DatabaseBuilder
from arena_assets.impl.build.HumanDatabaseBuilder import (
    HumanAnnotation,
    HumanDatabaseBuilder,
)
from arena_assets.utils.geom import BoundingBox


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


def test_human_annotation_adds_no_new_fields():
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
    }


def test_human_builder_registers_previews():
    assert "previews" in DatabaseBuilder.get_all_options()[AssetType.HUMAN]
