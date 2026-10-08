from arena_assets.impl import Annotation


def test_metadata_roundtrip():
    annotation = Annotation(
        name="OakWood_01",
        path="materials/oak",
        desc="oak wood planks",
        tags=["wood", "floor"],
    )
    restored = Annotation.from_metadata(annotation.as_metadata)
    assert restored == annotation


def test_name_text_splits_and_strips_counter():
    annotation = Annotation(name="OakWood_01", path="materials/oak")
    assert annotation.name_text == "Oak Wood"


def test_name_text_keeps_meaningful_digits():
    assert Annotation(name="Table_200cm", path="m/t").name_text == "Table 200cm"
    assert Annotation(name="01", path="m/01").name_text == "01"


def test_as_text_includes_name_and_fields():
    annotation = Annotation(name="OakWood_01", path="materials/oak", tags=["floor", "color::brown"])
    text = annotation.as_text
    assert "Oak Wood" in text
    assert "brown" in text
    assert "floor" in text


def test_as_text_skips_empty_parts():
    annotation = Annotation(name="Brick", path="materials/brick")
    assert annotation.as_text == "Brick"


def test_as_metadata_stamps_asa_and_defaults_to_zero():
    assert Annotation(name="Brick", path="materials/brick").as_metadata["asa"] == 0
    stamped = Annotation(name="Brick", path="materials/brick", asa=1)
    assert stamped.as_metadata["asa"] == 1
    assert Annotation.from_metadata(stamped.as_metadata).asa == 1


def test_as_metadata_adds_facet_columns():
    annotation = Annotation(
        name="Brick",
        path="materials/brick",
        tags=["domain::office", "kind::brick", "legacybare"],
    )
    metadata = annotation.as_metadata
    assert metadata["domain"] == "office"
    assert metadata["kind"] == "brick"
    assert "legacybare" not in metadata


def test_facets_property_groups_by_key():
    annotation = Annotation(
        name="Brick",
        path="materials/brick",
        tags=["domain::office", "domain::hospital"],
    )
    assert annotation.facets == {"domain": ["office", "hospital"]}
