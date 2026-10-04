import math

import pytest

from arena_assets.impl import Face, converter
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.utils.geom import BoundingBox


def make_annotation():
    return ObjectAnnotation(
        name="OfficeChair",
        path="objects/chair",
        desc="ergonomic office chair",
        tags=[
            "chair",
            "office",
            "hoi::sit",
            "material::plastic",
            "material::fabric",
            "color::black",
        ],
        face=Face.POS_X,
        note="wheels included",
        bounding_box=BoundingBox(((-0.5, 0.5), (-0.4, 0.4), (0.0, 1.2))),
    )


def test_metadata_roundtrip():
    annotation = make_annotation()
    restored = ObjectAnnotation.from_metadata(annotation.as_metadata)
    assert restored == annotation
    assert restored.as_metadata == annotation.as_metadata


def test_metadata_extents():
    metadata = make_annotation().as_metadata
    assert metadata["width"] == 1.0
    assert metadata["depth"] == pytest.approx(0.8)
    assert metadata["height"] == 1.2
    assert metadata["volume"] == pytest.approx(0.96)


def test_parse_face():
    parse = ObjectAnnotation._parse_face
    assert parse(None) is Face.NEG_Y
    assert parse("") is Face.NEG_Y
    assert parse("+x") is Face.POS_X
    assert parse("garbage") is Face.NEG_Y


def test_parse_face_legacy_angles():
    for face in Face:
        assert ObjectAnnotation._parse_face(face.angle) is face
        assert ObjectAnnotation._parse_face(str(face.angle)) is face


def test_face_angle_is_back_azimuth():
    fronts = {
        Face.POS_X: (1.0, 0.0),
        Face.NEG_X: (-1.0, 0.0),
        Face.POS_Y: (0.0, 1.0),
        Face.NEG_Y: (0.0, -1.0),
        Face.XY: (math.sqrt(0.5), math.sqrt(0.5)),
    }
    for face, (x, y) in fronts.items():
        theta = math.radians(face.angle)
        assert math.isclose(-math.cos(theta), x, abs_tol=1e-9), face
        assert math.isclose(-math.sin(theta), y, abs_tol=1e-9), face


def test_as_text():
    text = make_annotation().as_text
    for part in (
        "Office Chair",
        "plastic",
        "black",
        "office",
        "ergonomic office chair",
        "wheels included",
    ):
        assert part in text


def test_structures_yaml_with_unknown_keys():
    """Unknown top-level keys must be ignored extra keys, not structuring errors."""
    data = {
        "name": "Armchair",
        "path": "Object/Armchair",
        "desc": "Armchair",
        "tags": ["office"],
        "color": ["black", "silver"],
        "material": ["leather", "metal"],
        "hoi": [],
        "face": "+x",
        "note": "",
        "bounding_box": [[-0.41, 0.41], [-0.52, 0.52], [0.0, 0.79]],
    }
    annotation = converter.structure(data, ObjectAnnotation)
    assert annotation.name == "Armchair"
    assert annotation.tags == ["office"]
    assert annotation.face is Face.POS_X
    assert annotation.asa == 0
