import pytest

from arena_assets.impl import DATABASE_NAME, Annotation
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.utils.Database import Database
from arena_assets.utils.geom import BoundingBox

MATERIALS = [
    Annotation(
        name="OakWood",
        path="materials/oak",
        desc="light brown oak wood planks",
        tags=["wood"],
    ),
    Annotation(name="RedBrick", path="materials/brick", desc="red clay brick wall"),
    Annotation(name="SteelPlate", path="materials/steel", desc="brushed steel metal plate"),
]

OBJECTS = [
    ObjectAnnotation(
        name="OfficeChair",
        path="objects/chair",
        desc="ergonomic office chair",
        tags=["chair"],
        bounding_box=BoundingBox(((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.9))),
    ),
    ObjectAnnotation(
        name="BarStool",
        path="objects/stool",
        desc="tall wooden bar stool",
        tags=["chair"],
        bounding_box=BoundingBox(((-0.2, 0.2), (-0.2, 0.2), (0.0, 1.1))),
    ),
]


@pytest.fixture(scope="session")
def database_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    db = Database(path / DATABASE_NAME)
    for annotation in MATERIALS:
        db.store("material", annotation)
    for annotation in OBJECTS:
        db.store("object", annotation)
    return path
