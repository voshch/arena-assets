from arena_assets.impl import DATABASE_NAME, Annotation
from arena_assets.utils.Database import Database


def test_store_upserts_by_path(tmp_path):
    db = Database(tmp_path / DATABASE_NAME)
    db.store("material", Annotation(name="OakWood", path="materials/oak", desc="oak"))
    db.store(
        "material",
        Annotation(name="OakWood", path="materials/oak", desc="updated oak"),
    )

    result = db.list_all("material")
    assert len(result["ids"]) == 1
    assert result["metadatas"][0]["desc"] == "updated oak"


def test_list_all(database_path):
    db = Database(database_path / DATABASE_NAME)
    result = db.list_all("material")
    assert sorted(result["ids"]) == [
        "materials/brick",
        "materials/oak",
        "materials/steel",
    ]


def test_query_returns_requested_count(database_path):
    db = Database(database_path / DATABASE_NAME)
    result = db.query("material", "wooden floor", 2)
    assert len(result["metadatas"][0]) == 2
