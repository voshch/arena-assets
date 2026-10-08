import math

import pytest

from arena_assets.impl import Annotation, AssetType
from arena_assets.impl.query import _lexical_rank, query_database


def test_top_n_ranked(database_path):
    results = query_database(str(database_path), AssetType.MATERIAL, "wooden floor", n=2)
    assert len(results) == 2
    assert results[0][0].path == "materials/oak"
    scores = [score for _, score in results]
    assert scores == sorted(scores)


def test_default_returns_single(database_path):
    results = query_database(str(database_path), AssetType.MATERIAL, "metal plate")
    assert len(results) == 1
    assert results[0][0].path == "materials/steel"


def test_empty_database_raises(tmp_path):
    with pytest.raises(ValueError):
        query_database(str(tmp_path), AssetType.MATERIAL, "anything")


def test_lexical_rank_exact_name_first():
    corpus = [
        Annotation(name="SoftLeather", path="m/soft", tags=["leather"]),
        Annotation(name="HardLeather", path="m/hard", tags=["leather"]),
        Annotation(name="OakWood", path="m/oak", tags=["wood"]),
    ]
    ranked = _lexical_rank("HardLeather", corpus)
    assert ranked[0].path == "m/hard"
    assert all(annotation.path != "m/oak" for annotation in ranked)


def test_lexical_rank_matches_split_name_and_tags():
    corpus = [
        Annotation(name="OakWood_01", path="m/oak"),
        Annotation(name="SteelPlate", path="m/steel", tags=["wood"]),
    ]
    ranked = _lexical_rank("oak wood", corpus)
    assert ranked[0].path == "m/oak"


def test_lexical_rank_no_overlap_is_empty():
    corpus = [Annotation(name="OakWood", path="m/oak")]
    assert _lexical_rank("ceramic tile", corpus) == []


def test_exact_name_query_ranks_first(database_path):
    results = query_database(str(database_path), AssetType.MATERIAL, "RedBrick", n=1)
    assert results[0][0].path == "materials/brick"
    assert results[0][1] != math.inf


def test_where_filter(database_path):
    results = query_database(
        str(database_path),
        AssetType.OBJECT,
        "chair to sit on",
        n=2,
        where={"height": {"$lt": 1.0}},
    )
    assert [annotation.path for annotation, _ in results] == ["objects/chair"]
