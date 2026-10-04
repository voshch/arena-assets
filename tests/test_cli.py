from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.cli.db.query import parse_filters

runner = CliRunner()


def test_parse_filters():
    assert parse_filters([]) is None
    assert parse_filters(["height<0.75"]) == {"height": {"$lt": 0.75}}
    assert parse_filters(["height<=1", "volume>0.1"]) == {
        "$and": [{"height": {"$lte": 1.0}}, {"volume": {"$gt": 0.1}}]
    }


def test_parse_filters_equality_on_strings_and_booleans():
    assert parse_filters(["kind=alarm"]) == {"kind": {"$eq": "alarm"}}
    assert parse_filters(["loop=true", "kind = motor"]) == {"$and": [{"loop": {"$eq": True}}, {"kind": {"$eq": "motor"}}]}


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("Arena Assets v")


def test_invalid_log_level():
    result = runner.invoke(app, ["--log-level", "bogus", "version"])
    assert result.exit_code == 1


def test_db_query_outputs_path(database_path):
    result = runner.invoke(
        app, ["-s", "db", str(database_path), "query", "material", "wooden floor"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == str(database_path / "materials/oak")


def test_db_query_filter(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "object",
            "chair to sit on",
            "-n",
            "2",
            "--filter",
            "height<1.0",
        ],
    )
    assert result.exit_code == 0
    assert result.output.strip() == str(database_path / "objects/chair")


def test_db_query_invalid_filter(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "object",
            "chair",
            "--filter",
            "height!!1",
        ],
    )
    assert result.exit_code != 0


def test_db_query_scores(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "material",
            "wooden floor",
            "-n",
            "2",
            "--scores",
        ],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        path, score = line.split("\t")
        float(score)
