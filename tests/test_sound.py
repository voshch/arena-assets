import wave
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder
from arena_assets.impl.build.SoundDatabaseBuilder import (
    SoundAnnotation,
    SoundDatabaseBuilder,
)
from arena_assets.impl.listall import list_database
from arena_assets.impl.query import query_database

SOUNDS = {
    "alarm_loop": {
        "version": 2,
        "kind": "alarm",
        "level_db": 88.0,
        "reference_distance_m": 1.0,
        "normalize_dbfs": -17.5,
        "loop": True,
        "model": "wav_loop",
        "desc": "loud fire alarm siren",
        "tags": ["siren", "emergency"],
        "variants": [{"id": "alarm_loop_01", "file": "alarm_loop.wav", "tags": ["alarm"]}],
    },
    "footstep": {
        "version": 2,
        "kind": "footstep",
        "level_db": 45.0,
        "reference_distance_m": 1.0,
        "normalize_dbfs": -26.5,
        "loop": False,
        "model": "wav",
        "surface": "floor",
        "desc": "footsteps walking on a wooden floor",
        "variants": [
            {
                "id": "footstep_oak_planks_01",
                "file": "footstep_oak_planks.wav",
                "match": {"floor": ["oak"]},
                "tags": ["walk", "oak_planks"],
            },
            {
                "id": "footstep_default_01",
                "file": "footstep_default.wav",
                "default": True,
                "tags": ["walk", "default"],
            },
        ],
    },
}


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * 80)


def _write_sound(root: Path, name: str, manifest: dict) -> Path:
    directory = root / "Sound" / name
    directory.mkdir(parents=True)
    with open(directory / f"{name}.yaml", "w") as f:
        yaml.safe_dump(manifest, f)
    for variant in manifest.get("variants", []):
        if "file" in variant:
            _write_wav(directory / variant["file"])
    return directory


@pytest.fixture(scope="module")
def sound_db(tmp_path_factory):
    input_path = tmp_path_factory.mktemp("sound_input")
    output_path = tmp_path_factory.mktemp("sound_output")
    for name, manifest in SOUNDS.items():
        _write_sound(input_path, name, manifest)
    DatabaseBuilder.Builder(AssetType.SOUND)(input_path=input_path, output_path=output_path).build()
    return output_path


def test_builder_factory_resolves_sound():
    builder_cls = DatabaseBuilder.Builder(AssetType.SOUND)
    assert builder_cls is SoundDatabaseBuilder
    assert builder_cls._annotation_cls is SoundAnnotation
    assert builder_cls._DISCOVER_PATH == "Sound"


def test_sound_build_indexes_every_manifest_dir(sound_db):
    paths = sorted(annotation.path for annotation in list_database(str(sound_db), AssetType.SOUND))
    assert paths == ["Sound/alarm_loop", "Sound/footstep"]


def test_sound_build_copies_manifest_and_wavs(sound_db):
    assert sorted(path.name for path in (sound_db / "Sound" / "footstep").iterdir()) == [
        ANNOTATION_NAME,
        "footstep.yaml",
        "footstep_default.wav",
        "footstep_oak_planks.wav",
    ]


def test_sound_build_writes_annotation_derived_from_manifest(sound_db):
    with open(sound_db / "Sound" / "alarm_loop" / ANNOTATION_NAME) as f:
        annotation = yaml.safe_load(f)
    assert annotation["name"] == "alarm_loop"
    assert annotation["path"] == "Sound/alarm_loop"
    assert annotation["desc"] == "loud fire alarm siren"
    assert annotation["tags"] == ["siren", "emergency", "alarm"]
    assert annotation["kind"] == "alarm"
    assert annotation["loop"] is True
    assert annotation["level_db"] == 88.0
    assert annotation["variants"] == ["alarm_loop_01"]


def test_sound_annotation_tags_union_variant_tags_and_kind(sound_db):
    by_name = {annotation.name: annotation for annotation in list_database(str(sound_db), AssetType.SOUND)}
    assert by_name["footstep"].tags == ["walk", "oak_planks", "default", "footstep"]
    assert by_name["footstep"].variants == [
        "footstep_oak_planks_01",
        "footstep_default_01",
    ]


def test_sound_query_by_text_ranks_matching_sound_first(sound_db):
    results = query_database(str(sound_db), AssetType.SOUND, "emergency siren", n=2)
    assert results[0][0].path == "Sound/alarm_loop"


def test_sound_query_where_kind_filters(sound_db):
    results = query_database(
        str(sound_db),
        AssetType.SOUND,
        "emergency siren",
        n=2,
        where={"kind": "footstep"},
    )
    assert [annotation.path for annotation, _ in results] == ["Sound/footstep"]


def test_sound_query_where_loop_filters(sound_db):
    results = query_database(str(sound_db), AssetType.SOUND, "walking", n=2, where={"loop": True})
    assert [annotation.path for annotation, _ in results] == ["Sound/alarm_loop"]


def test_sound_annotation_metadata_roundtrip():
    annotation = SoundAnnotation(
        name="footstep",
        path="Sound/footstep",
        desc="steps",
        tags=["walk", "footstep"],
        kind="footstep",
        loop=False,
        level_db=45.0,
        variants=["footstep_oak_planks_01", "footstep_default_01"],
    )
    assert SoundAnnotation.from_metadata(annotation.as_metadata) == annotation


def test_cli_db_list_sound(sound_db):
    result = CliRunner().invoke(app, ["-s", "db", str(sound_db), "list", "sound"])
    assert result.exit_code == 0
    assert sorted(result.output.split()) == [
        str(sound_db / "Sound/alarm_loop"),
        str(sound_db / "Sound/footstep"),
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 1}, "version must be 2"),
        ({"kind": None}, "'kind' must be a non-empty string"),
        ({"variants": None}, "'variants' must be a non-empty list"),
        (
            {"variants": [{"file": "x.wav"}]},
            "every variant must be a mapping with an id",
        ),
        ({"tags": "siren"}, "tags must be a list of strings"),
    ],
)
def test_malformed_manifest_raises(tmp_path, change, message):
    manifest = {key: value for key, value in {**SOUNDS["alarm_loop"], **change}.items() if value is not None}
    directory = _write_sound(tmp_path, "broken", manifest)
    with pytest.raises(ValueError, match=message):
        SoundAnnotation.from_manifest(directory)


def test_sound_build_fails_on_malformed_manifest(tmp_path):
    _write_sound(tmp_path / "input", "broken", {**SOUNDS["alarm_loop"], "version": 3})
    builder = DatabaseBuilder.Builder(AssetType.SOUND)(input_path=tmp_path / "input", output_path=tmp_path / "output")
    with pytest.raises(ValueError, match="broken.yaml: manifest version must be 2"):
        builder.build()
