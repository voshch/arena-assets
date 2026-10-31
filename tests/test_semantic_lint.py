import pytest
import yaml

from arena_assets.semantic.lint import lint_file, lint_tree
from arena_assets.semantic.taxonomy import load_spec

SPEC = load_spec()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _codes(findings):
    return [finding.code for finding in findings]


@pytest.mark.parametrize(
    ("tags", "code", "severity", "message"),
    [
        (["Domain::Office"], "malformed-tag", "error", ""),
        (["nonsense::value"], "unknown-facet", "warning", ""),
        (["kind::chair", "kind::sofa"], "cardinality-violation", "warning", ""),
        (["mount::orbit"], "closed-vocab-violation", "warning", ""),
        (["human::gender::male"], "facet-not-applicable", "warning", ""),
        (["kind::chairs"], "novel-value", "info", "chair"),
        (["office"], "bare-tag", "info", ""),
    ],
)
def test_lint_finds_one_issue_of_the_tag(tmp_path, tags, code, severity, message):
    path = _write(tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml", {"name": "Chair", "path": "Object/Chair", "tags": tags})
    matches = [f for f in lint_file(path, SPEC) if f.code == code]
    assert len(matches) == 1
    assert matches[0].severity == severity
    assert message in matches[0].message


def test_missing_recommended_facet_is_info(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {
            "name": "Chair",
            "path": "Object/Chair",
            "tags": ["kind::chair", "domain::office"],
        },
    )
    findings = lint_file(path, SPEC)
    # domain and kind are both present, so neither should be reported missing
    assert "missing-recommended-facet" not in _codes(findings)

    path2 = _write(
        tmp_path / "Office" / "Object" / "Stool" / "annotation.yaml",
        {"name": "Stool", "path": "Object/Stool", "tags": []},
    )
    findings2 = lint_file(path2, SPEC)
    missing = {f.message for f in findings2 if f.code == "missing-recommended-facet"}
    assert any("domain" in m for m in missing)
    assert any("kind" in m for m in missing)
    assert all(f.severity == "info" for f in findings2 if f.code == "missing-recommended-facet")


def test_missing_asa_stamp_is_info(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": []},
    )
    findings = lint_file(path, SPEC)
    assert "missing-asa-stamp" in _codes(findings)

    path2 = _write(
        tmp_path / "Office" / "Object" / "Stamped" / "annotation.yaml",
        {"name": "Stamped", "path": "Object/Stamped", "tags": [], "asa": 1},
    )
    findings2 = lint_file(path2, SPEC)
    assert "missing-asa-stamp" not in _codes(findings2)


def test_asset_class_inferred_from_path_field(tmp_path):
    # directory nesting doesn't reflect Domain/Class/name, only the "path" field does
    path = _write(
        tmp_path / "flat" / "annotation.yaml",
        {"name": "Steel", "path": "Material/Steel", "tags": ["human::gender::male"]},
    )
    findings = lint_file(path, SPEC)
    assert any(f.code == "facet-not-applicable" for f in findings)


def test_sound_class_is_inferred_and_shares_kind_and_origin(tmp_path):
    path = _write(
        tmp_path / "Common" / "Sound" / "door_knock" / "annotation.yaml",
        {"name": "door_knock", "path": "Sound/door_knock", "asa": 1, "tags": ["domain::common", "kind::onset", "origin::fsd50k"]},
    )
    findings = lint_file(path, SPEC)
    assert [(f.severity, f.code) for f in findings] == [("info", "novel-value")]
    assert "'fsd50k'" in findings[0].message


def test_object_facet_on_a_sound_is_a_warning(tmp_path):
    path = _write(
        tmp_path / "Common" / "Sound" / "door_knock" / "annotation.yaml",
        {"name": "door_knock", "path": "Sound/door_knock", "asa": 1, "tags": ["domain::common", "kind::onset", "mount::wall"]},
    )
    matches = [f for f in lint_file(path, SPEC) if f.code == "facet-not-applicable"]
    assert len(matches) == 1
    assert matches[0].severity == "warning"
    assert "sound" in matches[0].message


def test_novel_sound_kind_is_info(tmp_path):
    path = _write(
        tmp_path / "Common" / "Sound" / "rain" / "annotation.yaml",
        {"name": "rain", "path": "Sound/rain", "asa": 1, "tags": ["domain::common", "kind::weather"]},
    )
    assert [(f.severity, f.code) for f in lint_file(path, SPEC)] == [("info", "novel-value")]


def test_lint_tree_walks_all_annotation_files(tmp_path):
    _write(
        tmp_path / "Office" / "Object" / "A" / "annotation.yaml",
        {"name": "A", "path": "Object/A", "tags": []},
    )
    _write(
        tmp_path / "Office" / "Object" / "B" / "annotation.yaml",
        {"name": "B", "path": "Object/B", "tags": []},
    )
    findings = lint_tree(tmp_path, SPEC)
    files = {finding.file for finding in findings}
    assert len(files) == 2
