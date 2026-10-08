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


def test_malformed_tag_is_an_error(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["Domain::Office"]},
    )
    findings = lint_file(path, SPEC)
    malformed = [f for f in findings if f.code == "malformed-tag"]
    assert len(malformed) == 1
    assert malformed[0].severity == "error"


def test_unknown_facet_is_a_warning(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["nonsense::value"]},
    )
    findings = lint_file(path, SPEC)
    matches = [f for f in findings if f.code == "unknown-facet"]
    assert len(matches) == 1
    assert matches[0].severity == "warning"


def test_cardinality_violation_is_a_warning(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {
            "name": "Chair",
            "path": "Object/Chair",
            "tags": ["kind::chair", "kind::sofa"],
        },
    )
    findings = lint_file(path, SPEC)
    matches = [f for f in findings if f.code == "cardinality-violation"]
    assert len(matches) == 1
    assert matches[0].severity == "warning"


def test_closed_vocab_violation_is_a_warning(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["mount::orbit"]},
    )
    findings = lint_file(path, SPEC)
    matches = [f for f in findings if f.code == "closed-vocab-violation"]
    assert len(matches) == 1
    assert matches[0].severity == "warning"


def test_facet_not_applicable_is_a_warning(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["human::gender::male"]},
    )
    findings = lint_file(path, SPEC)
    matches = [f for f in findings if f.code == "facet-not-applicable"]
    assert len(matches) == 1
    assert matches[0].severity == "warning"


def test_novel_value_on_open_facet_is_info_with_suggestion(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["kind::chairs"]},
    )
    findings = lint_file(path, SPEC)
    matches = [f for f in findings if f.code == "novel-value"]
    assert len(matches) == 1
    assert matches[0].severity == "info"
    assert "chair" in matches[0].message


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


def test_bare_tag_is_info(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["office"]},
    )
    findings = lint_file(path, SPEC)
    bare = [f for f in findings if f.code == "bare-tag"]
    assert len(bare) == 1
    assert bare[0].severity == "info"


def test_asset_class_inferred_from_path_field(tmp_path):
    # directory nesting doesn't reflect Domain/Class/name, only the "path" field does
    path = _write(
        tmp_path / "flat" / "annotation.yaml",
        {"name": "Steel", "path": "Material/Steel", "tags": ["human::gender::male"]},
    )
    findings = lint_file(path, SPEC)
    assert any(f.code == "facet-not-applicable" for f in findings)


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
