import yaml

from arena_assets.semantic.fixes import fix_file
from arena_assets.semantic.taxonomy import load_spec

SPEC = load_spec()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_bare_domain_tag_becomes_domain_facet(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Widget" / "annotation.yaml",
        {"name": "Widget", "path": "Object/Widget", "tags": ["office"]},
    )
    assert fix_file(path, SPEC) is True
    assert _load(path)["tags"] == ["domain::office"]


def test_bare_pedestrian_tag_is_dropped(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": ["pedestrian"]},
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    assert "pedestrian" not in tags


def test_bare_makehuman_tag_becomes_origin_facet(tmp_path):
    path = _write(
        tmp_path / "misc" / "annotation.yaml",
        {"name": "Nurse", "path": "Human/Nurse", "tags": ["makehuman"]},
    )
    fix_file(path, SPEC)
    assert _load(path)["tags"] == ["origin::makehuman"]


def test_domain_added_from_storage_path_when_absent(tmp_path):
    path = _write(
        tmp_path / "Warehouse" / "Material" / "Crate" / "annotation.yaml",
        {"name": "Crate", "path": "Material/Crate", "tags": []},
    )
    fix_file(path, SPEC)
    assert "domain::warehouse" in _load(path)["tags"]


def test_domain_not_duplicated_when_already_present(tmp_path):
    path = _write(
        tmp_path / "Warehouse" / "Material" / "Crate" / "annotation.yaml",
        {"name": "Crate", "path": "Material/Crate", "tags": ["domain::office"]},
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    assert tags.count("domain::office") == 1
    assert "domain::warehouse" not in tags


def test_object_kind_inferred_from_name_hit(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Sofa_2" / "annotation.yaml",
        {"name": "Sofa_2", "path": "Object/Sofa_2", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::sofa" in _load(path)["tags"]


def test_object_kind_inference_picks_first_matching_name_token(tmp_path):
    # "Sofa" is the first token of the (camelCase-split) name, "Table" is second;
    # the first *matching* token wins, not table-declaration order.
    path = _write(
        tmp_path / "Office" / "Object" / "SofaTable" / "annotation.yaml",
        {"name": "SofaTable", "path": "Object/SofaTable", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::sofa" in _load(path)["tags"]
    assert "kind::table" not in _load(path)["tags"]


def test_object_kind_inference_miss_leaves_tags_absent(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "SM_Gizmo" / "annotation.yaml",
        {"name": "SM_Gizmo", "path": "Object/SM_Gizmo", "tags": []},
    )
    fix_file(path, SPEC)
    assert not any(tag.startswith("kind::") for tag in _load(path)["tags"])


def test_object_kind_override_beats_leading_token(tmp_path):
    # Lab_Bench is a worktable, the bench token would mislabel it
    path = _write(
        tmp_path / "Hospital" / "Object" / "Lab_Bench" / "annotation.yaml",
        {"name": "Lab_Bench", "path": "Object/Lab_Bench", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::counter" in _load(path)["tags"]


def test_object_kind_tokenizes_digit_boundaries(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "SM_Frame01" / "annotation.yaml",
        {"name": "SM_Frame01", "path": "Object/SM_Frame01", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::decor" in _load(path)["tags"]


def test_material_kind_inferred_from_name_hit(tmp_path):
    path = _write(
        tmp_path / "Common" / "Material" / "Oak_Plank" / "annotation.yaml",
        {"name": "Oak_Plank", "path": "Material/Oak_Plank", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::wood" in _load(path)["tags"]


def test_material_kind_inference_miss_leaves_tags_absent(tmp_path):
    path = _write(
        tmp_path / "Common" / "Material" / "Unobtainium_Rough" / "annotation.yaml",
        {"name": "Unobtainium_Rough", "path": "Material/Unobtainium_Rough", "tags": []},
    )
    fix_file(path, SPEC)
    assert not any(tag.startswith("kind::") for tag in _load(path)["tags"])


def test_material_kind_override_for_granite_trade_names(tmp_path):
    path = _write(
        tmp_path / "Common" / "Material" / "Volga_Blue" / "annotation.yaml",
        {"name": "Volga_Blue", "path": "Material/Volga_Blue", "tags": []},
    )
    fix_file(path, SPEC)
    assert "kind::stone" in _load(path)["tags"]


def test_color_values_are_normalized(tmp_path):
    path = _write(
        tmp_path / "Residential" / "Object" / "Bed_B" / "annotation.yaml",
        {
            "name": "Bed_B",
            "path": "Object/Bed_B",
            "tags": [
                "color::light yellow",
                "color::+yellow",
                "color::grey",
                "color::colored",
                "color::white",
            ],
        },
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    colors = [tag for tag in tags if tag.startswith("color::")]
    assert colors == ["color::yellow", "color::yellow", "color::gray", "color::white"]


def test_material_typos_are_normalized(tmp_path):
    path = _write(
        tmp_path / "Residential" / "Object" / "Vase" / "annotation.yaml",
        {
            "name": "Vase",
            "path": "Object/Vase",
            "tags": ["material::porceplain", "material::pottery"],
        },
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    assert "material::porcelain" in tags
    assert "material::ceramic" in tags


def test_existing_malformed_facet_values_are_renormalized(tmp_path):
    path = _write(
        tmp_path / "Residential" / "Object" / "Fans" / "annotation.yaml",
        {
            "name": "Fans",
            "path": "Object/Fans",
            "tags": ["color::dark grey", "color::tinted"],
        },
    )
    fix_file(path, SPEC)
    colors = [tag for tag in _load(path)["tags"] if tag.startswith("color::")]
    assert colors == ["color::gray"]


def test_bare_room_and_redundant_tags(tmp_path):
    path = _write(
        tmp_path / "Residential" / "Object" / "Oven" / "annotation.yaml",
        {
            "name": "Oven",
            "path": "Object/Oven",
            "tags": ["kitchen", "livingroom", "furniture", "appliance"],
        },
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    assert "room::kitchen" in tags
    assert "room::living_room" in tags
    assert "kind::appliance" in tags
    assert "furniture" not in tags
    assert "appliance" not in tags


def test_existing_kind_tag_is_not_overridden(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Sofa_2" / "annotation.yaml",
        {"name": "Sofa_2", "path": "Object/Sofa_2", "tags": ["kind::chair"]},
    )
    fix_file(path, SPEC)
    tags = _load(path)["tags"]
    assert tags.count("kind::chair") == 1
    assert "kind::sofa" not in tags


def test_tags_reordered_by_facet_declaration_order(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {
            "name": "Chair",
            "path": "Object/Chair",
            "tags": ["material::wood", "color::brown", "domain::office", "kind::chair"],
        },
    )
    fix_file(path, SPEC)
    assert _load(path)["tags"] == [
        "domain::office",
        "kind::chair",
        "color::brown",
        "material::wood",
    ]


def test_unknown_tags_kept_at_the_end_in_original_order(tmp_path):
    path = _write(
        tmp_path / "misc" / "annotation.yaml",
        {
            "name": "Widget",
            "path": "Object/Widget",
            "tags": ["zeta", "kind::chair", "alpha"],
        },
    )
    fix_file(path, SPEC)
    assert _load(path)["tags"] == ["kind::chair", "zeta", "alpha"]


def test_stamps_asa_version(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {"name": "Chair", "path": "Object/Chair", "tags": []},
    )
    fix_file(path, SPEC)
    assert _load(path)["asa"] == 1


def test_other_keys_preserved_untouched(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {
            "name": "Chair",
            "path": "Object/Chair",
            "tags": ["hoi::sit"],
            "note": "wheels included",
            "face": "+x",
            "bounding_box": [[-0.5, 0.5], [-0.4, 0.4], [0.0, 1.2]],
        },
    )
    fix_file(path, SPEC)
    data = _load(path)
    assert "hoi::sit" in data["tags"]
    assert data["note"] == "wheels included"
    assert data["face"] == "+x"
    assert data["bounding_box"] == [[-0.5, 0.5], [-0.4, 0.4], [0.0, 1.2]]


def test_fix_returns_false_and_is_idempotent(tmp_path):
    path = _write(
        tmp_path / "Office" / "Object" / "Chair" / "annotation.yaml",
        {
            "name": "Chair",
            "path": "Object/Chair",
            "tags": ["office"],
            "color": ["black"],
        },
    )
    assert fix_file(path, SPEC) is True
    after_first = _load(path)

    assert fix_file(path, SPEC) is False
    assert _load(path) == after_first


def test_storage_class_domain_override_used_when_path_does_not_match_convention(
    tmp_path,
):
    path = _write(
        tmp_path / "orphan" / "annotation.yaml",
        {"name": "Oak_Plank", "path": "Material/Oak_Plank", "tags": []},
    )
    fix_file(path, SPEC, storage_class_domain=("hospital", "material"))
    data = _load(path)
    assert "domain::hospital" in data["tags"]
    assert "kind::wood" in data["tags"]


def test_no_domain_added_when_storage_domain_unknown(tmp_path):
    path = _write(
        tmp_path / "orphan" / "annotation.yaml",
        {"name": "Widget", "path": "Object/Widget", "tags": []},
    )
    fix_file(path, SPEC)
    assert not any(tag.startswith("domain::") for tag in _load(path)["tags"])
