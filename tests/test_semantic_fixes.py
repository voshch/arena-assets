import pytest
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


def _fixed_tags(root, directory, asset, tags, **fields):
    path = _write(root / directory / "annotation.yaml", {"name": asset.rpartition("/")[2], "path": asset, "tags": tags, **fields})
    fix_file(path, SPEC)
    return _load(path)["tags"]


@pytest.mark.parametrize(
    ("directory", "asset", "tags", "present", "absent"),
    [
        ("Office/Object/Chair", "Object/Chair", ["pedestrian"], [], ["pedestrian"]),
        ("Warehouse/Material/Crate", "Material/Crate", [], ["domain::warehouse"], []),
        ("Warehouse/Material/Crate", "Material/Crate", ["domain::office"], ["domain::office"], ["domain::warehouse"]),
        ("Residential/Object/Vase", "Object/Vase", ["material::porceplain", "material::pottery"], ["material::porcelain", "material::ceramic"], []),
        ("Residential/Object/Oven", "Object/Oven", ["kitchen", "livingroom", "furniture", "appliance"], ["room::kitchen", "room::living_room", "kind::appliance"], ["furniture", "appliance"]),
        ("Office/Object/Sofa_2", "Object/Sofa_2", ["kind::chair"], ["kind::chair"], ["kind::sofa"]),
    ],
)
def test_fix_rewrites_tags(tmp_path, directory, asset, tags, present, absent):
    fixed = _fixed_tags(tmp_path, directory, asset, tags)
    assert [fixed.count(tag) for tag in present] == [1] * len(present)
    assert not set(fixed) & set(absent)


@pytest.mark.parametrize(
    ("directory", "asset", "tags", "expected"),
    [
        ("Office/Object/Widget", "Object/Widget", ["office"], ["domain::office"]),
        ("misc", "Human/Nurse", ["makehuman"], ["origin::makehuman"]),
        ("Common/Sound/chair", "Sound/chair", [], ["domain::common"]),
    ],
)
def test_fix_gives_exactly_these_tags(tmp_path, directory, asset, tags, expected):
    assert _fixed_tags(tmp_path, directory, asset, tags) == expected


@pytest.mark.parametrize(
    ("domain", "asset_class", "name", "kind"),
    [
        ("Office", "Object", "Sofa_2", "sofa"),
        ("Office", "Object", "SofaTable", "sofa"),
        ("Office", "Object", "SM_Gizmo", None),
        ("Hospital", "Object", "Lab_Bench", "counter"),
        ("Office", "Object", "SM_Frame01", "decor"),
        ("Common", "Material", "Oak_Plank", "wood"),
        ("Common", "Material", "Unobtainium_Rough", None),
        ("Common", "Material", "Volga_Blue", "stone"),
    ],
)
def test_kind_is_inferred_from_the_name(tmp_path, domain, asset_class, name, kind):
    fixed = _fixed_tags(tmp_path, f"{domain}/{asset_class}/{name}", f"{asset_class}/{name}", [])
    assert [tag for tag in fixed if tag.startswith("kind::")] == ([f"kind::{kind}"] if kind else [])


def test_sound_kind_comes_from_the_kind_field(tmp_path):
    path = _write(
        tmp_path / "Common" / "Sound" / "door_knock" / "annotation.yaml",
        {"name": "door_knock", "path": "Sound/door_knock", "kind": "onset", "tags": ["knock", "onset"]},
    )
    assert fix_file(path, SPEC) is True
    assert _load(path)["tags"] == ["domain::common", "kind::onset", "knock"]
    assert _load(path)["kind"] == "onset"


def test_sound_bare_tags_stay_bare(tmp_path):
    path = _write(
        tmp_path / "Common" / "Sound" / "greeting" / "annotation.yaml",
        {"name": "greeting", "path": "Sound/greeting", "kind": "speech", "tags": ["pedestrian", "kitchen", "office", "appliance", "kind::speech"]},
    )
    fix_file(path, SPEC)
    assert _load(path)["tags"] == ["domain::common", "kind::speech", "pedestrian", "kitchen", "office", "appliance"]


@pytest.mark.parametrize(
    ("tags", "colors"),
    [
        (["color::light yellow", "color::+yellow", "color::grey", "color::colored", "color::white"], ["color::yellow", "color::yellow", "color::gray", "color::white"]),
        (["color::dark grey", "color::tinted"], ["color::gray"]),
    ],
)
def test_color_values_are_normalized(tmp_path, tags, colors):
    fixed = _fixed_tags(tmp_path, "Residential/Object/Bed_B", "Object/Bed_B", tags)
    assert [tag for tag in fixed if tag.startswith("color::")] == colors


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
