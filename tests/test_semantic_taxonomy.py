from arena_assets.semantic import metadata_columns, parse_tag, parse_tags, words
from arena_assets.semantic.taxonomy import class_from_dirname, is_grammatical, load_spec


def test_parse_tag_splits_key_and_value():
    assert parse_tag("domain::office") == ("domain", "office")


def test_parse_tag_nested_facet_key():
    assert parse_tag("human::gender::male") == ("human::gender", "male")


def test_parse_tag_none_without_separator():
    assert parse_tag("bareword") is None


def test_parse_tags_groups_and_preserves_order():
    grouped = parse_tags(["domain::office", "bare", "domain::hospital", "kind::chair"])
    assert grouped == {"domain": ["office", "hospital"], "kind": ["chair"]}


def test_is_grammatical_rejects_uppercase_and_empty_segments():
    assert is_grammatical("domain::office")
    assert not is_grammatical("Domain::Office")
    assert not is_grammatical("domain::")
    assert not is_grammatical("::office")
    assert not is_grammatical("bareword")


def test_is_grammatical_allows_nested_facet_keys():
    assert is_grammatical("human::gender::male")


def test_class_from_dirname():
    assert class_from_dirname("Object") == "object"
    assert class_from_dirname("Material") == "material"
    assert class_from_dirname("Human") == "human"
    assert class_from_dirname("Nonsense") is None


def test_load_spec_default_has_expected_facets():
    spec = load_spec()
    assert spec.version == 1
    assert "domain" in spec.facets
    assert "human::gender" in spec.facets
    domain = spec.facets["domain"]
    assert domain.applies == ["object", "material", "human"]
    assert domain.cardinality == "multi"
    assert domain.open is True
    assert domain.embed is True
    kind = spec.facets["kind"]
    assert kind.values_for("object") == [
        "appliance",
        "bag",
        "barrier",
        "bed",
        "bench",
        "bin",
        "box",
        "cabinet",
        "cart",
        "chair",
        "container",
        "cookware",
        "counter",
        "curtain",
        "decor",
        "desk",
        "dispenser",
        "equipment",
        "food",
        "lamp",
        "mirror",
        "plant",
        "safety",
        "sanitary",
        "shelf",
        "sign",
        "sofa",
        "stool",
        "structure",
        "table",
        "tableware",
    ]
    assert kind.values_for("material") == [
        "asphalt",
        "brick",
        "ceramic",
        "composite",
        "concrete",
        "fabric",
        "gem",
        "glass",
        "gravel",
        "ground",
        "leather",
        "light",
        "metal",
        "paint",
        "paper",
        "plaster",
        "plastic",
        "rubber",
        "stone",
        "tile",
        "water",
        "wood",
    ]


def test_words_drops_embed_false_facets_keeps_unknown_and_bare():
    tags = ["domain::office", "mount::wall", "unknownfacet::value", "bareword"]
    # domain: embed true -> "office"; mount: embed false -> dropped;
    # unknown facet -> keep value verbatim; bare tag -> keep verbatim
    assert words(tags) == "office value bareword"


def test_words_dedups_preserving_order():
    assert words(["domain::office", "domain::office", "bareword", "bareword"]) == "office bareword"


def test_words_empty_list():
    assert words([]) == ""


def test_metadata_columns_uses_dotted_facet_keys():
    columns = metadata_columns(["human::gender::male", "human::age::young"])
    assert columns == {"human.gender": "male", "human.age": "young"}


def test_metadata_columns_comma_joins_multi_valued_facets():
    columns = metadata_columns(["domain::office", "domain::hospital"])
    assert columns == {"domain": "office,hospital"}


def test_metadata_columns_skips_bare_tags():
    assert metadata_columns(["bareword"]) == {}


def test_metadata_columns_skips_reserved_names():
    # "note" is both a reserved annotation field and a plausible-looking facet key
    assert metadata_columns(["note::something"]) == {}
