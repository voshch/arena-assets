# ASA: Arena Semantic Annotations

ASA is the semantic labeling standard for `annotation.yaml` files (objects,
materials, humans). It gives every tag a stable shape so it can be used for
embedding text, filterable metadata, and lint, without every consumer inventing
its own parsing.

ASA covers *semantics only*: what an asset is, what it affords, where it belongs.
First-order geometric properties (`bounding_box`, `face`, and the extents derived
from them) are measured or computed facts about the model file, not annotations;
they live as typed fields next to the tags and are outside this standard.

## Grammar

A tag is `key::path::value`: lowercase `[a-z0-9_]+` segments joined by `::`, at
least two segments. Splitting on `::`, the last segment is the value, everything
before it is the facet key (a facet key can itself contain `::`, e.g.
`human::gender::male` is facet `human::gender`, value `male`). Tags without `::`
are bare tags: permitted but unstructured.

Cardinality (`single`/`multi`) and vocabulary (`open`/closed `values`) are lint
dimensions, never hard gates: nothing in the toolchain refuses to load a tag that
violates them.

## Facets (asa: 1)

| facet | applies to | cardinality | recommended | open | embedded |
| --- | --- | --- | --- | --- | --- |
| `domain` | object, material, human | multi | yes | yes | yes |
| `kind` | object, material | single | yes | yes | yes |
| `mount` | object | multi | | | |
| `room` | object | multi | | yes | yes |
| `hoi` | object | multi | | yes | yes |
| `color` | object, material | multi | | yes | yes |
| `material` | object | multi | | yes | yes |
| `human::gender` | human | single | yes | | yes |
| `human::ethnicity` | human | single | yes | yes | yes |
| `human::age` | human | single | yes | | yes |
| `human::role` | human | single | yes | yes | yes |
| `origin` | object, material, human | single | | yes | |

`hoi` is order-sensitive: the first `hoi::` tag is the primary interaction
property in the ProcTHOR export, the rest are secondary.

The full seed vocabularies live in [`asa.yaml`](asa.yaml), which is the source of
truth loaded at runtime; this table is a summary.

## Projections (`arena_assets.semantic`)

- `words(tags)`: text for embedding. Tags on an `embed: true` facet (or an unknown
  facet, or a bare tag) contribute their value; `embed: false` facets are dropped.
  Order-preserving, deduplicated, space-joined.
- `metadata_columns(tags)`: one Chroma metadata column per facet present
  (`human::gender` -> `human.gender`), comma-joined values, skipping any name that
  collides with a non-tag annotation field (`name`, `path`, `desc`, `tags`, `note`,
  `face`, `bounding_box`, `width`, `depth`, `height`, `volume`, `asa`).
- `parse_tag`/`parse_tags`: the raw `key::path::value` split, grouped by facet key.

## Chapters

Revision 1 consists of the labels chapter (facets above). `machines:` is reserved
for a future chapter describing semantic state machines (doors, elevators,
human locomotion): it is empty, not normative, and not part of revision 1
conformance.

## Versioning

`asa: 1` in `asa.yaml` is the spec revision; every annotation stamps the revision
it was authored against.

Within a revision, facets may be added and open vocabularies may grow. Consumers
must tolerate facets and values they do not know: unknown facets still parse, and
recurring novel values graduate into the seed vocabulary without a revision bump.

A revision bump is reserved for changes that alter the meaning or parsing of
existing tags: grammar changes, facet renames or removals, cardinality tightening,
or changes to a closed vocabulary. `--fix` restamps every file it touches to the
current revision; an older stamp only means the file has not been revisited since.

## Conformance

An annotation conforms to ASA when lint reports no errors and no warnings for it;
info findings are advisory and do not affect conformance. `asa.yaml` is the
normative artifact and `arena-assets semantic lint` the reference checker; this
document is explanatory.

## Lint and fix

`arena-assets semantic lint PATH [--fix] [--json]` walks `PATH` for
`annotation.yaml` files (or lints a single file) and reports findings:

- **error**: malformed tag grammar (bad segments, wrong case, empty segment).
- **warning**: unknown facet key, cardinality violation, closed-vocabulary
  violation, a facet used outside its `applies` list.
- **info**: missing recommended facet, novel value on an open facet (with a
  nearest-match suggestion), missing `asa` version stamp, bare tags.

Exit code is 1 if any error or warning remains after linting (including after
`--fix`), 0 otherwise; info-only runs always exit 0.

`--fix` applies mechanical, order-preserving rewrites: bare tags with a facet
equivalent become facet tags, `color::`/`material::` values are normalized to the
grammar (modifier words dropped, spelling unified), a missing `domain` facet is
filled in from the asset's `<Domain>/<Class>/<name>/annotation.yaml` storage path,
a missing `kind` facet is inferred from the asset name against a per-class word
table (first matching name token wins, no match leaves it absent for lint to
flag), tags are reordered by facet declaration order, and the file is stamped
`asa: 1`. Fixing an already-fixed file is a no-op.
