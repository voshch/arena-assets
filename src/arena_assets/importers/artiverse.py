"""Artiverse articulated objects from the gated Hugging Face release chunks, each part-segmented glTF staged with its part nodes named by link."""

from __future__ import annotations

import collections
import json
import re
import struct
import tarfile
import typing
import urllib.error
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.articulated import ASSET_TYPE, DOMAIN, Y_UP, Body, Edge, assemble, blender_name, build, refused, slug, write_articulation
from arena_assets.importers.gso import annotate
from arena_assets.importers.providers import HFHub, Provider
from arena_assets.utils.logging import get_logger

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.articulation import Articulation

REPO = "3dlg-hcvc/artiverse"
REVISION = "8599042e8bd0ce3e9f8ccbbc5a67866c4efb683f"
RELEASES = "releases.json"
MANIFEST = "manifest.json"
CHUNKS = "chunks.json"
SOURCE = "Artiverse"
GLB = ".segmented.glb"
JOINTS = ".articulations.json"
ORIGIN = "origin::artiverse"
LICENSE = "per object the terms of its source dataset (ShapeNet, ABO, 3D-FUTURE, HSSD, Objaverse and others), non-commercial research use"
AUTHORS = "Denys Iliash, Jiayi Liu, Egor Fokin, Qirui Wu, Ali Mahdavi-Amiri, Manolis Savva, Angel X. Chang (Simon Fraser University), objects from their source datasets"
CITATION = "Iliash et al., Artiverse: A Diverse and Physically Grounded Dataset for Articulated Objects, CVPR 2026"
JSON_CHUNK = 0x4E4F534A
BINARY_CHUNK = 0x004E4942
ROOT = "base"
TYPES = {
    "revolute": "revolute",
    "rotation": "revolute",
    "continuous": "continuous",
    "prismatic": "prismatic",
    "translation": "prismatic",
    "fixed": "fixed",
    "free": "fixed",
    "cylindrical": "cylindrical",
    "screw": "cylindrical",
    "universal": "universal",
}
TURNS = ("revolute", "continuous", "cylindrical", "universal")
SLIDES = ("prismatic", "cylindrical")

_ROOT = re.compile(r"data/([^/]+)/[^/]+/([^/]+)")
logger = get_logger("import.artiverse")

Entry = Mapping[str, typing.Any]


def read_glb(path: Path) -> tuple[dict, bytes]:
    """JSON document and binary buffer of a glTF binary."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path.name} is no glTF 2 binary")
    document: dict | None = None
    binary = b""
    offset = 12
    while offset < length:
        size, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + size]
        if kind == JSON_CHUNK:
            document = json.loads(chunk)
        elif kind == BINARY_CHUNK:
            binary = chunk
        offset += 8 + size
    if document is None:
        raise ValueError(f"{path.name} has no JSON chunk")
    return document, binary


def write_glb(path: Path, document: dict, binary: bytes) -> None:
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(text), JSON_CHUNK) + text + (struct.pack("<II", len(binary), BINARY_CHUNK) + binary if binary else b"")
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def parts(document: Mapping[str, typing.Any]) -> dict[int, tuple[int, str]]:
    """Node index and label of every part: the children of the scene's first root node that carry an `id` extra."""
    scene = document["scenes"][document.get("scene", 0)]
    root = document["nodes"][scene["nodes"][0]]
    found: dict[int, tuple[int, str]] = {}
    for child in root.get("children", []):
        extras = document["nodes"][child].get("extras") or {}
        if "id" in extras:
            found[int(extras["id"])] = (child, str(extras.get("label", "")))
    if not found:
        raise ValueError("the glTF root node has no part children with an id extra")
    return found


def link_names(found: Mapping[int, tuple[int, str]]) -> dict[int, str]:
    """Link of every part, its label in snake case with its part id."""
    return {pid: blender_name(f"{slug(label) if label else 'part'}_{pid}") for pid, (_, label) in found.items()}


def rename(document: dict, names: Mapping[int, str]) -> None:
    """Name every part node after its link and every other node apart from them, so object names identify the parts."""
    found = parts(document)
    for index, node in enumerate(document["nodes"]):
        node["name"] = f"node_{index}"
    for pid, (index, _) in found.items():
        document["nodes"][index]["name"] = names[pid]


def _articulations(joints: Mapping[str, typing.Any]) -> list[Entry]:
    if "articulations" in joints:
        return list(joints["articulations"])
    return list((joints.get("annotation") or {}).get("articulations", []))


def _model(vector: Iterable[float]) -> np.ndarray:
    return Y_UP[:3, :3] @ np.array([float(component) for component in vector])


def _range(entry: Entry, low: Sequence[str], high: Sequence[str]) -> tuple[float, float] | None:
    found = [next((float(entry[key]) for key in keys if entry.get(key) is not None), None) for keys in (low, high)]
    if found[0] is None or found[1] is None:
        return None
    return (min(found[0], found[1]), max(found[0], found[1]))


def _turn(name: str, parent: str, child: str, axis: Iterable[float], origin: Iterable[float], limit: tuple[float, float] | None, continuous: bool = False) -> Edge:
    direction = _model(axis)
    return Edge(
        name=name,
        type="continuous" if continuous or limit is None else "revolute",
        parent=parent,
        child=child,
        origin=typing.cast("tuple[float, float, float]", tuple(float(component) for component in _model(origin))),
        axis=typing.cast("tuple[float, float, float]", tuple(float(component) for component in direction / np.linalg.norm(direction))),
        limit=None if continuous else limit,
        label=name,
    )


def _slide(name: str, parent: str, child: str, entry: Entry, keys: tuple[Sequence[str], Sequence[str]]) -> Edge:
    limit = _range(entry, *keys)
    if limit is None:
        raise ValueError(f"prismatic joint {name} has no range, an articulation needs one")
    direction = _model(entry.get("axis", entry.get("axis1", (0.0, 0.0, 1.0))))
    return Edge(name=name, type="prismatic", parent=parent, child=child, axis=typing.cast("tuple[float, float, float]", tuple(float(component) for component in direction / np.linalg.norm(direction))), limit=limit, label=name)


def part_edges(link: str, parent: str, child: str, entries: Sequence[Entry], bodies: dict[str, Body]) -> list[Edge]:
    """Joints between a part and its parent: one per degree of freedom, two degrees chained over an intermediate link without nodes."""
    kinds = []
    for entry in entries:
        kind = str(entry.get("type", "fixed")).lower()
        if kind not in TYPES:
            raise ValueError(f"part {link} has joint type {kind!r}, known: {', '.join(TYPES)}")
        kinds.append(TYPES[kind])
    turns = [entry for entry, kind in zip(entries, kinds, strict=True) if kind in TURNS]
    slides = [entry for entry, kind in zip(entries, kinds, strict=True) if kind in SLIDES]
    rotation_range = (("rangeMin", "rotRangeMin", "rotation_rangeMin"), ("rangeMax", "rotRangeMax", "rotation_rangeMax"))
    if "universal" in kinds:
        entry = entries[kinds.index("universal")]
        middle = f"{child}:axis1"
        bodies[middle] = Body(link=f"{link}_axis1")
        origin = entry.get("origin", entry.get("origin1", (0.0, 0.0, 0.0)))
        return [
            _turn(f"{link}_axis1", parent, middle, entry.get("axis", entry.get("axis1", (0.0, 0.0, 1.0))), origin, _range(entry, ("rangeMin", "rot1RangeMin"), ("rangeMax", "rot1RangeMax"))),
            _turn(f"{link}_axis2", middle, child, entry.get("axis2", entry.get("ref", (1.0, 0.0, 0.0))), entry.get("origin2", origin), _range(entry, ("rot2RangeMin", "range2Min", "rotation2_rangeMin"), ("rot2RangeMax", "range2Max", "rotation2_rangeMax"))),
        ]
    if "cylindrical" in kinds or (turns and slides and turns[0] is not slides[0]):
        turn, slide = turns[0], slides[-1]
        middle = f"{child}:slide"
        bodies[middle] = Body(link=f"{link}_slide")
        single = turn is slide
        keys = (("prismatic_rangeMin",), ("prismatic_rangeMax",)) if single else (("prismatic_rangeMin", "rangeMin"), ("prismatic_rangeMax", "rangeMax"))
        return [
            _slide(f"{link}_slide", parent, middle, slide, keys),
            _turn(f"{link}_turn", middle, child, turn.get("axis", (0.0, 0.0, 1.0)), turn.get("origin", (0.0, 0.0, 0.0)), _range(turn, *rotation_range)),
        ]
    if turns:
        entry = turns[0]
        return [_turn(link, parent, child, entry.get("axis", (0.0, 0.0, 1.0)), entry.get("origin", (0.0, 0.0, 0.0)), _range(entry, *rotation_range), continuous=TYPES[str(entry.get("type")).lower()] == "continuous")]
    if slides:
        return [_slide(link, parent, child, slides[0], (("prismatic_rangeMin", "rangeMin"), ("prismatic_rangeMax", "rangeMax")))]
    return [Edge(name=link, type="fixed", parent=parent, child=child)]


def motion_states(entries: Iterable[Entry], joint_of: Mapping[int, str], articulation: Articulation) -> dict[str, dict[str, float]]:
    """Named per-joint values the annotation gives single-joint parts in `motionStates`, the values inside each joint's limit."""
    joints = articulation.joint_map
    states: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for entry in entries:
        joint = joints.get(joint_of.get(int(entry.get("pid", -1)), ""))
        if joint is None or not joint.movable:
            continue
        for state, motion in (entry.get("motionStates") or {}).items():
            value = (motion or {}).get("value")
            if value is None or (joint.limit is not None and not joint.limit[0] - 1e-6 <= float(value) <= joint.limit[1] + 1e-6):
                continue
            states[slug(state)][joint.name] = float(value)
    return dict(states)


def artiverse_articulation(document: Mapping[str, typing.Any], joints: Mapping[str, typing.Any]) -> Articulation | None:
    """The articulation of a segmented glTF and its articulations file in the Z-up model frame, None when no part moves."""
    from arena_simulation_setup.utils.articulation import RESERVED_STATES, Articulation

    found = parts(document)
    names = link_names(found)
    graph = (document.get("extras") or {}).get("flattened_graph") or {}
    entries: dict[int, list[Entry]] = collections.defaultdict(list)
    listed = _articulations(joints)
    for entry in listed:
        entries[int(entry["pid"])].append(entry)

    def parent_of(pid: int) -> int:
        node = graph.get(str(pid)) or {}
        if node.get("parent") is not None:
            return int(node["parent"])
        bases = [base for entry in entries[pid] for base in entry.get("base") or ()]
        return int(bases[0]) if bases else -1

    def moves(pid: int) -> bool:
        return any(TYPES.get(str(entry.get("type", "fixed")).lower(), "") != "fixed" for entry in entries[pid])

    top = [pid for pid in found if parent_of(pid) not in found]
    bodies = {str(pid): Body(link=names[pid], nodes=(names[pid],), label=label) for pid, (_, label) in found.items()}
    if len(top) == 1 and not moves(top[0]):
        root = str(top[0])
    else:
        root = ROOT
        bodies[root] = Body(link=ROOT)
    edges: list[Edge] = []
    single: dict[int, str] = {}
    for pid in found:
        if str(pid) == root:
            continue
        made = part_edges(names[pid], str(parent_of(pid)) if parent_of(pid) in found else root, str(pid), entries[pid], bodies)
        if len(made) == 1:
            single[pid] = made[0].name
        edges.extend(made)
    articulation = assemble(bodies, edges, root)
    if articulation is None:
        return None
    states = {state: dict(values) for state, values in articulation.states.items()}
    detents: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for name, values in motion_states(listed, single, articulation).items():
        if name in ("closed", "open"):
            states.setdefault(name, {}).update(values)
        elif name not in RESERVED_STATES:
            for joint, value in values.items():
                detents[joint][name] = value
    return Articulation(links=articulation.links, joints=articulation.joints, states=states, detents=dict(detents))


def release_chunks(releases: Mapping[str, typing.Any], manifests: Mapping[str, Mapping[str, typing.Any]]) -> list[tuple[str, list[str]]]:
    """Every chunk archive of the releases in the order they apply, with the model folders it holds."""
    return [(f"{version['chunks_dir']}/{chunk['archive']}", list(chunk["roots"])) for version in releases["versions"] for chunk in manifests[version["chunks_dir"]]["chunks"]]


def extract(stream: typing.IO[bytes], root: str, dest: Path) -> list[str]:
    """Write the segmented glTF and articulations of one model folder from a tar.gz stream into dest, reading no further than that folder."""
    item = root.rsplit("/", 1)[-1]
    wanted = {f"{root}/{item}{suffix}": suffix for suffix in (GLB, JOINTS)}
    written: list[str] = []
    inside = False
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if member.name == root or member.name.startswith(f"{root}/"):
                inside = True
            elif inside:
                break
            suffix = wanted.get(member.name)
            if suffix is None or not member.isfile():
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            part = dest / f"{item}{suffix}.part"
            part.write_bytes(data.read())
            part.replace(dest / f"{item}{suffix}")
            written.append(suffix)
    return written


def object_name(item: str) -> str:
    return f"artiverse_{slug(item.rsplit('/', 1)[-1])}"


def stage(item: str, src: Path, dataset: Path) -> Path:
    """Lay one object out as builder input under dataset: its glTF with part nodes named by link, its articulation, its category as kind."""
    name = object_name(item)
    category = item.rsplit("/", 1)[0] if "/" in item else ""
    source = dataset / DOMAIN / ASSET_TYPE / name
    source.mkdir(parents=True)
    document, binary = read_glb(src)
    joints = json.loads(src.with_name(src.name.removesuffix(GLB) + JOINTS).read_text(encoding="utf-8-sig"))
    articulation = artiverse_articulation(document, joints)
    rename(document, link_names(parts(document)))
    write_glb(source / f"{name}.glb", document, binary)
    labels = sorted({label for _, label in parts(document).values() if label})
    what = category.replace("_", " ") if category else f"{SOURCE} object"
    kind = slug(category) if category else None
    annotate(source / ANNOTATION_NAME, name, f"{what} with {', '.join(labels)}" if labels else what, [ORIGIN, *([kind] if kind else [])], kind)
    write_articulation(source / ANNOTATION_NAME, articulation)
    return source


class Artiverse(Importer):
    meta = IMPORTERS.meta("object", "artiverse")

    def __init__(self, cache: Path | None = None, provider: Provider | None = None):
        super().__init__(cache, provider)
        self._chunks: list[tuple[str, list[str]]] | None = None

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO, revision=REVISION)

    def _json(self, relpath: str) -> dict[str, typing.Any]:
        try:
            body, _ = self.provider.read(self.provider.url(relpath))
        except urllib.error.HTTPError as error:
            raise refused(error, REPO, SOURCE) from None
        return json.loads(body)

    @property
    def chunks(self) -> list[tuple[str, list[str]]]:
        """Chunk archives in application order with their model folders, read once from the releases and their manifests into the cache."""
        if self._chunks is None:
            path = self.cache / CHUNKS
            if not path.is_file():
                releases = self._json(RELEASES)
                manifests = {version["chunks_dir"]: self._json(f"{version['chunks_dir']}/{MANIFEST}") for version in releases["versions"]}
                self.cache.mkdir(parents=True, exist_ok=True)
                part = path.with_name(path.name + ".part")
                part.write_text(json.dumps(release_chunks(releases, manifests)))
                part.replace(path)
            self._chunks = [(archive, roots) for archive, roots in json.loads(path.read_text())]
        return self._chunks

    def roots(self) -> dict[str, str]:
        """Model folder of every item, `<category>/<id>`."""
        found: dict[str, str] = {}
        for _, roots in self.chunks:
            for root in roots:
                if match := _ROOT.fullmatch(root):
                    found.setdefault(f"{match[1]}/{match[2]}", root)
        return found

    def items(self) -> Iterator[str]:
        return iter(sorted(self.roots()))

    def _root(self, item: str) -> str:
        roots = self.roots()
        if item in roots:
            return roots[item]
        matches = [root for key, root in roots.items() if key.rsplit("/", 1)[-1] == item]
        if len(matches) != 1:
            raise KeyError(f"no {SOURCE} object '{item}', items are <category>/<id> such as armoire/7402940980c4a422cb8fd9b8e3886ff7, list them with: import object {self.meta.name}")
        return matches[0]

    def fetch(self, item: str) -> Path:
        root = self._root(item)
        model = root.rsplit("/", 1)[-1]
        dest = self.cache / "objects" / model
        glb = dest / f"{model}{GLB}"
        done = dest / ".complete"
        if done.is_file():
            return glb
        holding = [(archive, roots) for archive, roots in self.chunks if root in roots]
        for n, (archive, roots) in enumerate(holding, 1):
            logger.info("%s: reading %s from %s (%d of %d), model %d of %d in the chunk", SOURCE, root, archive, n, len(holding), roots.index(root) + 1, len(roots))
            try:
                with self.provider.open(archive) as stream:
                    extract(stream, root, dest)
            except urllib.error.HTTPError as error:
                raise refused(error, REPO, SOURCE) from None
        if not glb.is_file() or not (dest / f"{model}{JOINTS}").is_file():
            raise ValueError(f"{SOURCE} object {item} ships no {GLB} with {JOINTS} in any release")
        done.touch()
        return glb

    def convert(self, item: str, src: Path, out: Path) -> str:
        match = _ROOT.fullmatch(self._root(item))
        key = f"{match[1]}/{match[2]}" if match else item
        return build(lambda dataset: stage(key, src, dataset), self.cache, out, object_name(item))

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}#{self._root(item)}", license=LICENSE, authors=AUTHORS, citation=CITATION)
