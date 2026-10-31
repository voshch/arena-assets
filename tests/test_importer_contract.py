from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder
from arena_assets.importers import Importer
from arena_assets.semantic import CURRENT_ASA
from arena_assets.utils.Database import Database
from http_host import Host


@dataclass(frozen=True)
class Case:
    item: str
    name: str
    importer: Importer
    source: Callable[[], Path]
    stage: Callable[[Path], Path]
    articulated: bool = False


def ai2thor(tmp_path: Path, host: Host) -> Case:
    import test_ai2thor as t

    t.seed(tmp_path)
    imp = t.importer(tmp_path)
    return Case(t.ITEM, t.ITEM, imp, lambda: imp.fetch(t.ITEM), lambda dataset: t.stage(t.ITEM, imp.fetch(t.ITEM), dataset, t.OBJECT_TYPE))


def gso(tmp_path: Path, host: Host) -> Case:
    import test_gso as t

    src = t.seed(tmp_path, {"name": t.ITEM, "categories": ["Consumer Goods"]})
    return Case(t.ITEM, t.ITEM, t.importer(tmp_path), lambda: src, lambda dataset: t.stage(t.ITEM, src, dataset))


def openrobotics(tmp_path: Path, host: Host) -> Case:
    import test_openrobotics as t

    t.seed(tmp_path)
    imp = t.importer(tmp_path)
    return Case(t.ITEM, t.ITEM, imp, lambda: imp.fetch(t.ITEM), lambda dataset: t.OpenRobotics.stage(t.ITEM, imp.fetch(t.ITEM), dataset))


def artiverse(tmp_path: Path, host: Host) -> Case:
    pytest.importorskip("arena_simulation_setup")
    import test_artiverse as t

    t.item_files(tmp_path / "src")
    t.serve_releases(host, tmp_path / "src")
    imp = t.importer(host, tmp_path)
    item = f"armoire/{t.ITEM}"
    return Case(item, f"artiverse_{t.ITEM}", imp, lambda: imp.fetch(item), lambda dataset: t.stage(item, imp.fetch(item), dataset), articulated=True)


def artvip(tmp_path: Path, host: Host) -> Case:
    pytest.importorskip("arena_simulation_setup")
    import test_artvip as t

    t.repository(host)
    imp = t.importer(host, tmp_path)
    return Case(t.ITEM, t.ITEM, imp, lambda: imp.fetch(t.ITEM), lambda dataset: imp.stage(t.ITEM, imp.fetch(t.ITEM), dataset), articulated=True)


def partnet_mobility(tmp_path: Path, host: Host) -> Case:
    pytest.importorskip("arena_simulation_setup")
    import test_partnet_mobility as t

    src = t.item_zip(tmp_path)
    return Case(t.ITEM, t.NAME, t.importer(host, tmp_path), lambda: src, lambda dataset: t.stage(t.ITEM, src, dataset), articulated=True)


CASES = [ai2thor, gso, openrobotics, artiverse, artvip, partnet_mobility]


@pytest.fixture(params=CASES, ids=lambda make: make.__name__)
def case(request: pytest.FixtureRequest, tmp_path: Path, host: Host) -> Case:
    return request.param(tmp_path, host)


def convert(case: Case, out: Path) -> Path:
    relpath = Path("Common") / "Object" / case.name
    assert case.importer.convert(case.item, case.source(), out) == str(relpath)
    return out / relpath


def test_stage_is_object_builder_input(case, tmp_path):
    dataset = tmp_path / "dataset"
    source = case.stage(dataset)
    assert source == dataset / "Common" / "Object" / case.name
    found = list(ObjectDatabaseBuilder(input_path=dataset / "Common", output_path=tmp_path / "build").discover())
    assert [annotation.name for annotation in found] == [case.name]
    assert Path(found[0].path) == source


def test_stage_annotation_states_the_schema_description_origin_and_domain(case, tmp_path):
    raw = yaml.safe_load((case.stage(tmp_path / "dataset") / ANNOTATION_NAME).read_text())
    assert raw["asa"] == CURRENT_ASA
    assert raw["desc"]
    assert f"origin::{case.importer.meta.name}" in raw["tags"]
    assert any(tag.startswith("domain::") for tag in raw["tags"])


def test_attribution_states_the_source_license_and_url(case):
    attribution = case.importer.attribution(case.item)
    assert attribution.source == case.importer.meta.name
    assert attribution.license
    assert attribution.url


def test_convert_builds_a_native_object_item(blender, case, tmp_path):
    target = convert(case, tmp_path / "out")
    assert (target / f"{case.name}.usdz").stat().st_size > 0
    assert (target / f"{case.name}.fbx").stat().st_size > 0
    sdf = (target / f"{case.name}.sdf" / f"{case.name}.sdf").read_text()
    if not case.articulated:
        assert sdf.count(f"<uri>{case.name}.dae</uri>") == 2
    for preview in (f"{case.name}_thumb.png", f"{case.name}_topdown.png"):
        assert (target / preview).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not list((tmp_path / "cache").glob("stage-*"))
    assert not (tmp_path / "cache" / "stage" / case.name).exists()


def test_convert_indexes_the_item_in_the_domain_database(blender, case, tmp_path):
    convert(case, tmp_path / "out")
    assert Database(tmp_path / "out" / "Common" / DATABASE_NAME).list_all("object")["ids"] == [f"Object/{case.name}"]


def test_reconvert_replaces_the_item(blender, case, tmp_path):
    target = convert(case, tmp_path / "out")
    stale = target / "stale.txt"
    stale.write_text("x")
    convert(case, tmp_path / "out")
    assert not stale.exists()
    assert (target / f"{case.name}.usdz").is_file()
