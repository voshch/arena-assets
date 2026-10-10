import re
from pathlib import Path

import attrs
import numpy as np
import pytest
import yaml
from pxr import Gf, Usd, UsdGeom, UsdVol, Vt
from typer.testing import CliRunner

from arena_assets.__main__ import app

runner = CliRunner()

COUNT = 400
SH_PER_PARTICLE = 16
HUGE = np.arange(0, 10)
MEDIUM = np.arange(10, 20)
WIDTH, HEIGHT, FLOOR, DEPTH, BEHIND = 3.0, 2.5, 0.03, 0.05, 0.3
CROP = np.array([[-WIDTH / 2, -BEHIND, FLOOR], [WIDTH / 2, DEPTH, HEIGHT]])
FRONT_CROP = np.array([[-WIDTH / 2, 0.0, FLOOR], [WIDTH / 2, DEPTH, HEIGHT]])
SIZE = ("--width", str(WIDTH), "--height", str(HEIGHT))
DROP = np.array([[-1.5, -0.1, 0.0], [-0.5, 0.05, 1.0]])


def similarity(scale: float, axis: tuple[float, float, float], degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = scale * np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*axis), degrees))).T
    matrix[:3, 3] = translation
    return matrix


Y_UP_TO_Z_UP = similarity(1.0, (1, 0, 0), 90.0, (0, 0, 0))
TO_WALL = similarity(0.25, (0, 0, 1), 20.0, (0.4, -0.2, 0.1)) @ Y_UP_TO_Z_UP
TO_BUNDLE = similarity(1.0, (0, 0, 1), 0.0, (0, BEHIND, 0)) @ TO_WALL
METERS_PER_LOCAL_UNIT = 0.5


@attrs.frozen
class Source:
    path: Path
    wall: np.ndarray
    positions: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    harmonics: np.ndarray
    to_stage: np.ndarray

    def inside(self, box: np.ndarray) -> np.ndarray:
        return np.all((self.wall >= box[0]) & (self.wall <= box[1]), axis=1)

    def kept(self, max_scale: float = 0.2, min_opacity: float = 0.0, drop: np.ndarray | None = None) -> np.ndarray:
        mask = self.inside(CROP)
        mask &= self.scales.astype(np.float64).max(1) * METERS_PER_LOCAL_UNIT <= max_scale
        mask &= self.opacities.astype(np.float64) >= min_opacity
        if drop is not None:
            mask &= ~self.inside(drop)
        return np.nonzero(mask)[0]


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def flat(values: np.ndarray) -> str:
    return " ".join(repr(float(value)) for value in np.ravel(values))


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Source:
    rng = np.random.default_rng(7)
    path = tmp_path_factory.mktemp("capture") / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    world.AddRotateYOp().Set(30.0)
    world.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, "/World/gauss")
    to_stage = np.asarray(UsdGeom.Xformable(field).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

    low, high = np.array([-2.0, -0.6, -0.2]), np.array([2.0, 0.4, 3.0])
    wall = np.floor(rng.uniform(low, high, (COUNT, 3)) * 100) / 100 + 0.005
    wall[: len(HUGE) + len(MEDIUM)] = np.floor(rng.uniform([-1.4, -0.2, 1.1], [1.4, 0.04, 2.4], (len(HUGE) + len(MEDIUM), 3)) * 100) / 100 + 0.005
    positions = apply(np.linalg.inv(TO_WALL @ to_stage), wall).astype(np.float32)
    scales = rng.uniform(0.01, 0.04, (COUNT, 3)).astype(np.float32)
    scales[HUGE, 1] = 0.5
    scales[MEDIUM, 2] = 0.3
    opacities = rng.uniform(0.2, 1.0, COUNT).astype(np.float32)
    orientations = rng.normal(size=(COUNT, 4))
    orientations = (orientations / np.linalg.norm(orientations, axis=1, keepdims=True)).astype(np.float32)
    harmonics = np.stack(
        [
            np.repeat(np.arange(COUNT), SH_PER_PARTICLE),
            np.tile(np.arange(SH_PER_PARTICLE), COUNT),
            rng.uniform(-1.0, 1.0, COUNT * SH_PER_PARTICLE),
        ],
        axis=1,
    ).astype(np.float32)

    field.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(positions))
    field.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(orientations))
    field.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales))
    field.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacities))
    field.CreateRadianceSphericalHarmonicsDegreeAttr(3)
    field.CreateRadianceSphericalHarmonicsCoefficientsAttr(Vt.Vec3fArray.FromNumpy(harmonics))
    field.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([positions.min(0) - 2, positions.max(0) + 2], np.float32)))
    stage.GetRootLayer().Save()
    return Source(path=path, wall=wall, positions=positions, scales=scales, opacities=opacities, harmonics=harmonics, to_stage=to_stage)


def splat(source: Path, out: Path, *extra: str, registration: tuple[str, ...] = ("--matrix", flat(TO_WALL)), crop: tuple[str, ...] = (*SIZE, "--behind", str(BEHIND))):
    return runner.invoke(app, ["splat", str(source), str(out), *registration, *crop, *extra])


def only_field(stage: Usd.Stage) -> Usd.Prim:
    fields = [prim for prim in stage.Traverse() if prim.IsA(UsdVol.ParticleField)]
    assert len(fields) == 1
    return fields[0]


def array(path: Path, name: str) -> np.ndarray:
    stage = Usd.Stage.Open(str(path))
    return np.array(only_field(stage).GetAttribute(name).Get())


def assert_refused(result, out: Path, *needles: str) -> None:
    assert result.exit_code != 0
    message = result.output.strip()
    assert message.startswith("arena-assets splat: ")
    assert message.endswith("nothing was written")
    assert "\n" not in message
    for needle in needles:
        assert needle in message
    assert not out.exists()


def test_crop_keeps_exactly_the_particles_inside_the_wall_frame_box(source, tmp_path):
    out = tmp_path / "lab_wall"
    result = splat(source.path, out)
    assert result.exit_code == 0, result.output
    kept = source.kept()
    assert 0 < len(kept) < source.inside(CROP).sum()
    assert f"kept {len(kept)} of {COUNT} particles" in result.output
    assert result.output.strip().splitlines()[-1] == str(out)
    assert sorted(path.name for path in out.iterdir()) == ["annotation.yaml", "lab_wall.gaussians.usdc", "lab_wall.usda"]
    assert len(array(out / "lab_wall.gaussians.usdc", "positions")) == len(kept)


def test_per_particle_and_harmonics_arrays_stay_aligned(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert splat(source.path, out).exit_code == 0
    kept = source.kept()
    bundle = out / "lab_wall.gaussians.usdc"
    stage = Usd.Stage.Open(str(bundle))
    assert only_field(stage).GetTypeName() == "ParticleField3DGaussianSplat"
    assert only_field(stage).GetAttribute("radiance:sphericalHarmonicsDegree").Get() == 3
    np.testing.assert_array_equal(array(bundle, "positions"), source.positions[kept])
    np.testing.assert_array_equal(array(bundle, "scales"), source.scales[kept])
    np.testing.assert_array_equal(array(bundle, "opacities"), source.opacities[kept])
    np.testing.assert_array_equal(array(bundle, "orientations"), array(source.path, "orientations")[kept])
    harmonics = array(bundle, "radiance:sphericalHarmonicsCoefficients").reshape(len(kept), SH_PER_PARTICLE, 3)
    np.testing.assert_array_equal(harmonics, source.harmonics.reshape(COUNT, SH_PER_PARTICLE, 3)[kept])
    np.testing.assert_array_equal(harmonics[:, :, 0], np.repeat(kept[:, None], SH_PER_PARTICLE, axis=1))
    np.testing.assert_array_equal(harmonics[:, :, 1], np.tile(np.arange(SH_PER_PARTICLE), (len(kept), 1)))


def test_drop_removes_the_particles_inside_each_box(source, tmp_path):
    out = tmp_path / "lab_wall"
    second = np.array([[0.5, -0.3, 1.5], [1.5, 0.05, 2.5]])
    result = splat(source.path, out, "--drop", flat(DROP), "--drop", flat(second).replace(" ", ","))
    assert result.exit_code == 0, result.output
    kept = np.setdiff1d(source.kept(drop=DROP), np.nonzero(source.inside(second))[0])
    assert len(kept) < len(source.kept(drop=DROP)) < len(source.kept())
    np.testing.assert_array_equal(array(out / "lab_wall.gaussians.usdc", "positions"), source.positions[kept])


def test_max_scale_filters_on_the_registered_metric_size(source, tmp_path):
    default = tmp_path / "default"
    assert splat(source.path, default).exit_code == 0
    kept = source.kept()
    assert not np.intersect1d(kept, HUGE).size
    assert np.intersect1d(kept, MEDIUM).size == len(MEDIUM)

    tight = tmp_path / "tight"
    assert splat(source.path, tight, "--max-scale", "0.1").exit_code == 0
    np.testing.assert_array_equal(array(tight / "tight.gaussians.usdc", "positions"), source.positions[np.setdiff1d(kept, MEDIUM)])

    loose = tmp_path / "loose"
    assert splat(source.path, loose, "--max-scale", "0.3").exit_code == 0
    np.testing.assert_array_equal(array(loose / "loose.gaussians.usdc", "positions"), source.positions[source.kept(max_scale=0.3)])
    assert np.intersect1d(source.kept(max_scale=0.3), HUGE).size == len(HUGE)


def test_min_opacity_filters_faint_particles(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert splat(source.path, out, "--min-opacity", "0.6").exit_code == 0
    kept = source.kept(min_opacity=0.6)
    assert 0 < len(kept) < len(source.kept())
    opacities = array(out / "lab_wall.gaussians.usdc", "opacities")
    np.testing.assert_array_equal(opacities, source.opacities[kept])
    assert float(opacities.min()) >= 0.6


def test_extent_is_the_kept_positions_padded_by_three_scales(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert splat(source.path, out).exit_code == 0
    kept = source.kept()
    pad = 3 * source.scales[kept].max()
    extent = array(out / "lab_wall.gaussians.usdc", "extent")
    np.testing.assert_allclose(extent[0], source.positions[kept].min(0) - pad, atol=1e-5)
    np.testing.assert_allclose(extent[1], source.positions[kept].max(0) + pad, atol=1e-5)
    assert np.all(extent[1] - extent[0] < array(source.path, "extent")[1] - array(source.path, "extent")[0])


def test_wrapper_composes_the_registration_over_the_source_transform(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert splat(source.path, out).exit_code == 0
    kept = source.kept()
    stage = Usd.Stage.Open(str(out / "lab_wall.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/lab_wall"
    prim = only_field(stage)
    assert prim.GetPath() == "/lab_wall/splat/Field_0"
    composed = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    world = apply(composed, array(out / "lab_wall.usda", "positions").astype(np.float64))
    source_world = apply(source.to_stage, source.positions[kept].astype(np.float64))
    np.testing.assert_allclose(world, apply(TO_WALL, source_world) + [0, BEHIND, 0], atol=1e-9)
    np.testing.assert_allclose(world, source.wall[kept] + [0, BEHIND, 0], atol=1e-5)
    assert np.all((world >= [-WIDTH / 2, 0, FLOOR]) & (world <= [WIDTH / 2, BEHIND + DEPTH, HEIGHT]))
    assert world[:, 1].min() < BEHIND < world[:, 1].max()


def test_default_keeps_nothing_behind_the_wall_plane(source, tmp_path):
    out = tmp_path / "lab_wall"
    result = splat(source.path, out, crop=SIZE)
    assert result.exit_code == 0, result.output
    front = np.nonzero(source.inside(FRONT_CROP) & (source.scales.max(1) * METERS_PER_LOCAL_UNIT <= 0.2))[0]
    assert 0 < len(front) < len(source.kept())
    np.testing.assert_array_equal(array(out / "lab_wall.gaussians.usdc", "positions"), source.positions[front])
    stage = Usd.Stage.Open(str(out / "lab_wall.usda"))
    composed = np.asarray(UsdGeom.Xformable(only_field(stage)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    world = apply(composed, source.positions[front].astype(np.float64))
    np.testing.assert_allclose(world, source.wall[front], atol=1e-5)
    assert world[:, 1].min() >= 0
    np.testing.assert_allclose(np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/lab_wall/splat")).GetLocalTransformation()).T, TO_WALL, atol=1e-12)


def test_points_recover_a_known_similarity(source, tmp_path):
    rng = np.random.default_rng(3)
    stage_points = rng.uniform(-5.0, 5.0, (5, 3))
    points = tmp_path / "pairs.txt"
    lines = [flat(pair) for pair in np.hstack([stage_points, apply(TO_WALL, stage_points)])]
    points.write_text("# sx sy sz tx ty tz\n\n" + "\n".join(lines) + "\n")
    out = tmp_path / "lab_wall"
    result = splat(source.path, out, registration=("--points", str(points)))
    assert result.exit_code == 0, result.output
    scale, rms = re.search(r"fitted scale (\S+), RMS residual (\S+) m over 5 pairs", result.output).groups()
    assert float(scale) == pytest.approx(0.25, abs=1e-6)
    assert float(rms) < 1e-9
    stage = Usd.Stage.Open(str(out / "lab_wall.usda"))
    fitted = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/lab_wall/splat")).GetLocalTransformation()).T
    np.testing.assert_allclose(fitted, TO_BUNDLE, atol=1e-9)
    np.testing.assert_array_equal(array(out / "lab_wall.gaussians.usdc", "positions"), source.positions[source.kept()])


def test_points_with_fewer_than_three_pairs_are_refused(source, tmp_path):
    points = tmp_path / "pairs.txt"
    points.write_text("0 0 0 0 0 0\n# 1 1 1 1 1 1\n1 0 0 0.25 0 0\n")
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--points", str(points))), out, "holds 2 pairs", "at least 3 non-collinear", "sx sy sz tx ty tz")


def test_collinear_points_are_refused(source, tmp_path):
    stage_points = np.outer(np.arange(4.0), [1.0, 2.0, 3.0])
    points = tmp_path / "pairs.txt"
    points.write_text("\n".join(flat(pair) for pair in np.hstack([stage_points, apply(TO_WALL, stage_points)])))
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--points", str(points))), out, "4 collinear pairs", "add a pair off that line")


def test_points_line_without_six_numbers_is_refused(source, tmp_path):
    points = tmp_path / "pairs.txt"
    points.write_text("0 0 0 0 0 0\n1 0 0 0.25 0\n0 1 0 0 0.25 0\n")
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--points", str(points))), out, "pairs.txt:2", "takes 6 finite numbers", "got 5")


def test_missing_points_file_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--points", str(tmp_path / "absent.txt"))), out, "absent.txt does not exist", "sx sy sz tx ty tz")


def test_annotation_describes_the_crop_box_from_its_back(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert splat(source.path, out, crop=SIZE).exit_code == 0
    annotation = yaml.safe_load((out / "annotation.yaml").read_text())
    assert annotation == {
        "asa": 1,
        "bounding_box": [[-1.5, 1.5], [0.0, 0.05], [0.03, 2.5]],
        "desc": "Gaussian splat capture of a wall, 3 m wide and 2.5 m high, 0.05 m deep.",
        "face": "+y",
        "name": "lab_wall",
        "note": "",
        "path": "Object/lab_wall",
        "source": "scene.usda",
        "tags": ["domain::common", "kind::splat"],
    }


def test_annotation_takes_name_desc_source_and_crop_options(source, tmp_path):
    out = tmp_path / "bundle"
    crop = ("--width", "2", "--height", "2.2", "--floor", "0.1", "--depth", "0.4", "--behind", "0.5")
    result = splat(source.path, out, "--name", "hall", "--desc", "Hallway wall with a notice board.", "--source", "hall take 3", crop=crop)
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in out.iterdir()) == ["annotation.yaml", "hall.gaussians.usdc", "hall.usda"]
    annotation = yaml.safe_load((out / "annotation.yaml").read_text())
    assert annotation["name"] == "hall"
    assert annotation["path"] == "Object/hall"
    assert annotation["desc"] == "Hallway wall with a notice board."
    assert annotation["source"] == "hall take 3"
    assert annotation["bounding_box"] == [[-1.0, 1.0], [0.0, 0.9], [0.1, 2.2]]
    box = np.array([[-1.0, -0.5, 0.1], [1.0, 0.4, 2.2]])
    expected = np.nonzero(source.inside(box) & (source.scales.max(1) * METERS_PER_LOCAL_UNIT <= 0.2))[0]
    np.testing.assert_array_equal(array(out / "hall.gaussians.usdc", "positions"), source.positions[expected])


def test_bare_verb_prints_usage():
    result = runner.invoke(app, ["splat"])
    output = " ".join(result.output.split())
    assert "Usage" in output
    for needle in ("SOURCE", "OUT_DIR", "--object", "--matrix", "--points", "--width", "--height", "--drop", "Wall frame", "Object frame", "arena asset push object"):
        assert needle in output


def test_help_documents_every_option():
    result = runner.invoke(app, ["splat", "--help"])
    assert result.exit_code == 0
    for option in ("--name", "--object", "--matrix", "--points", "--width", "--height", "--floor", "--depth", "--behind", "--drop", "--max-scale", "--min-opacity", "--desc", "--source"):
        assert option in result.output


def test_missing_registration_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=()), out, "no registration given", "--matrix", "--points FILE")


def test_matrix_together_with_points_is_refused(source, tmp_path):
    points = tmp_path / "pairs.txt"
    points.write_text("0 0 0 0 0 0\n")
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--matrix", flat(TO_WALL), "--points", str(points))), out, "--matrix and --points are exclusive")


def test_missing_crop_size_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, crop=("--width", "3")), out, "--width and --height are required")


def test_crop_with_height_below_floor_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, crop=("--width", "3", "--height", "0.02")), out, "--height (0.02) must be above --floor (0.03)")


def test_matrix_without_sixteen_numbers_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--matrix", "1 0 0 0 1 0 0 0 1")), out, "takes 16 finite numbers", "got 9", "row-major 4x4")


def test_transposed_matrix_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, registration=("--matrix", flat(TO_WALL.T))), out, "must end with the row 0 0 0 1", "transposed")


def test_drop_without_six_numbers_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, "--drop", "0 0 0 1 one"), out, "--drop", "XMIN YMIN ZMIN XMAX YMAX ZMAX", "takes 6 numbers")


def test_drop_with_swapped_corners_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, "--drop", "1 0 0 0 1 1"), out, "minimum corner first")


def test_source_without_particle_field_is_refused(tmp_path):
    path = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/mesh")
    stage.GetRootLayer().Save()
    out = tmp_path / "lab_wall"
    assert_refused(splat(path, out), out, "has no ParticleField prim", "export_usd.enabled=true")


def test_missing_source_is_refused(tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(tmp_path / "absent.usdz", out), out, "absent.usdz does not exist")


def test_unreadable_source_is_refused(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a stage")
    out = tmp_path / "lab_wall"
    assert_refused(splat(path, out), out, "is not a readable USD stage")


def test_crop_that_keeps_no_particle_is_refused(source, tmp_path):
    out = tmp_path / "lab_wall"
    assert_refused(splat(source.path, out, "--min-opacity", "2"), out, f"keeps 0 of {COUNT} particles", "--width", "--min-opacity")


def test_name_that_is_no_usd_identifier_is_refused(source, tmp_path):
    out = tmp_path / "lab-wall"
    assert_refused(splat(source.path, out), out, "'lab-wall' is not a valid USD prim name", "--name")


OBJECT_SIZE = ("--object", "--width", "2", "--depth", "0.8", "--height", "2")
OBJECT_CROP = np.array([[-1.0, -0.4, FLOOR], [1.0, 0.4, 2.0]])


def test_object_crop_keeps_the_box_centered_over_the_origin(source, tmp_path):
    out = tmp_path / "table"
    result = splat(source.path, out, crop=OBJECT_SIZE)
    assert result.exit_code == 0, result.output
    kept = np.nonzero(source.inside(OBJECT_CROP) & (source.scales.max(1) * METERS_PER_LOCAL_UNIT <= 0.2))[0]
    assert 0 < len(kept) < COUNT
    assert f"kept {len(kept)} of {COUNT} particles" in result.output
    np.testing.assert_array_equal(array(out / "table.gaussians.usdc", "positions"), source.positions[kept])
    stage = Usd.Stage.Open(str(out / "table.usda"))
    np.testing.assert_allclose(np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/table/splat")).GetLocalTransformation()).T, TO_WALL, atol=1e-12)
    composed = np.asarray(UsdGeom.Xformable(only_field(stage)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    world = apply(composed, source.positions[kept].astype(np.float64))
    np.testing.assert_allclose(world, source.wall[kept], atol=1e-5)
    assert world[:, 1].min() < 0 < world[:, 1].max()


def test_object_annotation_is_the_centered_crop(source, tmp_path):
    out = tmp_path / "table"
    assert splat(source.path, out, crop=OBJECT_SIZE).exit_code == 0
    annotation = yaml.safe_load((out / "annotation.yaml").read_text())
    assert annotation["bounding_box"] == [[-1.0, 1.0], [-0.4, 0.4], [0.03, 2.0]]
    assert annotation["desc"] == "Gaussian splat capture of an object, 2 m wide, 0.8 m deep and 2 m high."
    assert annotation["tags"] == ["domain::common", "kind::splat"]
    assert annotation["face"] == "+y"


def test_object_drop_uses_the_object_frame(source, tmp_path):
    out = tmp_path / "table"
    box = np.array([[-1.0, -0.4, 0.0], [0.0, 0.4, 1.0]])
    assert splat(source.path, out, "--drop", flat(box), crop=OBJECT_SIZE).exit_code == 0
    kept = np.nonzero(source.inside(OBJECT_CROP) & ~source.inside(box) & (source.scales.max(1) * METERS_PER_LOCAL_UNIT <= 0.2))[0]
    np.testing.assert_array_equal(array(out / "table.gaussians.usdc", "positions"), source.positions[kept])


def test_object_without_depth_is_refused(source, tmp_path):
    out = tmp_path / "table"
    assert_refused(splat(source.path, out, crop=("--object", *SIZE)), out, "--object needs --depth", "-depth/2 to +depth/2")


def test_object_with_behind_is_refused(source, tmp_path):
    out = tmp_path / "table"
    assert_refused(splat(source.path, out, crop=(*OBJECT_SIZE, "--behind", "0.2")), out, "--behind applies to walls only")


def test_object_with_nonpositive_depth_is_refused(source, tmp_path):
    out = tmp_path / "table"
    assert_refused(splat(source.path, out, crop=("--object", *SIZE, "--depth", "0")), out, "--depth must be positive")


def test_object_crop_that_keeps_no_particle_names_the_object_frame(source, tmp_path):
    out = tmp_path / "table"
    assert_refused(splat(source.path, out, "--min-opacity", "2", crop=OBJECT_SIZE), out, f"keeps 0 of {COUNT} particles", "the object frame is meters")
