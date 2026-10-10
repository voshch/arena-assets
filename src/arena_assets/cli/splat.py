"""Splat command: frame a trained Gaussian splat as a wall or object bundle."""

from pathlib import Path

import typer

from .utils import safe_echo

HELP = """Turn a trained 3D Gaussian splat into an object bundle for Isaac Sim. By default it frames a wall capture that wall kinds place with 'model: NAME'. With --object it frames a free-standing object that worlds and scenarios place like any other object.

SOURCE is a USD file with ParticleField prims, as 3DGRUT writes with export_usd.enabled=true. OUT_DIR receives NAME.gaussians.usdc (the kept Gaussians), NAME.usda (a Z-up wrapper in meters that carries the registration) and annotation.yaml.

Target frame (registration, crop and --drop), meters, Z up. Wall frame: origin at the base midpoint of the captured wall, X along the wall, Y facing into the room, Y = 0 on the captured wall plane. Object frame (--object): origin on the floor under the object center, X and Y level, Y toward the object front.

Registration, exactly one of --matrix (a known transform) or --points (fit one from point pairs, prints the fitted scale and the RMS residual).

Crop, in the target frame: X from -width/2 to +width/2 and Z from --floor to --height. Y runs from -behind to --depth for a wall and from -depth/2 to +depth/2 for an object. Anything that should stay must be wholly inside the crop, take the rest out with --drop.

Captured geometry is seen by lidar and depth cameras but has no collision. A wall crop therefore keeps 5 cm in front of the captured wall plane by default and deeper crops are opt-in. An object is visual only: annotation.yaml records the crop as its bounding box, and physics ignores it.

Bundle origin: for a wall, the wall base midpoint at the BACK of the crop, which is the wall frame shifted by +behind along Y. The physical wall of the wall kind sits there, so the whole capture lies in front of it. With --behind B the captured wall surface stands B meters in front of the physical wall. For an object, the object frame origin, so a pose at floor height stands it on the floor.

Next: check the printed counts, then publish with 'arena asset push object NAME'."""

EPILOG = """Examples: arena-assets splat lab.usdz ./Object/lab_wall --points lab_wall.txt --width 3.2 --height 2.5 --drop '-1.6 0 0 -1.1 0.05 0.9'
arena-assets splat garden.usdz ./Object/garden_table --object --points table.txt --width 1.2 --depth 0.9 --height 1.1"""


def _fail(message: str) -> typer.Exit:
    typer.echo(f"arena-assets splat: {message}, nothing was written", err=True)
    return typer.Exit(1)


def splat_command(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="USD file with the trained splat (ParticleField prims), for example the USD export of a 3DGRUT run"),
    out_dir: Path = typer.Argument(..., help="Bundle directory to write, created if missing, for example ./Object/lab_wall"),
    name: str | None = typer.Option(None, "--name", help="Bundle and prim name, a valid USD identifier. Defaults to the OUT_DIR basename"),
    object_: bool = typer.Option(False, "--object", help="Frame a free-standing object in the object frame (origin on the floor under its center) instead of a wall capture. Needs --depth and refuses --behind"),
    matrix: str | None = typer.Option(None, "--matrix", metavar="'16 NUMBERS'", help="Registration as 16 numbers in one quoted argument: the row-major 4x4 that maps the source stage frame to the target frame in meters (p_target = M @ p_stage, last row 0 0 0 1). Exclusive with --points"),
    points: Path | None = typer.Option(
        None, "--points", metavar="FILE", help="Registration from correspondences: one 'sx sy sz tx ty tz' per line (source stage point, target-frame point in meters), blank and # lines skipped, at least 3 non-collinear pairs. Fits uniform scale, rotation and translation. Exclusive with --matrix"
    ),
    width: float | None = typer.Option(None, "--width", help="Required. Crop width in meters, X from -width/2 to +width/2"),
    height: float | None = typer.Option(None, "--height", help="Required. Crop top in meters, Z from --floor to --height"),
    floor: float = typer.Option(0.03, "--floor", help="Crop bottom in meters, cuts the captured floor away"),
    depth: float | None = typer.Option(None, "--depth", help="Wall: how far in front of the captured wall plane kept geometry may reach, in meters (Y max, default 0.05). Kept geometry has no collision, so raise it on purpose. Object: required, the crop depth, Y from -depth/2 to +depth/2"),
    behind: float | None = typer.Option(
        None, "--behind", help="Wall only: how far behind the captured wall plane kept geometry may reach, in meters (Y min is -behind, default 0 keeps nothing behind it). The bundle is shifted by +behind along Y, so B puts the captured wall surface B meters in front of the physical wall"
    ),
    drop: list[str] = typer.Option([], "--drop", metavar="'6 NUMBERS'", help="Remove the Gaussians inside this target-frame box, 'XMIN YMIN ZMIN XMAX YMAX ZMAX' in meters as one quoted argument. Repeatable. Takes out things the crop would cut in half"),
    max_scale: float = typer.Option(0.2, "--max-scale", help="Drop Gaussians whose largest axis exceeds this many meters after registration, they blow up the extent"),
    min_opacity: float = typer.Option(0.0, "--min-opacity", help="Drop Gaussians below this opacity, 0 keeps all and 1 keeps only opaque ones"),
    desc: str | None = typer.Option(None, "--desc", help="Description for annotation.yaml. Defaults to a sentence with the crop size"),
    source_note: str | None = typer.Option(None, "--source", help="Provenance for annotation.yaml. Defaults to the SOURCE file name"),
):
    """Frame a trained Gaussian splat as a wall or object bundle."""
    from arena_assets.impl.splat import ObjectCrop, SplatError, WallCrop, fit_similarity, frame_splat, parse_drop, parse_matrix, read_points, write_annotation

    if matrix is None and points is None:
        raise _fail("no registration given, pass --matrix 'M00 ... M33' (16 numbers, row-major 4x4, source stage frame to the target frame in meters) or --points FILE (lines 'sx sy sz tx ty tz', at least 3 non-collinear pairs)")
    if matrix is not None and points is not None:
        raise _fail("--matrix and --points are exclusive, pass --matrix for a known transform or --points to fit one from correspondences")
    if width is None or height is None:
        raise _fail("--width and --height are required: the crop spans X from -width/2 to +width/2 and Z from --floor to --height, in target-frame meters")
    if object_ and depth is None:
        raise _fail("--object needs --depth: the object crop spans Y from -depth/2 to +depth/2 in object-frame meters")
    if object_ and behind is not None:
        raise _fail("--behind applies to walls only, an object crop is centered on its origin: drop --behind or set --depth to the full crop depth")

    name = name or out_dir.resolve().name
    try:
        if object_:
            crop = ObjectCrop(width=width, depth=depth, height=height, floor=floor)
        else:
            crop = WallCrop(width=width, height=height, floor=floor, depth=0.05 if depth is None else depth, behind=behind or 0.0)
        drops = [parse_drop(box) for box in drop]
        if matrix is not None:
            to_target = parse_matrix(matrix)
        else:
            fit = fit_similarity(*read_points(points))
            to_target = fit.matrix
            safe_echo(f"fitted scale {fit.scale:.6g}, RMS residual {fit.rms:.4g} m over {fit.pairs} pairs", ctx)
        bundle = frame_splat(source, out_dir, name, to_target, crop, drops, max_scale=max_scale, min_opacity=min_opacity)
    except SplatError as error:
        raise _fail(str(error)) from None

    write_annotation(out_dir, name, crop, desc or crop.desc, source_note or source.name)
    safe_echo(f"kept {bundle.kept} of {bundle.total} particles", ctx)
    typer.echo(str(out_dir))


def add_to_cmd(cmd: typer.Typer):
    cmd.command("splat", help=HELP, epilog=EPILOG, no_args_is_help=True)(splat_command)
