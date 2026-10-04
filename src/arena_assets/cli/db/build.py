from pathlib import Path

import typer

from arena_assets.impl import AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.utils.logging import get_logger, get_manager

logger = get_logger("cli.db.build")


def build_command(
    ctx: typer.Context,
    asset_type: list[str] = typer.Option(
        [],
        "--buildtype",
        "-t",
        help="Type of asset to build. Optional, defaults to all types. Specify multiple times for multiple types.",
    ),
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input dataset directory path",
        file_okay=False,
        dir_okay=True,
        readable=True,
        exists=True,
    ),
    options: list[str] = typer.Option(
        [],
        "--option",
        "-o",
        help="Options for the build process",
    ),
    overwrite: OverwriteMode = typer.Option(
        OverwriteMode.SKIP,
        "--overwrite",
        help=("Overwrite behavior: 'skip' keeps existing entries, 'overwrite' rebuilds them, 'annotations' assumes files are built and only updates annotations."),
    ),
):
    """Build a database from source model files."""

    database_path = ctx.obj.get("database_path") if ctx.obj else None
    if database_path is None:
        logger.error("No database path configured. Set it via the parent command's --database option.")
        raise typer.Exit(1)
    output_path = Path(database_path)

    if asset_type:
        try:
            asset_type_enum = [AssetType[t.upper()] for t in asset_type]
        except KeyError:
            valid_types = [e.name for e in AssetType]
            logger.error(f"Invalid asset_type '{asset_type}'. Valid options: {', '.join(valid_types)}")
            raise typer.Exit(1) from None
    else:
        asset_type_enum = list(DatabaseBuilder.get_registered())

    logger.info(f"Starting database build from {input_path} to {output_path} with types: {', '.join([t.name for t in asset_type_enum])}")
    manager = get_manager()
    _ = manager.status_bar()  # bug in display removes the progress bar at the end otherwise
    global_progress = manager.counter(
        total=len(asset_type_enum),
        desc="Databases",
        bar_format="{desc}{desc_pad}{count:{len_total}d}/{total:d} [{elapsed}]",
        autorefresh=True,
    )
    with global_progress:
        for t in global_progress(asset_type_enum):
            logger.info(f"Building {t} database from {input_path} to {output_path}")

            builder = DatabaseBuilder.Builder(t)(input_path=input_path, output_path=output_path, overwrite=overwrite)
            for extra in options:
                builder.enable(*extra.split("="))
            try:
                builder.build()
            except Exception as e:
                logger.error(f"Failed to build {t} database: {e}")
                import traceback

                traceback.print_exc()
                raise

    logger.info(f"Database build completed in {global_progress.elapsed:.1f}s")


def add_to_cmd(db_cmd: typer.Typer):
    options = [""]
    for builder, option in DatabaseBuilder.get_all_options().items():
        options.append(f"{builder.name}: {', '.join(option)}")
    merged_options = "\n\t".join(options)
    expanded_help = f"{build_command.__doc__}\n\nAvailable options: {merged_options}\n"
    db_cmd.command("build", help=expanded_help)(build_command)


if __name__ == "__main__":
    typer.run(build_command)
