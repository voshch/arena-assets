import os

import typer

from arena_assets.cli.utils import safe_echo


def fetch_command(
    ctx: typer.Context,
    relative_paths: list[str] = typer.Argument(None, help="Paths inside the bucket to download"),
    output_dir: str = typer.Option(os.getcwd(), "--output", "-o", help="Directory to download the files to"),
    no_annotation: bool = typer.Option(False, "--no-annotation", help="Do not download annotation files", is_flag=True),
    model_formats: list[str] | None = typer.Option(
        None,
        "--format",
        help="Only download models of specified formats",
    ),
):
    """Fetch specific files from paths within the bucket."""
    source = ctx.obj["source"]

    if not relative_paths:
        relative_paths = [""]

    safe_echo(f"Fetching {len(relative_paths)} path(s) from bucket: {source}", ctx)
    safe_echo(f"Output directory: {output_dir}", ctx)

    from arena_assets.impl.fetch import fetch_database

    fetch_database(
        bucket=source,
        bucket_paths=relative_paths,
        destination=output_dir,
        annotations=not no_annotation,
        model_formats=model_formats,
    )

    safe_echo("Fetch completed successfully!", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add fetch command to the network command group."""
    cmd.command("fetch")(fetch_command)


if __name__ == "__main__":
    typer.run(fetch_command)
