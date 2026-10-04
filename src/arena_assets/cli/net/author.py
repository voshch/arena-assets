from pathlib import Path

import typer

from arena_assets.cli.utils import safe_echo


def author_command(
    ctx: typer.Context,
    database_dir: Path = typer.Argument(
        ...,
        help="Built database directory to upload",
        file_okay=False,
        dir_okay=True,
        readable=True,
        exists=True,
    ),
    destination: str = typer.Option("", "--destination", "-d", help="Path inside the bucket to upload under"),
    token: str | None = typer.Option(None, "--token", help="GCS access token (defaults to GCS_ACCESS_TOKEN or gcloud)"),
):
    """Upload a built database to the bucket."""
    source = ctx.obj["source"]

    safe_echo(f"Uploading database from {database_dir} to bucket: {source}", ctx)

    from arena_assets.impl.author import author_database

    author_database(
        bucket=source,
        source=str(database_dir),
        destination=destination,
        token=token,
    )

    safe_echo("Upload completed successfully!", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add author command to the network command group."""
    cmd.command("author")(author_command)


if __name__ == "__main__":
    typer.run(author_command)
