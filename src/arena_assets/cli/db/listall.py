"""List command for database operations."""

import os

import typer

from ...impl import AssetType
from ..utils import safe_echo


def list_cmd(
    ctx: typer.Context,
    asset_type: str = typer.Argument(
        ...,
        help=f"Type of asset to query. Options: {', '.join([e.name.lower() for e in AssetType])}",
    ),
):
    """Search for models in the database using natural language."""
    database_path = ctx.obj.get("database_path") if ctx.obj else None
    if not database_path:
        safe_echo("Error: Database path not found in context", ctx)
        raise typer.Exit(1)

    try:
        asset_type_enum = AssetType[asset_type.upper()]
    except KeyError:
        valid_types = [e.name for e in AssetType]
        safe_echo(
            f"Invalid asset_type '{asset_type}'. Valid options: {', '.join(valid_types)}",
            ctx,
        )
        raise typer.Exit(1) from None

    safe_echo(f"Searching for {asset_type_enum.name} in database {database_path}", ctx)

    from arena_assets.impl.listall import list_database

    result = list_database(database_path=database_path, asset_type=asset_type_enum)
    for item in result:
        typer.echo(os.path.join(database_path, item.path))


def add_to_cmd(db_cmd: typer.Typer):
    db_cmd.command("list")(list_cmd)


if __name__ == "__main__":
    typer.run(list_cmd)
