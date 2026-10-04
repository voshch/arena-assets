"""Database command group for arena_assets CLI."""

import typer

from .build import add_to_cmd as add_build
from .listall import add_to_cmd as add_list
from .query import add_to_cmd as add_query


def add_to_cmd(cmd: typer.Typer):
    """Add database command group to the main CLI."""

    db_app = typer.Typer(
        name="db",
        help="Database operations for arena assets",
        rich_markup_mode="rich",
        no_args_is_help=True,
    )

    @db_app.callback()
    def db_callback(
        ctx: typer.Context,
        database_path: str = typer.Argument(..., help="Path to database directory"),
    ):
        """Database operations with specified database path."""
        if ctx:
            ctx.ensure_object(dict)
            ctx.obj["database_path"] = database_path
            if ctx.parent and ctx.parent.obj:
                ctx.obj["silent"] = ctx.parent.obj.get("silent", False)

    add_build(db_app)
    add_query(db_app)
    add_list(db_app)

    cmd.add_typer(db_app, name="db")
