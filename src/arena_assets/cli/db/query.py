"""Query command for database operations."""

import os
import re

import typer

from ...impl import AssetType
from ..utils import safe_echo

_FILTER_OPS = {"<=": "$lte", ">=": "$gte", "<": "$lt", ">": "$gt", "=": "$eq"}


def parse_filters(filters: list[str]) -> dict | None:
    clauses = []
    for expression in filters:
        match = re.fullmatch(r"\s*(\w+)\s*(<=|>=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*", expression)
        if match is None:
            raise typer.BadParameter(f"Invalid filter '{expression}', expected e.g. 'height<0.75'")
        field, op, value = match.groups()
        clauses.append({field: {_FILTER_OPS[op]: float(value)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query_command(
    ctx: typer.Context,
    asset_type: str = typer.Argument(
        ...,
        help=f"Type of asset to query. Options: {', '.join([e.name.lower() for e in AssetType])}",
    ),
    query_text: str = typer.Argument(..., help="Text description to search for"),
    count: int = typer.Option(1, "--count", "-n", min=1, help="Number of results to return"),
    scores: bool = typer.Option(False, "--scores", help="Append the distance score to each result"),
    filters: list[str] = typer.Option([], "--filter", help="Numeric metadata filter like 'height<0.75'. Repeatable."),
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

    safe_echo(
        f"Searching for {asset_type_enum.name} '{query_text}' in database {database_path}",
        ctx,
    )

    from arena_assets.impl.query import query_database

    results = query_database(database_path=database_path, asset_type=asset_type_enum, query_target=query_text, n=count, where=parse_filters(filters))
    for annotation, score in results:
        line = os.path.join(database_path, annotation.path)
        if scores:
            line += f"\t{score}"
        typer.echo(line)


def add_to_cmd(db_cmd: typer.Typer):
    db_cmd.command("query")(query_command)


if __name__ == "__main__":
    typer.run(query_command)
