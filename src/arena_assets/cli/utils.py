"""Utility functions for CLI operations."""

import typer


def safe_echo(message: str, ctx: typer.Context | None = None):
    """Echo a message only if not in silent mode."""
    if ctx and ctx.obj and ctx.obj.get("silent", False):
        return
    typer.echo(message, err=True)


def get_silent_flag(ctx: typer.Context) -> bool:
    """Get the silent flag from context."""
    return ctx.obj.get("silent", False) if ctx and ctx.obj else False
