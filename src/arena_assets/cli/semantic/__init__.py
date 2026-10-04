"""Semantic (ASA) command group for arena_assets CLI."""

import typer

from .lint import add_to_cmd as add_lint


def add_to_cmd(cmd: typer.Typer):
    """Add semantic command group to the main CLI."""

    semantic_app = typer.Typer(
        name="semantic",
        help="ASA (Arena Semantic Annotations) tooling",
        rich_markup_mode="rich",
        no_args_is_help=True,
    )

    add_lint(semantic_app)

    cmd.add_typer(semantic_app, name="semantic")
