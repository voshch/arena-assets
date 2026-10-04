"""CLI module for arena_assets."""

import typer

from arena_assets.cli.db import add_to_cmd as add_db
from arena_assets.cli.net import add_to_cmd as add_net
from arena_assets.cli.semantic import add_to_cmd as add_semantic


def add_cmd(app: typer.Typer):
    add_db(app)
    add_net(app)
    add_semantic(app)
