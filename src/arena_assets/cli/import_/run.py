import json
import os
import tempfile
from pathlib import Path

import typer

from arena_assets.cli.utils import safe_echo


def _assets_root() -> Path | None:
    root = os.environ.get("ARENA_ASSETS_DIR")
    return Path(root) if root else None


def _default_out(importer: type) -> Path:
    try:
        return importer.default_out()
    except RuntimeError as error:
        raise typer.BadParameter(f"{error}, pass --out DIR", param_hint="--out") from error


def _target_hint(importer: type) -> str:
    try:
        return str(importer.default_out())
    except RuntimeError:
        return "unset, pass --out DIR"


def _default_cache(source: str) -> Path:
    root = _assets_root()
    if root is None:
        return Path(tempfile.gettempdir()) / "arena_assets_import" / source
    return root / ".import" / source


def _list_sources(as_json: bool, kind: str | None = None) -> None:
    from arena_assets.importers import IMPORTERS

    rows = [(meta.kind, meta.name, meta.license.value, meta.host, meta.on_demand.value, meta.description) for meta in IMPORTERS if kind is None or meta.kind == kind]
    if as_json:
        print(json.dumps([dict(zip(("kind", "name", "license", "host", "on_demand", "description"), row, strict=True)) for row in rows], indent=2))
        return
    header = ("kind", "source", "license", "host", "on_demand", "description")
    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header) - 1)]
    for row in (header, *rows):
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=False)) + f"  {row[-1]}")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"arena-assets import: {message}", err=True)
    return typer.Exit(1)


def _elsewhere(source: str) -> str:
    from arena_assets.importers import IMPORTERS

    kinds = IMPORTERS.kinds_of(source)
    return f", '{source}' is a source of kind {' and '.join(kinds)}: import {kinds[0]} {source}" if kinds else ""


def import_command(
    ctx: typer.Context,
    kind: str | None = typer.Argument(None, help="Kind of item to import, bare lists every kind and source"),
    source: str | None = typer.Argument(None, help="Upstream dataset of that kind, bare with KIND lists them"),
    items: list[str] = typer.Argument(None, help="Upstream item ids, bare with KIND SOURCE lists them"),
    all_: bool = typer.Option(False, "--all", help="Import every item the source lists, logging and skipping a failed one, exit 1 if any failed", is_flag=True),
    out: Path | None = typer.Option(None, "--out", "-o", help="Target root, default the imported root the resolvers read (shown by a bare import KIND SOURCE)"),
    cache: Path | None = typer.Option(None, "--cache", help="Download cache, default $ARENA_ASSETS_DIR/.import/SOURCE, else under the system temp dir"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable source or item listing", is_flag=True),
):
    """Import items of one kind from an upstream dataset into the asset layout the resolvers read.

    Bare, lists the registered sources by kind with license verdict (yes, nc, sa, no,
    unclear), host and on-demand granularity (item, chunk, bulk). With KIND alone, lists
    the sources of that kind. With KIND SOURCE, lists that source's items and the default
    target. Each imported item lands under OUT/<relpath> with an ATTRIBUTION.yaml, and its
    path is printed.
    """
    from arena_assets.importers import IMPORTERS, ITEM_ERRORS, reason

    if kind is None:
        _list_sources(as_json)
        return
    if kind not in IMPORTERS.kinds():
        raise typer.BadParameter(f"no importer kind '{kind}', kinds: {', '.join(IMPORTERS.kinds())}{_elsewhere(kind)}", param_hint="KIND")
    if source is None:
        _list_sources(as_json, kind)
        if not as_json:
            safe_echo(f"{len(IMPORTERS.names(kind))} source(s) of kind {kind}, list one's items with: import {kind} SOURCE", ctx)
        return
    if (kind, source) not in IMPORTERS:
        raise typer.BadParameter(f"no {kind} importer '{source}', registered: {', '.join(IMPORTERS.names(kind))}{_elsewhere(source)}", param_hint="SOURCE")
    if items and all_:
        raise typer.BadParameter("pass item ids or --all, not both", param_hint="ITEMS")

    importer_class = IMPORTERS.load(kind, source)
    importer = importer_class(cache=cache if cache is not None else _default_cache(source))

    if not items and not all_:
        try:
            listed = importer.cached_items()
        except ITEM_ERRORS as error:
            raise _fail(f"{kind} {source}: {reason(error)}") from error
        if as_json:
            print(json.dumps({"kind": kind, "source": source, "target": _target_hint(importer_class), "count": len(listed), "items": listed}))
            return
        count = 0
        for item in listed:
            print(item)
            count += 1
        safe_echo(f"{count} {kind} item(s) in {source}, import one with: import {kind} {source} ITEM [--out DIR], default target: {_target_hint(importer_class)}", ctx)
        return

    out = out if out is not None else _default_out(importer_class)
    if all_:
        try:
            names = list(importer.items())
            safe_echo(f"Importing {len(names)} {kind} item(s) from {source}", ctx)
            imported = importer.import_items(names, out)
        except ITEM_ERRORS as error:
            raise _fail(f"{kind} {source}: {reason(error)}") from error
        for path in imported.values():
            print(path)
        failed = len([item for item in names if item not in imported])
        if failed:
            raise _fail(f"imported {len(imported)} {kind} item(s) from {source}, {failed} failed, the warnings above name them")
        safe_echo(f"Imported {len(imported)} {kind} item(s) from {source}", ctx)
        return
    for item in items:
        safe_echo(f"Importing {kind} {source} {item}", ctx)
        try:
            print(importer.import_item(item, out))
        except (KeyError, ValueError) as error:
            raise typer.BadParameter(reason(error), param_hint="ITEM") from error
        except ITEM_ERRORS as error:
            raise _fail(f"{kind} {source} {item}: {reason(error)}") from error


def add_to_cmd(cmd: typer.Typer):
    """Add import command to the main CLI."""
    cmd.command("import")(import_command)


if __name__ == "__main__":
    typer.run(import_command)
