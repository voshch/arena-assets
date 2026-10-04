import typer

from arena_assets.cli.utils import safe_echo


def list_command(
    ctx: typer.Context,
    prefix: str = typer.Argument("", help="Prefix to list assets under"),
    children: bool = typer.Option(False, "--children", help="List immediate child directories instead of globbing for annotated assets", is_flag=True),
):
    """List available assets under a prefix in the bucket.

    The default globs for annotation sentinels, which only model assets carry.
    Use --children for buckets whose payloads carry no annotation (worlds,
    benchmark configs), where the child directories are themselves the assets.
    """
    source = ctx.obj["source"]

    safe_echo(f"Listing assets in bucket: {source} under prefix: {prefix or '(root)'}", ctx)

    from arena_assets.impl.fetch import Bucket

    b = Bucket(source)
    assets = b.listchildren(prefix) if children else b.list_assets(prefix)
    for asset_dir in assets:
        print(asset_dir)

    safe_echo(f"Found {len(assets)} asset(s).", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add list command to the network command group."""
    cmd.command("list")(list_command)


if __name__ == "__main__":
    typer.run(list_command)
