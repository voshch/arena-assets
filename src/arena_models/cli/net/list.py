import typer

from arena_models.cli.utils import safe_echo


def list_command(
    ctx: typer.Context,
    prefix: str = typer.Argument("", help="Prefix to list assets under"),
):
    """List available assets under a prefix in the bucket."""
    source = ctx.obj["source"]

    safe_echo(f"Listing assets in bucket: {source} under prefix: {prefix or '(root)'}", ctx)

    from arena_models.impl.fetch import Bucket

    assets = Bucket(source).list_assets(prefix)
    for asset_dir in assets:
        print(asset_dir)

    safe_echo(f"Found {len(assets)} asset(s).", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add list command to the network command group."""
    cmd.command("list")(list_command)


if __name__ == "__main__":
    typer.run(list_command)
