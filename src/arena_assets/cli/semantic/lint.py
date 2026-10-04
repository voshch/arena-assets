"""Lint command for ASA annotation.yaml files."""

import json
from collections import Counter
from pathlib import Path

import typer

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.semantic.fixes import fix_file
from arena_assets.semantic.lint import Finding, lint_file
from arena_assets.semantic.taxonomy import load_spec

from ..utils import safe_echo


def _discover(path: Path) -> list[Path]:
    if path.name == ANNOTATION_NAME:
        return [path]
    return sorted(path.rglob(ANNOTATION_NAME))


def _finding_dict(finding: Finding) -> dict:
    return {
        "file": str(finding.file),
        "severity": finding.severity,
        "code": finding.code,
        "message": finding.message,
        "fixable": finding.fixable,
    }


def lint_command(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ...,
        help="annotation.yaml file, or a directory tree to scan for them",
        exists=True,
    ),
    fix: bool = typer.Option(False, "--fix", help="Apply mechanical ASA migration fixes before reporting"),
    json_output: bool = typer.Option(False, "--json", help="Print findings as a JSON array"),
):
    """Lint annotation.yaml files against the ASA spec."""
    spec = load_spec()
    files = _discover(path)

    if fix:
        for file in files:
            fix_file(file, spec)

    findings: list[Finding] = []
    for file in files:
        findings.extend(lint_file(file, spec))

    if json_output:
        typer.echo(json.dumps([_finding_dict(finding) for finding in findings]))
    else:
        for finding in findings:
            typer.echo(f"{finding.severity} {finding.file} {finding.code} {finding.message}")
        counts = Counter(finding.severity for finding in findings)
        safe_echo(
            f"{len(findings)} findings ({counts.get('error', 0)} errors, {counts.get('warning', 0)} warnings, {counts.get('info', 0)} info)",
            ctx,
        )

    if any(finding.severity in ("error", "warning") for finding in findings):
        raise typer.Exit(1)


def add_to_cmd(semantic_cmd: typer.Typer):
    semantic_cmd.command("lint")(lint_command)


if __name__ == "__main__":
    typer.run(lint_command)
