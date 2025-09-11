# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command-line interface for chatgpt-saved-session-to-markdown."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich import print

from . import __version__
from .extractor import process_many

app = typer.Typer(
    add_completion=False,
    help="Convert ChatGPT HTML/MHTML/PDF exports to Markdown (embeds inline resources, warns on better formats).",
)


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@app.callback()
def _main(
    ctx: typer.Context,
    outdir: Path | None = typer.Option(
        None, "--outdir", "-o", help="Directory for output .md files."
    ),
    jobs: int | None = typer.Option(
        None, "--jobs", "-j", help="Parallel workers (default: CPU count/auto)."
    ),
    verbose: int = typer.Option(
        0, "-v", count=True, help="Increase verbosity (-v: INFO, -vv: DEBUG)."
    ),
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """Global options; if no subcommand is given, print help and exit 0."""
    if version:
        typer.echo(__version__)
        raise typer.Exit(code=0)
    _setup_logging(verbose)
    # If run without subcommand: show help (exit 0 as a smoke test behavior for CI).
    if not ctx.invoked_subcommand:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@app.command("run")
def run(
    inputs: list[str] = typer.Argument(
        ..., help="Inputs (.html, .htm, .mhtml, .mht, .pdf). Shell globs allowed."
    ),
    outdir: Path | None = typer.Option(
        None, "--outdir", "-o", help="Directory for output .md files."
    ),
    jobs: int | None = typer.Option(
        None, "--jobs", "-j", help="Parallel workers (default: CPU count/auto)."
    ),
    verbose: int = typer.Option(
        0, "-v", count=True, help="Increase verbosity (-v: INFO, -vv: DEBUG)."
    ),
) -> None:
    """Convert the given inputs to Markdown; exit non-zero if nothing produced."""
    _setup_logging(verbose)
    try:
        produced = process_many(inputs, outdir, (jobs or 0))
    except Exception as exc:  # surface clear non-zero on any batch failure
        logging.error("%s", exc)
        raise typer.Exit(code=1)
    if not produced:
        logging.error("No outputs were produced.")
        raise typer.Exit(code=1)
    for p in produced:
        print(str(p))


if __name__ == "__main__":
    app()
