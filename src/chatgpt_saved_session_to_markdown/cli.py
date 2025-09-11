# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command-line interface for chatgpt-saved-session-to-markdown."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .extractor import process_many


def _setup_logging(verbose: int) -> None:
    """Configure logging level based on verbosity count."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> None:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert ChatGPT HTML/MHTML/PDF exports to Markdown "
        "(embeds inline resources, warns on better formats)."
    )

    parser.add_argument(
        "inputs", nargs="*", help="Inputs (.html, .htm, .mhtml, .mht, .pdf). Shell globs allowed."
    )
    parser.add_argument("--outdir", "-o", type=Path, help="Directory for output .md files.")
    parser.add_argument(
        "--jobs", "-j", type=int, help="Parallel workers (default: CPU count/auto)."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v: INFO, -vv: DEBUG).",
    )
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()

    _setup_logging(args.verbose)

    # If no files are given, exit successfully
    if not args.inputs:
        sys.exit(0)

    try:
        produced = process_many(args.inputs, args.outdir, (args.jobs or 0))
    except Exception as exc:  # surface clear non-zero on any batch failure
        logging.error("%s", exc)
        sys.exit(1)

    if not produced:
        logging.error("No outputs were produced.")
        sys.exit(1)

    for p in produced:
        print(str(p))


def app() -> None:
    """Typer compatibility function."""
    main()


if __name__ == "__main__":
    main()
