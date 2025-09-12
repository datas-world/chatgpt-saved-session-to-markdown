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

LOGGER = logging.getLogger(__name__)


class _WarningTracker(logging.Handler):
    """Handler to track if any warnings were logged."""
    
    def __init__(self) -> None:
        super().__init__()
        self.warnings_shown = False
    
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.warnings_shown = True


# Global warning tracker instance
_warning_tracker = _WarningTracker()


def _setup_logging(verbose: int, quiet: int, log_level: str | None) -> int:
    """Configure logging level based on verbosity/quiet count or explicit log level.
    
    Args:
        verbose: Number of -v/--verbose flags (increases verbosity)
        quiet: Number of -q/--quiet flags (decreases verbosity)
        log_level: Explicit log level name or integer string
        
    Returns:
        The final log level that was set
    """
    # Start with explicit log level or Python default (WARNING)
    if log_level is not None:
        base_level = _parse_log_level(log_level)
    else:
        base_level = logging.WARNING  # Python default
    
    # Calculate net verbosity (verbose flags minus quiet flags)
    net_verbosity = verbose - quiet
    
    # Apply verbosity adjustment with 10-point steps (like predefined levels)
    level = base_level - (net_verbosity * 10)
    
    # Ensure level is non-negative
    level = max(0, level)
    
    logging.basicConfig(
        level=level, 
        format="%(levelname)s: %(message)s",
        force=True  # Override any existing configuration
    )
    
    # Add warning tracker to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(_warning_tracker)
    
    # Log the final log level at INFO level
    logger = logging.getLogger(__name__)
    logger.info("Log level set to %d (%s)", level, logging.getLevelName(level))
    
    return level


def _parse_log_level(log_level_str: str) -> int:
    """Parse log level from string (name or integer).
    
    Args:
        log_level_str: Log level as string (name like 'DEBUG' or integer like '10')
        
    Returns:
        Log level as integer
        
    Raises:
        ValueError: If log level is invalid
    """
    # Try parsing as integer first
    try:
        level_int = int(log_level_str)
        if level_int < 0:
            raise ValueError("Log level must be non-negative")
        return level_int
    except ValueError:
        pass
    
    # Try parsing as log level name
    level_name = log_level_str.upper()
    level_map = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR, 
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
    }
    
    if level_name in level_map:
        return level_map[level_name]
    
    raise ValueError(
        f"Invalid log level '{log_level_str}'. "
        f"Use log level names (CRITICAL, ERROR, WARNING, INFO, DEBUG) "
        f"or non-negative integers."
    )


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
        help="Increase verbosity (-v: INFO, -vv: DEBUG). Can be applied multiple times.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease verbosity (-q: ERROR, -qq: CRITICAL). Can be applied multiple times.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        help="Set log level explicitly (CRITICAL, ERROR, WARNING, INFO, DEBUG) or non-negative integer.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()

    try:
        log_level = _setup_logging(args.verbose, args.quiet, args.log_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # If no files are given, exit successfully  
    if not args.inputs:
        LOGGER.debug("No input files provided, exiting successfully")
        sys.exit(0)

    try:
        LOGGER.info("Starting processing of %d input pattern(s): %s", 
                   len(args.inputs), ", ".join(args.inputs))
        produced = process_many(args.inputs, args.outdir, (args.jobs or 0))
    except Exception as exc:  # surface clear non-zero on any batch failure
        LOGGER.critical("Processing failed: %s", exc, exc_info=exc)
        sys.exit(1)

    if not produced:
        LOGGER.critical("No outputs were produced")
        sys.exit(1)

    LOGGER.info("Successfully processed %d input(s), produced %d output file(s)", 
               len(args.inputs), len(produced))
    for p in produced:
        LOGGER.info("Output: %s", str(p))
    
    # Show debug level suggestion if warnings were shown and log level is more quiet than DEBUG
    if _warning_tracker.warnings_shown and log_level > logging.DEBUG:
        LOGGER.info(
            "Issues detected during processing. For detailed diagnostics, rerun with DEBUG level: "
            "add -vv or -l DEBUG. To report issues: "
            "https://github.com/datas-world/chatgpt-saved-session-to-markdown/issues/new"
        )


def app() -> None:
    """Typer compatibility function."""
    main()


if __name__ == "__main__":
    main()
