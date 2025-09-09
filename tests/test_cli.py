# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the CLI module."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from chat_export_md.cli import app


def test_version_command() -> None:
    """Test the version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "chatgpt-saved-session-to-markdown" in result.stdout


def test_headers_fix_command() -> None:
    """Test the headers-fix command."""
    runner = CliRunner()
    result = runner.invoke(app, ["headers-fix"])
    assert result.exit_code == 0
    assert "not implemented" in result.stdout


def test_help() -> None:
    """Test that help is displayed when no arguments given."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Convert ChatGPT" in result.stdout