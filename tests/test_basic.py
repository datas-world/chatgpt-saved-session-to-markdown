# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later
"""Basic tests for the chat_export_md package."""

import pytest

from chat_export_md import __version__


def test_version() -> None:
    """Test that version is defined."""
    assert __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_import() -> None:
    """Test that the package can be imported."""
    import chat_export_md  # noqa: F401
    
    assert True  # If we get here, import succeeded