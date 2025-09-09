# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the extractor module."""

import pytest
import tempfile
from pathlib import Path
from chat_export_md.extractor import extract_to_markdown, _format_messages_as_markdown


def test_unsupported_file_format() -> None:
    """Test that unsupported file formats raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_to_markdown(tmp_path)
    finally:
        tmp_path.unlink()


def test_file_not_found() -> None:
    """Test that missing files raise FileNotFoundError."""
    non_existent = Path("non_existent_file.html")
    with pytest.raises(FileNotFoundError):
        extract_to_markdown(non_existent)


def test_format_messages_as_markdown() -> None:
    """Test markdown formatting of messages."""
    messages = [
        {"role": "User", "content": "Hello, how are you?"},
        {"role": "Assistant", "content": "I'm doing well, thank you!"},
    ]
    
    result = _format_messages_as_markdown(messages, Path("test.html"))
    
    assert "# Conversation from test.html" in result
    assert "## User" in result
    assert "## Assistant" in result
    assert "Hello, how are you?" in result
    assert "I'm doing well, thank you!" in result


def test_empty_messages() -> None:
    """Test handling of empty message list."""
    result = _format_messages_as_markdown([], Path("test.html"))
    
    assert "# Conversation from test.html" in result
    assert "*No messages found*" in result