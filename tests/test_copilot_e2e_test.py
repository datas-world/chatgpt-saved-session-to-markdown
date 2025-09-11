# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end tests for Microsoft Copilot support."""

import subprocess
import tempfile
from pathlib import Path

import pytest


def test_microsoft_copilot_mhtml_e2e():
    """Test end-to-end processing of Microsoft Copilot MHTML file."""
    test_data_dir = Path(__file__).parent / "data"
    mhtml_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.mhtml"

    # Ensure test file exists
    assert mhtml_file.exists(), f"Test file not found: {mhtml_file}"  # nosec

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "-o", str(temp_path), str(mhtml_file)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        # Check exit code
        assert result.returncode == 0, (  # nosec
            f"CLI failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec

        # Read the output
        output_file = output_files[0]
        content = output_file.read_text()

        # Verify conversation structure
        assert "## User" in content, "User messages not found in output"  # nosec
        assert "## Assistant" in content, "Assistant messages not found in output"  # nosec

        # Verify specific content from the test file
        assert "Wie du magst, sei kreativ" in content, "Expected user message not found"  # nosec
        assert "kreativer Gruß" in content, "Expected assistant response not found"  # nosec

        # Verify no unexpected content
        assert (
            "Nachricht an Copilot" not in content
        ), "UI elements leaked into conversation"  # nosec
        assert "Schnelle Antwort" not in content, "UI elements leaked into conversation"  # nosec

        # Verify clean markdown structure
        lines = content.split("\n")
        user_lines = [i for i, line in enumerate(lines) if line.strip() == "## User"]
        assistant_lines = [i for i, line in enumerate(lines) if line.strip() == "## Assistant"]

        assert len(user_lines) > 0, "No user message headers found"  # nosec
        assert len(assistant_lines) > 0, "No assistant message headers found"  # nosec


def test_microsoft_copilot_html_e2e():
    """Test end-to-end processing of Microsoft Copilot HTML file."""
    test_data_dir = Path(__file__).parent / "data"
    html_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.html"

    # Ensure test file exists
    assert html_file.exists(), f"Test file not found: {html_file}"  # nosec

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "-o", str(temp_path), str(html_file)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        # Check exit code
        assert result.returncode == 0, (  # nosec
            f"CLI failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec


def test_microsoft_copilot_pdf_e2e():
    """Test end-to-end processing of Microsoft Copilot PDF file."""
    test_data_dir = Path(__file__).parent / "data"
    pdf_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.pdf"

    # Ensure test file exists
    assert pdf_file.exists(), f"Test file not found: {pdf_file}"  # nosec

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "-o", str(temp_path), str(pdf_file)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        # Check exit code
        assert result.returncode == 0, (  # nosec
            f"CLI failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec


def test_no_warnings_or_errors():
    """Test that processing Microsoft Copilot files generates no unexpected warnings or errors."""
    test_data_dir = Path(__file__).parent / "data"
    mhtml_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.mhtml"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run with verbose output to capture any warnings
        result = subprocess.run(
            [
                "chatgpt-saved-session-to-markdown",
                "-vv",
                "-o",
                str(temp_path),
                str(mhtml_file),
            ],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        # Should succeed without errors
        assert result.returncode == 0, f"CLI failed: {result.stderr}"  # nosec

        # Check for expected info messages only (no warnings or errors)
        stderr_lower = result.stderr.lower()

        # Should not contain error messages
        assert "error" not in stderr_lower, f"Unexpected error in output: {result.stderr}"  # nosec
        assert (
            "traceback" not in stderr_lower
        ), f"Unexpected traceback in output: {result.stderr}"  # nosec

        # Warnings about format comparison are acceptable, but not errors
        if "warning" in stderr_lower:
            # Only allow expected warnings about format comparison
            assert any(  # nosec
                expected in stderr_lower
                for expected in ["both html and mhtml present", "pdf provided alongside"]
            ), f"Unexpected warning in output: {result.stderr}"


def test_chatgpt_compatibility():
    """Test that ChatGPT-style HTML still works after Microsoft Copilot changes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a simple ChatGPT-style HTML file
        chatgpt_html = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><title>ChatGPT Conversation</title></head>\n"
            "<body>\n"
            '<div data-message-author-role="user">\n'
            '    <div class="message-content">Hello, can you help me with Python?</div>\n'
            "</div>\n"
            '<div data-message-author-role="assistant">\n'
            '    <div class="message-content">Of course! I\'d be happy to help you with Python. '
            "What do you need assistance with?</div>\n"
            "</div>\n"
            '<div data-message-author-role="user">\n'
            '    <div class="message-content">How do I create a list?</div>\n'
            "</div>\n"
            '<div data-message-author-role="assistant">\n'
            '    <div class="message-content">You can create a list in Python using '
            "square brackets: <code>my_list = [1, 2, 3]</code></div>\n"
            "</div>\n"
            "</body>\n"
            "</html>"
        )

        test_file = temp_path / "chatgpt_test.html"
        test_file.write_text(chatgpt_html)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "-o", str(temp_path), str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        # Check exit code
        assert result.returncode == 0, (  # nosec
            f"CLI failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec

        # Read the output
        content = output_files[0].read_text()

        # Verify conversation structure
        assert "## User" in content, "User messages not found in output"  # nosec
        assert "## Assistant" in content, "Assistant messages not found in output"  # nosec

        # Verify specific content
        assert "help me with Python" in content, "Expected user message not found"  # nosec
        assert "happy to help" in content, "Expected assistant response not found"  # nosec
        assert "create a list" in content, "Expected user question not found"  # nosec
        assert "square brackets" in content, "Expected assistant answer not found"  # nosec


if __name__ == "__main__":
    pytest.main([__file__])
