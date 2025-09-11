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
    assert mhtml_file.exists(), f"Test file not found: {mhtml_file}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(mhtml_file)],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"CLI failed with exit code {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        
        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"
        
        # Read the output
        output_file = output_files[0]
        content = output_file.read_text(encoding="utf-8")
        
        # Verify conversation structure
        assert "### User" in content, "User messages not found in output"
        assert "### Assistant" in content, "Assistant messages not found in output"
        
        # Verify specific content from the test file
        assert "Wie du magst, sei kreativ" in content, "Expected user message not found"
        assert "kreativer Gruß" in content, "Expected assistant response not found"
        
        # Verify no unexpected content
        assert "Nachricht an Copilot" not in content, "UI elements leaked into conversation"
        assert "Schnelle Antwort" not in content, "UI elements leaked into conversation"
        
        # Verify clean markdown structure
        lines = content.split('\n')
        user_lines = [i for i, line in enumerate(lines) if line.strip() == "### User"]
        assistant_lines = [i for i, line in enumerate(lines) if line.strip() == "### Assistant"]
        
        assert len(user_lines) > 0, "No user message headers found"
        assert len(assistant_lines) > 0, "No assistant message headers found"


def test_microsoft_copilot_html_e2e():
    """Test end-to-end processing of Microsoft Copilot HTML file."""
    test_data_dir = Path(__file__).parent / "data"
    html_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.html"
    
    # Ensure test file exists
    assert html_file.exists(), f"Test file not found: {html_file}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(html_file)],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"CLI failed with exit code {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        
        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"


def test_microsoft_copilot_pdf_e2e():
    """Test end-to-end processing of Microsoft Copilot PDF file."""
    test_data_dir = Path(__file__).parent / "data"
    pdf_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.pdf"
    
    # Ensure test file exists
    assert pdf_file.exists(), f"Test file not found: {pdf_file}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(pdf_file)],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"CLI failed with exit code {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        
        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"


def test_no_warnings_or_errors():
    """Test that processing Microsoft Copilot files generates no unexpected warnings or errors."""
    test_data_dir = Path(__file__).parent / "data"
    mhtml_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.mhtml"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run with verbose output to capture any warnings
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-vv", "-o", str(temp_path), str(mhtml_file)],
            capture_output=True,
            text=True
        )
        
        # Should succeed without errors
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Check for expected info messages only (no warnings or errors)
        stderr_lower = result.stderr.lower()
        stdout_lower = result.stdout.lower()
        
        # Should not contain error messages
        assert "error" not in stderr_lower, f"Unexpected error in output: {result.stderr}"
        assert "traceback" not in stderr_lower, f"Unexpected traceback in output: {result.stderr}"
        
        # Warnings about format comparison are acceptable, but not errors
        if "warning" in stderr_lower:
            # Only allow expected warnings about format comparison
            assert any(expected in stderr_lower for expected in [
                "both html and mhtml present",
                "pdf provided alongside"
            ]), f"Unexpected warning in output: {result.stderr}"


def test_chatgpt_compatibility():
    """Test that ChatGPT-style HTML still works after Microsoft Copilot changes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple ChatGPT-style HTML file
        chatgpt_html = """<!DOCTYPE html>
<html>
<head><title>ChatGPT Conversation</title></head>
<body>
<div data-message-author-role="user">
    <div class="message-content">Hello, can you help me with Python?</div>
</div>
<div data-message-author-role="assistant">
    <div class="message-content">Of course! I'd be happy to help you with Python. What do you need assistance with?</div>
</div>
<div data-message-author-role="user">
    <div class="message-content">How do I create a list?</div>
</div>
<div data-message-author-role="assistant">
    <div class="message-content">You can create a list in Python using square brackets: <code>my_list = [1, 2, 3]</code></div>
</div>
</body>
</html>"""
        
        test_file = temp_path / "chatgpt_test.html"
        test_file.write_text(chatgpt_html, encoding="utf-8")
        
        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        assert result.returncode == 0, f"CLI failed with exit code {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        
        # Check that output file was created
        output_files = list(temp_path.glob("*_test.md"))
        assert len(output_files) > 0, "No markdown files were created"
        
        # Read the output
        content = output_files[0].read_text(encoding="utf-8")
        
        # Verify conversation structure
        assert "### User" in content, "User messages not found in output"
        assert "### Assistant" in content, "Assistant messages not found in output"
        
        # Verify specific content
        assert "help me with Python" in content, "Expected user message not found"
        assert "happy to help" in content, "Expected assistant response not found"
        assert "create a list" in content, "Expected user question not found"
        assert "square brackets" in content, "Expected assistant answer not found"


if __name__ == "__main__":
    pytest.main([__file__])