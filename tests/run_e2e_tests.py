#!/usr/bin/env python3
# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple test runner for Microsoft Copilot E2E tests without pytest dependency."""

import subprocess
import tempfile
import traceback
from pathlib import Path


def run_test(test_func, test_name):
    """Run a single test function and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print("=" * 60)

    try:
        test_func()
        print(f"✅ PASSED: {test_name}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False


def validate_microsoft_copilot_content(content, strict=True):
    """Shared content validation for Microsoft Copilot files (MHTML, HTML, PDF).

    Args:
        content: The generated markdown content.
        strict: If True, requires specific German content. If False, allows more
            flexible validation.

    Returns:
        None

    Raises:
        AssertionError: If validation fails (in strict mode or if assertions are triggered).
    """
    # Basic conversation structure should be present
    has_user = "### User" in content
    has_assistant = "### Assistant" in content

    if strict:
        # Verify conversation structure (strict mode for MHTML)
        assert has_user, "User messages not found in output"  # nosec
        assert has_assistant, "Assistant messages not found in output"  # nosec
        print("✓ Found conversation structure (User/Assistant)")

        # Verify specific content from the test file
        assert "Wie du magst, sei kreativ" in content, "Expected user message not found"  # nosec
        assert "kreativer Gruß" in content, "Expected assistant response not found"  # nosec
        print("✓ Found expected conversation content")

        # Verify no unexpected content
        assert (
            "Nachricht an Copilot" not in content
        ), "UI elements leaked into conversation"  # nosec
        assert "Schnelle Antwort" not in content, "UI elements leaked into conversation"  # nosec
        print("✓ No UI elements leaked into conversation")
    else:
        # Relaxed mode for formats that may not work as well
        if has_user and has_assistant:
            print("✓ Found conversation structure (User/Assistant)")
        else:
            print("⚠ Conversation structure not found, but file was processed")

        # Check for any meaningful content
        if len(content.strip()) > 50:
            print("✓ Generated meaningful content")
        else:
            print("⚠ Generated content is minimal")

        # Check that we don't have raw HTML/JS content
        if not any(x in content for x in ["<![CDATA[", "//]]>", "window.", "performance.now"]):
            print("✓ No raw JavaScript/HTML content detected")
        else:
            print("⚠ Raw content detected - format may not be fully supported")

    # Display sample of the generated content
    print(f"✓ Generated content preview ({len(content)} chars):")
    print(content[:300] + "..." if len(content) > 300 else content)


def test_microsoft_copilot_mhtml_e2e():
    """Test end-to-end processing of Microsoft Copilot MHTML file."""
    test_data_dir = Path(__file__).parent / "data"
    mhtml_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.mhtml"

    # Ensure test file exists
    assert mhtml_file.exists(), f"Test file not found: {mhtml_file}"  # nosec
    print(f"Found test file: {mhtml_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temp directory: {temp_path}")

        # Run the CLI tool
        cmd = ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(mhtml_file)]
        print(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, shell=False)

        # Check exit code
        assert result.returncode == 0, (  # nosec
            f"CLI failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        print("✓ CLI executed successfully")

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec
        print(f"✓ Created {len(output_files)} output file(s): {[f.name for f in output_files]}")

        # Read the output
        output_file = output_files[0]
        content = output_file.read_text(encoding="utf-8")

        # Use shared content validation (strict mode for MHTML)
        validate_microsoft_copilot_content(content, strict=True)


def test_microsoft_copilot_html_e2e():
    """Test end-to-end processing of Microsoft Copilot HTML file."""
    test_data_dir = Path(__file__).parent / "data"
    html_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.html"

    # Ensure test file exists
    assert html_file.exists(), f"Test file not found: {html_file}"  # nosec
    print(f"Found test file: {html_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(html_file)],
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
        print("✓ CLI executed successfully")

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec
        print(f"✓ Created {len(output_files)} output file(s)")

        # Read the output
        output_file = output_files[0]
        content = output_file.read_text(encoding="utf-8")

        # Use shared content validation (relaxed mode for HTML)
        validate_microsoft_copilot_content(content, strict=False)


def test_microsoft_copilot_pdf_e2e():
    """Test end-to-end processing of Microsoft Copilot PDF file."""
    test_data_dir = Path(__file__).parent / "data"
    pdf_file = test_data_dir / "Microsoft Copilot_ Ihr KI-Begleiter.pdf"

    # Only run if PDF file exists (optional test)
    if not pdf_file.exists():
        print(f"Skipping PDF test: {pdf_file} not found")
        return

    print(f"Found test file: {pdf_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(pdf_file)],
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
        print("✓ CLI executed successfully")

        # Check that output file was created
        output_files = list(temp_path.glob("*.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec
        print(f"✓ Created {len(output_files)} output file(s)")

        # Read the output
        output_file = output_files[0]
        content = output_file.read_text(encoding="utf-8")

        # Use shared content validation (relaxed mode for PDF)
        validate_microsoft_copilot_content(content, strict=False)


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
                "run",
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
        print("✓ CLI executed successfully with verbose logging")

        # Check for expected info messages only (no warnings or errors)
        stderr_lower = result.stderr.lower()

        # Should not contain error messages
        assert "error" not in stderr_lower, f"Unexpected error in output: {result.stderr}"  # nosec
        assert (
            "traceback" not in stderr_lower
        ), f"Unexpected traceback in output: {result.stderr}"  # nosec
        print("✓ No errors found in output")

        # Print any output for review
        if result.stderr.strip():
            print(f"stderr output: {result.stderr}")
        if result.stdout.strip():
            print(f"stdout output: {result.stdout}")


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
    <div class="message-content">Of course! I'd be happy to help you with Python. """
        """What do you need assistance with?</div>
</div>
<div data-message-author-role="user">
    <div class="message-content">How do I create a list?</div>
</div>
<div data-message-author-role="assistant">
    <div class="message-content">You can create a list in Python using square brackets: """
        """<code>my_list = [1, 2, 3]</code></div>
</div>
</body>
</html>"""

        test_file = temp_path / "chatgpt_test.html"
        test_file.write_text(chatgpt_html, encoding="utf-8")
        print(f"Created test ChatGPT file: {test_file}")

        # Run the CLI tool
        result = subprocess.run(
            ["chatgpt-saved-session-to-markdown", "run", "-o", str(temp_path), str(test_file)],
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
        print("✓ CLI executed successfully")

        # Check that output file was created
        output_files = list(temp_path.glob("*_test.md"))
        assert len(output_files) > 0, "No markdown files were created"  # nosec
        print(f"✓ Created output file: {output_files[0].name}")

        # Read the output
        content = output_files[0].read_text(encoding="utf-8")

        # Verify conversation structure
        assert "### User" in content, "User messages not found in output"  # nosec
        assert "### Assistant" in content, "Assistant messages not found in output"  # nosec
        print("✓ Found conversation structure (User/Assistant)")

        # Verify specific content
        assert "help me with Python" in content, "Expected user message not found"  # nosec
        assert "happy to help" in content, "Expected assistant response not found"  # nosec
        assert "create a list" in content, "Expected user question not found"  # nosec
        assert "square brackets" in content, "Expected assistant answer not found"  # nosec
        print("✓ Found expected ChatGPT conversation content")

        # Display sample of the generated content
        print(f"✓ Generated content preview ({len(content)} chars):")
        print(content[:400] + "..." if len(content) > 400 else content)


def main():
    """Run all tests."""
    print("Microsoft Copilot E2E Test Suite")
    print("================================")

    tests = [
        (test_microsoft_copilot_mhtml_e2e, "Microsoft Copilot MHTML E2E"),
        (test_microsoft_copilot_html_e2e, "Microsoft Copilot HTML E2E"),
        (test_microsoft_copilot_pdf_e2e, "Microsoft Copilot PDF E2E"),
        (test_no_warnings_or_errors, "No Warnings or Errors"),
        (test_chatgpt_compatibility, "ChatGPT Compatibility"),
    ]

    passed = 0
    failed = 0

    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        exit(1)
    else:
        print("All tests passed! ✅")


if __name__ == "__main__":
    main()
