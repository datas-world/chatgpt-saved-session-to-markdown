# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core extraction functionality for converting ChatGPT exports to Markdown."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup, Tag
import PyPDF2
from io import StringIO


def extract_to_markdown(file_path: Path) -> str:
    """Extract content from ChatGPT export files and convert to Markdown.
    
    Args:
        file_path: Path to the input file (HTML, MHTML, or PDF)
        
    Returns:
        Markdown-formatted string of the conversation
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.html', '.htm']:
        return _extract_from_html(file_path)
    elif suffix == '.mhtml':
        return _extract_from_mhtml(file_path)
    elif suffix == '.pdf':
        return _extract_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _extract_from_html(file_path: Path) -> str:
    """Extract content from HTML files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'lxml')
    return _parse_chatgpt_html(soup, file_path)


def _extract_from_mhtml(file_path: Path) -> str:
    """Extract content from MHTML files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract HTML content from MHTML
    html_match = re.search(r'Content-Type: text/html.*?\n\n(.*?)(?=\n--)', content, re.DOTALL)
    if not html_match:
        raise ValueError("Could not extract HTML content from MHTML file")
    
    html_content = html_match.group(1)
    soup = BeautifulSoup(html_content, 'lxml')
    return _parse_chatgpt_html(soup, file_path)


def _extract_from_pdf(file_path: Path) -> str:
    """Extract content from PDF files (best effort)."""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text_content = []
            
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            full_text = '\n'.join(text_content)
            return _parse_pdf_text(full_text, file_path)
    
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {e}")


def _parse_chatgpt_html(soup: BeautifulSoup, file_path: Path) -> str:
    """Parse ChatGPT HTML content and extract conversation."""
    messages = []
    
    # Try different selectors for ChatGPT conversation elements
    conversation_selectors = [
        '[data-testid*="conversation"]',
        '.conversation',
        '[class*="conversation"]',
        '.message',
        '[class*="message"]'
    ]
    
    conversation_found = False
    
    for selector in conversation_selectors:
        elements = soup.select(selector)
        if elements:
            conversation_found = True
            for element in elements:
                message = _extract_message_from_element(element)
                if message:
                    messages.append(message)
            break
    
    if not conversation_found:
        # Fallback: extract all text and try to parse it
        all_text = soup.get_text()
        return _parse_fallback_text(all_text, file_path)
    
    return _format_messages_as_markdown(messages, file_path)


def _extract_message_from_element(element: Tag) -> Optional[Dict[str, str]]:
    """Extract message information from HTML element."""
    text = element.get_text(strip=True)
    if not text:
        return None
    
    # Try to detect role based on common patterns
    role = "User"  # Default assumption
    
    # Look for indicators of assistant messages
    assistant_indicators = ['gpt', 'assistant', 'ai', 'chatgpt']
    user_indicators = ['user', 'you', 'human']
    
    element_str = str(element).lower()
    
    for indicator in assistant_indicators:
        if indicator in element_str:
            role = "Assistant"
            break
    
    for indicator in user_indicators:
        if indicator in element_str:
            role = "User"
            break
    
    return {
        "role": role,
        "content": text
    }


def _parse_pdf_text(text: str, file_path: Path) -> str:
    """Parse extracted PDF text into conversation format."""
    lines = text.split('\n')
    messages = []
    current_message = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to detect message boundaries
        if any(indicator in line.lower() for indicator in ['user:', 'you:', 'human:']):
            if current_message:
                messages.append(current_message)
            current_message = {"role": "User", "content": line}
        elif any(indicator in line.lower() for indicator in ['assistant:', 'gpt:', 'ai:']):
            if current_message:
                messages.append(current_message)
            current_message = {"role": "Assistant", "content": line}
        elif current_message:
            current_message["content"] += f"\n{line}"
        else:
            # Start with user if no clear indicator
            current_message = {"role": "User", "content": line}
    
    if current_message:
        messages.append(current_message)
    
    return _format_messages_as_markdown(messages, file_path)


def _parse_fallback_text(text: str, file_path: Path) -> str:
    """Fallback parser when structured extraction fails."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Simple heuristic: alternate between user and assistant
    messages = []
    for i, line in enumerate(lines):
        role = "User" if i % 2 == 0 else "Assistant"
        messages.append({"role": role, "content": line})
    
    return _format_messages_as_markdown(messages, file_path)


def _format_messages_as_markdown(messages: List[Dict[str, str]], file_path: Path) -> str:
    """Format extracted messages as Markdown."""
    if not messages:
        return f"# Conversation from {file_path.name}\n\n*No messages found*\n"
    
    output = StringIO()
    output.write(f"# Conversation from {file_path.name}\n\n")
    
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        
        if not content:
            continue
        
        # Format as Markdown with role headers
        output.write(f"## {role}\n\n")
        output.write(f"{content}\n\n")
        
        # Add separator between messages (except last)
        if i < len(messages) - 1:
            output.write("---\n\n")
    
    return output.getvalue()
