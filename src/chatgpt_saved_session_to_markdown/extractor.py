# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""HTML/MHTML/PDF to Markdown extractor with role detection and warnings."""

from __future__ import annotations

import base64
import codecs
import contextlib
import glob
import locale
import logging
import os
import quopri
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as _md

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Heuristic quality signals & warnings                                        #
# --------------------------------------------------------------------------- #


def _warn_better_format_guess_for_html(html: str, path: Path) -> None:
    """Warn if HTML likely loses embeds vs. MHTML."""
    role_markers = len(re.findall(r'data-message-author-role=(["\'])', html))
    img_http = len(re.findall(r'<img[^>]+src=["\']https?://', html, flags=re.I))
    cid_refs = len(re.findall(r'src=["\']cid:', html, flags=re.I))
    if cid_refs > 0:
        LOGGER.warning(
            "%s: HTML references cid: resources; an MHTML export typically "
            "embeds those. Prefer MHTML if available.",
            path,
        )
    elif role_markers == 0 and img_http >= 5:
        LOGGER.warning(
            "%s: HTML has many external images but no clear chat role markers. "
            "An MHTML export often preserves inline assets better. "
            "Consider MHTML if available.",
            path,
        )


def _warn_better_format_guess_for_mhtml(
    html_parts: list[str], resources: dict[str, tuple[str, bytes]], path: Path
) -> None:
    """Warn if MHTML looks incomplete vs. HTML."""
    combined = "\n".join(html_parts)
    cid_refs = re.findall(r'(?:src|href)=["\'](cid:[^"\']+)["\']', combined, flags=re.I)
    resolved = sum(1 for c in cid_refs if c in resources)
    missing = len(cid_refs) - resolved
    if len(html_parts) == 1 and resolved == 0 and len(combined) < 20_000:
        LOGGER.warning(
            "%s: MHTML contains no resolved inline resources and limited text. "
            "An HTML export may yield richer content. Prefer HTML if available.",
            path,
        )
    elif missing > 0:
        LOGGER.warning(
            "%s: Some MHTML inline resources referenced by cid: were not found. "
            "If possible, try the HTML export as well.",
            path,
        )


def _warn_better_format_guess_for_pdf(pages_extracted: int, text_len: int, path: Path) -> None:
    """Warn that PDF is less preferred than HTML/MHTML."""
    LOGGER.warning(
        "%s: PDF text extraction is best-effort and loses structure. "
        "Prefer HTML or MHTML exports whenever available.",
        path,
    )


# --------------------------------------------------------------------------- #
# Content-Transfer-Encoding handling per RFC 1341                             #
# --------------------------------------------------------------------------- #


def _decode_content_transfer_encoding(payload: bytes, encoding: str | None) -> bytes:
    """Decode Content-Transfer-Encoding values as defined in RFC 1341.

    See: https://tools.ietf.org/html/rfc1341#section-5

    This implementation checks for encodings in the following order:
    quoted-printable > base64 > binary > 8bit > 7bit.
    (Note: This priority order is an implementation choice, not specified by RFC 1341.
     See: https://tools.ietf.org/html/rfc1341#section-5.1)
    Uses COTS packages (standard library base64, quopri) with proper validation.

    Args:
        payload: Raw payload bytes
        encoding: Content-Transfer-Encoding value (case-insensitive)

    Returns:
        Decoded payload bytes

    Raises:
        RuntimeError: If decoding fails for supported encodings
    """
    if not encoding:
        # Default to 7bit if no encoding specified (RFC 1341 default)
        # See: https://tools.ietf.org/html/rfc1341#section-5.1
        return payload

    encoding_lower = encoding.strip().lower()

    # Handle priority order - quoted-printable has highest priority
    if encoding_lower == "quoted-printable":
        try:
            # Use quopri module for quoted-printable (COTS package)
            return quopri.decodestring(payload)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode quoted-printable content: {exc}") from exc

    # base64 has second priority
    elif encoding_lower == "base64":
        try:
            # Use base64 module with validation (COTS package)
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode base64 content: {exc}") from exc

    # binary, 8bit, 7bit have lower priorities but no decoding needed
    elif encoding_lower in ("binary", "8bit", "7bit"):
        # No decoding needed for these encodings
        return payload

    else:
        # Unknown encoding - raise error for strict validation
        raise RuntimeError(
            f"Unknown Content-Transfer-Encoding '{encoding}'. "
            f"Strict validation requires known encoding."
        )


def _get_system_encoding() -> str:
    """Get the system's preferred encoding for text files.

    Returns the locale encoding, falling back to US-ASCII per RFC 2822/5322 standards
    if locale detection fails.

    See: https://tools.ietf.org/html/rfc2822#section-2.2
         https://tools.ietf.org/html/rfc5322#section-2.2
    """
    try:
        encoding = locale.getpreferredencoding(False)
        if encoding:
            return encoding
    except (AttributeError, LookupError):
        pass
    # Fallback to US-ASCII per RFC standards
    return "us-ascii"


def _get_email_charset_or_error(message: Message, context: str) -> str:
    """Get charset from email message or raise error if not available.

    Per RFC 2822/5322, if no charset is specified for text/* content,
    US-ASCII should be assumed.

    See: https://tools.ietf.org/html/rfc2822#section-2.2
         https://tools.ietf.org/html/rfc5322#section-2.2

    Args:
        message: Email message object
        context: Context string for error messages

    Returns:
        Charset string

    Raises:
        ValueError: If charset cannot be determined or is invalid
    """
    LOGGER.debug("Determining charset for message in context: %s", context)
    
    charset = message.get_content_charset()
    if charset:
        LOGGER.debug("Found charset in message: %s", charset)
        # Validate that the charset is usable
        try:
            codecs.lookup(charset)
            LOGGER.debug("Charset '%s' validated successfully", charset)
            return charset
        except LookupError as exc:
            LOGGER.error("Invalid charset '%s' in %s: %s", charset, context, exc, exc_info=True)
            raise ValueError(f"Invalid charset '{charset}' in {context}") from exc

    # Per RFC standards, default to US-ASCII for text content if no charset specified
    content_type = message.get_content_type() or ""
    LOGGER.debug("No charset specified, content type: %s", content_type)
    
    if content_type.startswith("text/"):
        LOGGER.debug("Text content type, defaulting to US-ASCII")
        return "us-ascii"

    LOGGER.error("No charset specified and non-text content type '%s' in %s", content_type, context)
    raise ValueError(
        f"No charset specified and content type '{content_type}' in {context} - "
        f"cannot determine encoding"
    )


def _extract_and_decode_payload(message: Message, log_context: str) -> bytes:
    """Extract and decode payload from email message with proper charset handling.

    Uses Message.get_payload(decode=True) to automatically handle Content-Transfer-Encoding
    decoding as recommended by the email library API.

    Args:
        message: Email message object
        log_context: Context string for error logging

    Returns:
        Decoded payload bytes

    Raises:
        ValueError: If payload cannot be extracted or decoded
    """
    LOGGER.debug("Extracting payload from message in context: %s", log_context)
    
    try:
        # Use the email library's built-in Content-Transfer-Encoding decoding
        payload = message.get_payload(decode=True)

        if payload is None:
            LOGGER.debug("Message payload is None in %s, returning empty bytes", log_context)
            return b""

        # get_payload(decode=True) should return bytes for binary data
        if isinstance(payload, bytes):
            LOGGER.debug("Successfully extracted payload: %d bytes in %s", len(payload), log_context)
            return payload

        # If we get a string, something went wrong - this shouldn't happen with decode=True
        LOGGER.error("Unexpected string payload from get_payload(decode=True) in %s", log_context)
        raise ValueError(
            f"Unexpected string payload from get_payload(decode=True) in {log_context}"
        )

    except Exception as exc:
        # Handle any decoding errors from the email library
        LOGGER.error("Failed to decode payload in %s: %s", log_context, exc, exc_info=True)
        raise ValueError(f"Failed to decode payload in {log_context}: {exc}") from exc


# --------------------------------------------------------------------------- #
# MHTML parsing & in-memory resource embedding                                 #
# --------------------------------------------------------------------------- #


def _build_resource_map_from_mhtml(
    path: Path,
) -> tuple[list[str], dict[str, tuple[str, bytes]]]:
    """Return (html_parts, resources) from an MHTML file; no temp files."""
    LOGGER.debug("Starting MHTML parsing for: %s", path)
    
    html_parts: list[str] = []
    resources: dict[str, tuple[str, bytes]] = {}
    
    try:
        with path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        LOGGER.debug("MHTML file parsed successfully")
    except Exception as exc:
        LOGGER.error("Failed to parse MHTML file %s: %s", path, exc, exc_info=True)
        raise ValueError(f"MHTML parsing failed for {path}: {exc}") from exc
    
    if msg.is_multipart():
        LOGGER.debug("Processing multipart MHTML message")
        parts = list(msg.walk())
        LOGGER.debug("Found %d MHTML parts to process", len(parts))
        
        for i, sub in enumerate(parts):
            # Only process Message objects
            if not hasattr(sub, "get_content_type"):
                LOGGER.debug("MHTML part %d has no get_content_type method, skipping", i)
                continue
            ctype = (sub.get_content_type() or "").lower()
            LOGGER.debug("Processing MHTML part %d: content_type=%s", i, ctype)

            # Extract and decode payload using helper function
            try:
                payload = _extract_and_decode_payload(sub, f"{path.name} part {i}")
                LOGGER.debug("MHTML part %d payload extracted: %d bytes", i, len(payload))
            except Exception as exc:
                LOGGER.error("Failed to extract payload from MHTML part %d in %s: %s", 
                           i, path, exc, exc_info=True)
                raise

            if ctype.startswith("text/html"):
                LOGGER.debug("Processing HTML part %d", i)
                # Get charset with strict error handling - no fallbacks
                try:
                    charset = _get_email_charset_or_error(sub, f"{path.name} HTML part {i}")
                    text = payload.decode(charset)  # Strict decoding - raise on invalid chars
                    html_parts.append(text)
                    LOGGER.info("HTML part %d processed successfully: charset=%s, length=%d chars", 
                               i, charset, len(text))
                except (ValueError, UnicodeDecodeError) as exc:
                    # Strict error handling - raise instead of falling back
                    LOGGER.error("HTML part encoding error in %s part %d: %s", 
                               path, i, exc, exc_info=True)
                    raise ValueError(f"HTML part encoding error in {path.name}: {exc}") from exc
            else:
                # Process as resource
                cid = (sub.get("Content-ID") or "").strip().strip("<>").strip()
                loc = (sub.get("Content-Location") or "").strip()
                
                resource_keys = []
                if cid:
                    resource_key = f"cid:{cid}"
                    resources[resource_key] = (ctype, payload)
                    resource_keys.append(resource_key)
                if loc:
                    resources[loc] = (ctype, payload)
                    resource_keys.append(loc)
                
                if resource_keys:
                    LOGGER.debug("Stored resource part %d: keys=%s, type=%s, size=%d bytes", 
                               i, resource_keys, ctype, len(payload))
                else:
                    LOGGER.debug("Part %d has no Content-ID or Content-Location, not stored as resource", i)
    else:
        LOGGER.debug("Processing single-part MHTML message")
        if (msg.get_content_type() or "").lower().startswith("text/html"):
            LOGGER.debug("Single-part message is HTML")
            # Extract and decode payload using helper function
            try:
                payload = _extract_and_decode_payload(msg, path.name)
                LOGGER.debug("Single-part payload extracted: %d bytes", len(payload))
            except Exception as exc:
                LOGGER.error("Failed to extract payload from single-part MHTML %s: %s", 
                           path, exc, exc_info=True)
                raise

            # Get charset with strict error handling - no fallbacks
            try:
                charset = _get_email_charset_or_error(msg, f"{path.name} main HTML")
                html_text = payload.decode(charset)  # Strict decoding - raise on invalid chars
                html_parts.append(html_text)
                LOGGER.info("Single-part HTML processed successfully: charset=%s, length=%d chars", 
                           charset, len(html_text))
            except (ValueError, UnicodeDecodeError) as exc:
                # Strict error handling - raise instead of falling back
                LOGGER.error("Main HTML encoding error in %s: %s", path, exc, exc_info=True)
                raise ValueError(f"Main HTML encoding error in {path.name}: {exc}") from exc
        else:
            LOGGER.warning("Single-part MHTML message is not HTML: %s", msg.get_content_type())
    
    LOGGER.info("MHTML parsing completed: %d HTML parts, %d resources", 
               len(html_parts), len(resources))
    return html_parts, resources


def _to_data_uri(mime: str, data: bytes) -> str:
    # base64 encoding always produces ASCII-safe output per RFC 4648
    # See: https://tools.ietf.org/html/rfc4648#section-4
    """Convert binary data to data URI format."""
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def _resolve_embeds(
    html: str, resources: dict[str, tuple[str, bytes]] | None, log_prefix: str = ""
) -> str:
    """Inline cid:/Content-Location resources as data: URIs so converter sees them."""
    if not resources:
        LOGGER.debug("%sNo resources provided for embed resolution", log_prefix)
        return html
    
    LOGGER.debug("%sStarting embed resolution with %d resources", log_prefix, len(resources))
    soup = BeautifulSoup(html, "lxml")
    
    embed_tags = soup.find_all(["img", "a", "source"])
    LOGGER.debug("%sFound %d potential embed tags to process", log_prefix, len(embed_tags))
    
    resolved_count = 0
    unresolved_count = 0
    
    for i, tag in enumerate(embed_tags):
        # Type narrowing: find_all with specific tag names returns Tag objects
        if not isinstance(tag, Tag):
            LOGGER.debug("%sTag %d is not a Tag instance, skipping", log_prefix, i)
            continue
        if not hasattr(tag, "name") or not hasattr(tag, "get"):
            LOGGER.debug("%sTag %d missing required attributes, skipping", log_prefix, i)
            continue
        
        attr = "src" if tag.name != "a" else "href"
        val_raw = tag.get(attr)
        if val_raw is None:
            LOGGER.debug("%sTag %d (%s) has no %s attribute", log_prefix, i, tag.name, attr)
            continue
        
        # Handle different types from BeautifulSoup get()
        if isinstance(val_raw, list):
            val = " ".join(str(v) for v in val_raw).strip()
        else:
            val = str(val_raw).strip()
        
        if not val:
            LOGGER.debug("%sTag %d (%s) has empty %s value", log_prefix, i, tag.name, attr)
            continue
        
        if val in resources or (val.startswith("cid:") and val in resources):
            mime, data = resources[val]
            data_uri = _to_data_uri(mime, data)
            tag[attr] = data_uri
            resolved_count += 1
            LOGGER.debug("%sResolved embed %d: %s -> data URI (%s, %d bytes)", 
                        log_prefix, i, val, mime, len(data))
        elif val.startswith("cid:"):
            unresolved_count += 1
            LOGGER.warning("%sUnresolved CID resource: %s", log_prefix, val)
        else:
            LOGGER.debug("%sTag %d (%s) has non-CID resource: %s", log_prefix, i, tag.name, val)
    
    LOGGER.info("%sEmbed resolution completed: %d resolved, %d unresolved CID references", 
               log_prefix, resolved_count, unresolved_count)
    return str(soup)


# --------------------------------------------------------------------------- #
# HTML -> Markdown conversion (markdownify)                                    #
# --------------------------------------------------------------------------- #


def _html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown using markdownify (no temp files)."""
    # Reasonable defaults: ATX headers, GFM-friendly bullets, keep code fences
    md = _md(
        html,
        heading_style="ATX",
        escape_asterisks=False,
        escape_underscores=False,
        bullets="*",
        strip=None,
        convert=[
            "a",
            "img",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
            "pre",
            "code",
        ],
    )
    # Normalize excessive blank lines a bit
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


# --------------------------------------------------------------------------- #
# Role extraction with BeautifulSoup                                           #
# --------------------------------------------------------------------------- #


def try_extract_messages_with_roles(html: str) -> list[tuple[str, str]] | None:
    """Use BeautifulSoup selectors to extract (role, inner_html) messages."""
    LOGGER.debug("Starting message extraction from HTML (%d chars)", len(html))
    
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []

    # Structured exports (preferred)
    LOGGER.debug("Attempting structured export detection via data-message-author-role")
    structured_elements = soup.select("[data-message-author-role]")
    LOGGER.debug("Found %d elements with data-message-author-role attribute", len(structured_elements))
    
    for i, el1 in enumerate(structured_elements):
        if not hasattr(el1, "get"):
            LOGGER.debug("Element %d has no 'get' method, skipping", i)
            continue
        role_raw = el1.get("data-message-author-role")
        if role_raw is None:
            LOGGER.debug("Element %d has null data-message-author-role, skipping", i)
            continue
        # Handle different types from BeautifulSoup get()
        if isinstance(role_raw, list):
            role = " ".join(str(r) for r in role_raw).strip().lower()
        else:
            role = str(role_raw).strip().lower()
        
        LOGGER.debug("Element %d has role: '%s'", i, role)
        if role in {"user", "assistant", "system", "gpt"}:
            content = (
                el1.select_one(".markdown, .prose, .message-content, [data-message-content]") or el1
            )
            if hasattr(content, "decode_contents"):
                body_html = content.decode_contents()
                out.append((role, body_html))
                LOGGER.debug("Extracted structured message: role=%s, content_length=%d", 
                           role, len(body_html))
            else:
                LOGGER.debug("Content element has no decode_contents method for role %s", role)
        else:
            LOGGER.debug("Unrecognized role '%s' in structured export", role)
    
    if out:
        LOGGER.info("Structured export detection successful: extracted %d messages", len(out))
        return out

    # Microsoft Copilot conversation detection
    LOGGER.debug("Structured export failed, attempting Microsoft Copilot detection")
    copilot_chat = soup.select_one('[data-testid="highlighted-chats"]')
    if copilot_chat:
        LOGGER.debug("Found Microsoft Copilot chat container")
        copilot_result = _extract_copilot_messages(copilot_chat)
        if copilot_result:
            LOGGER.info("Microsoft Copilot extraction successful: extracted %d messages", len(copilot_result))
            return copilot_result
        else:
            LOGGER.debug("Microsoft Copilot extraction returned no messages")
    else:
        LOGGER.debug("No Microsoft Copilot chat container found")

    # Heuristic class-based (with filtering for actual conversation content)
    LOGGER.debug("Attempting heuristic class-based extraction")
    candidates = soup.find_all(["div", "section", "article"], class_=True)
    LOGGER.debug("Found %d potential class-based candidates", len(candidates))
    
    for i, el2 in enumerate(candidates):
        if not isinstance(el2, Tag):
            LOGGER.debug("Candidate %d is not a Tag, skipping", i)
            continue
        # Elements from find_all with specific tag names are always Tag objects
        class_raw = el2.get("class", None)
        if class_raw is None:
            LOGGER.debug("Candidate %d has no class attribute, skipping", i)
            continue
        if isinstance(class_raw, list):
            classes = " ".join(str(c) for c in class_raw).lower()
        else:
            classes = str(class_raw).lower()
        role = (
            "assistant"
            if any(k in classes for k in ("assistant", "gpt", "bot"))
            else ("user" if any(k in classes for k in ("user", "you")) else "unknown")
        )
        
        if role != "unknown":
            # Filter out UI elements by checking for meaningful content
            text_content = el2.get_text().strip()
            if len(text_content) < 20:  # Skip short UI elements
                LOGGER.debug("Candidate %d has insufficient content (%d chars), skipping", 
                           i, len(text_content))
                continue

            # Skip elements that are clearly UI components
            if any(
                ui_term in classes
                for ui_term in (
                    "absolute",
                    "relative", 
                    "fixed",
                    "sticky",
                    "hidden",
                    "pointer-events-none",
                    "bottom-0",
                    "top-0",
                    "z-10",
                    "z-20",
                    "overlay",
                    "backdrop",
                )
            ):
                LOGGER.debug("Candidate %d contains UI classes, skipping", i)
                continue

            content = (
                el2.select_one(".markdown, .prose, .message-content, [data-message-content]") or el2
            )
            if hasattr(content, "decode_contents"):
                body_html = content.decode_contents()
                out.append((role, body_html))
                LOGGER.debug("Extracted heuristic message: role=%s, content_length=%d", 
                           role, len(body_html))
        else:
            LOGGER.debug("Candidate %d has unknown role (classes: %s)", i, classes[:100])
    
    if out:
        LOGGER.info("Heuristic class-based extraction successful: extracted %d messages", len(out))
        return out

    # ARIA/data-role hints
    LOGGER.debug("Class-based extraction failed, attempting ARIA/data-role extraction")
    aria_candidates = soup.select('[aria-label*="User" i], [aria-label*="Assistant" i], [data-role]')
    LOGGER.debug("Found %d ARIA/data-role candidates", len(aria_candidates))
    
    for i, el3 in enumerate(aria_candidates):
        if not hasattr(el3, "get"):
            LOGGER.debug("ARIA candidate %d has no 'get' method, skipping", i)
            continue
        aria_raw = el3.get("aria-label")
        drole_raw = el3.get("data-role")

        # Handle different types from BeautifulSoup get()
        if isinstance(aria_raw, list):
            aria = " ".join(str(a) for a in aria_raw).lower()
        else:
            aria = str(aria_raw or "").lower()

        if isinstance(drole_raw, list):
            drole = " ".join(str(d) for d in drole_raw).lower()
        else:
            drole = str(drole_raw or "").lower()

        role = (
            "assistant"
            if "assistant" in aria or "assistant" in drole
            else ("user" if "user" in aria or "user" in drole else "unknown")
        )
        
        LOGGER.debug("ARIA candidate %d: aria='%s', drole='%s', detected_role='%s'", 
                    i, aria[:50], drole[:50], role)
        
        if role != "unknown" and hasattr(el3, "decode_contents"):
            body_html = el3.decode_contents()
            out.append((role, body_html))
            LOGGER.debug("Extracted ARIA message: role=%s, content_length=%d", 
                       role, len(body_html))
    
    if out:
        LOGGER.info("ARIA/data-role extraction successful: extracted %d messages", len(out))
        return out
    
    LOGGER.warning("All message extraction methods failed - no structured dialogue found")
    return None


def _extract_copilot_messages(chat_container: Tag) -> list[tuple[str, str]] | None:
    """Extract conversation messages from Microsoft Copilot chat container."""
    LOGGER.debug("Starting Microsoft Copilot message extraction")
    
    # Get the full text content and parse it for conversation patterns
    full_text = chat_container.get_text()
    LOGGER.debug("Copilot chat container text length: %d chars", len(full_text))

    # Microsoft Copilot pattern: "Sie sagten" followed by content,
    # then "Copilot sagt[e]" followed by content
    messages = []

    # Split by "Sie sagten" to get conversation segments
    # Skip first split (before first "Sie sagten")
    segments = full_text.split("Sie sagten")[1:]
    LOGGER.debug("Found %d 'Sie sagten' segments in Copilot chat", len(segments))

    for i, segment in enumerate(segments):
        LOGGER.debug("Processing Copilot segment %d/%d (length: %d chars)", 
                    i+1, len(segments), len(segment))
        
        # Look for Copilot responses (handling both "Copilot sagt" and "Copilot sagte")
        copilot_match = re.search(
            r"Copilot sagt[e]?(.+?)(?=Sie sagten|Nachricht an Copilot|$)",
            segment,
            re.DOTALL,
        )
        if copilot_match:
            LOGGER.debug("Found Copilot response pattern in segment %d", i+1)
            
            # Extract user message (everything before "Copilot sagt[e]")
            user_split = re.split(pattern=r"Copilot sagt[e]?", string=segment, maxsplit=1)
            if len(user_split) > 0:
                user_content = user_split[0].strip()
                if user_content and len(user_content) > 5:
                    messages.append(("user", user_content))
                    LOGGER.debug("Extracted user message from segment %d: %d chars", 
                               i+1, len(user_content))
                else:
                    LOGGER.debug("User content in segment %d too short or empty: %d chars", 
                               i+1, len(user_content))

            # Extract assistant message
            assistant_content = copilot_match.group(1).strip()
            # Remove trailing input prompts and UI text
            assistant_content = re.sub(
                r"Nachricht an Copilot.*$", "", assistant_content, flags=re.DOTALL
            ).strip()
            if assistant_content and len(assistant_content) > 5:
                messages.append(("assistant", assistant_content))
                LOGGER.debug("Extracted assistant message from segment %d: %d chars", 
                           i+1, len(assistant_content))
            else:
                LOGGER.debug("Assistant content in segment %d too short or empty: %d chars", 
                           i+1, len(assistant_content))
        else:
            LOGGER.debug("No Copilot response pattern found in segment %d", i+1)

    if messages:
        LOGGER.info("Microsoft Copilot extraction successful: %d messages extracted", len(messages))
        return messages
    else:
        LOGGER.warning("Microsoft Copilot extraction found no valid messages")
        return None


def _extract_dialogue_title(html: str) -> str:
    """Extract a title for the dialogue from HTML content."""
    LOGGER.debug("Extracting dialogue title from HTML (%d chars)", len(html))
    
    soup = BeautifulSoup(html, "lxml")
    # Try to extract from HTML title tag
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)
        LOGGER.debug("Found HTML title tag: '%s'", title)
        
        # Clean up common ChatGPT/Copilot title patterns
        original_title = title
        title = re.sub(r"\s*-\s*(ChatGPT|Microsoft Copilot|OpenAI).*$", "", title, flags=re.I)
        if title != original_title:
            LOGGER.debug("Cleaned title: '%s' -> '%s'", original_title, title)
        
        if title and len(title.strip()) > 0:
            final_title = title.strip()
            LOGGER.debug("Using extracted title: '%s'", final_title)
            return final_title
        else:
            LOGGER.debug("Title became empty after cleaning")
    else:
        LOGGER.debug("No title tag found or title tag is empty")

    # Fall back to a default title
    default_title = "Chat Session"
    LOGGER.debug("Using default title: '%s'", default_title)
    return default_title


def dialogue_html_to_md(
    html: str,
    resources: dict[str, tuple[str, bytes]] | None = None,
    log_prefix: str = "",
) -> str:
    """Try role-bucketed rendering; fall back to full-page rendering."""
    LOGGER.debug("%sStarting dialogue extraction from HTML (%d chars)", 
                log_prefix, len(html))
    
    html_inlined = _resolve_embeds(html, resources, log_prefix=log_prefix)
    if len(html_inlined) != len(html):
        LOGGER.debug("%sHTML embed resolution changed size: %d -> %d chars", 
                    log_prefix, len(html), len(html_inlined))

    LOGGER.debug("%sAttempting structured dialogue extraction", log_prefix)
    msgs = try_extract_messages_with_roles(html_inlined)
    if msgs:
        LOGGER.info("%sSuccessfully extracted structured dialogue with %d messages", 
                   log_prefix, len(msgs))
        
        # Extract dialogue title
        title = _extract_dialogue_title(html_inlined)
        LOGGER.debug("%sExtracted dialogue title: '%s'", log_prefix, title)

        blocks: list[str] = [f"# {title}"]
        for i, (role, body) in enumerate(msgs):
            role_label = "User" if role == "user" else "Assistant"
            LOGGER.debug("%sProcessing message %d/%d: role=%s, content_length=%d", 
                        log_prefix, i+1, len(msgs), role, len(body))
            md = _html_to_markdown(body).strip()
            if md:
                blocks.append(f"## {role_label}\n\n{md}")
                LOGGER.debug("%sMessage %d converted to %d chars of markdown", 
                            log_prefix, i+1, len(md))
            else:
                LOGGER.warning("%sMessage %d produced empty markdown content (role=%s, html_length=%d)", 
                              log_prefix, i+1, role, len(body))
        
        if len(blocks) > 1:  # Only return if we have title + at least one message
            result = ("\n\n".join(blocks)).strip() + "\n"
            LOGGER.info("%sStructured dialogue extraction completed: %d blocks, %d total chars", 
                       log_prefix, len(blocks), len(result))
            return result
        else:
            LOGGER.warning("%sStructured extraction produced only title, no valid messages", log_prefix)

    LOGGER.warning("%sStructured dialogue extraction failed, falling back to full-page rendering", 
                  log_prefix)
    fallback_result = _html_to_markdown(html_inlined)
    LOGGER.info("%sFallback extraction completed: %d chars", log_prefix, len(fallback_result))
    return fallback_result


# --------------------------------------------------------------------------- #
# PDF extraction (pypdf)                                                       #
# --------------------------------------------------------------------------- #


def _pdf_to_text(path: Path) -> tuple[str, int]:
    """Extract text from PDF using pypdf (best-effort, structure lost)."""
    LOGGER.debug("Starting PDF text extraction: %s", path)
    
    try:
        reader = pypdf.PdfReader(str(path))
        LOGGER.debug("PDF reader initialized successfully")
    except Exception as exc:
        LOGGER.error("Failed to initialize PDF reader for %s: %s", path, exc, exc_info=True)
        raise RuntimeError(f"PDF reader initialization failed for {path}: {exc}") from exc
    
    LOGGER.debug("PDF has %d pages", len(reader.pages))
    pages: list[str] = []
    
    for i, page in enumerate(reader.pages):
        LOGGER.debug("Processing PDF page %d/%d", i+1, len(reader.pages))
        try:
            txt = page.extract_text() or ""
            LOGGER.debug("Page %d text extraction: %d chars", i+1, len(txt))
        except Exception as exc:
            LOGGER.warning("Failed to extract text from page %d in %s: %s", 
                         i+1, path, exc, exc_info=True)
            txt = ""
        if txt.strip():
            pages.append(txt.strip())
            LOGGER.debug("Page %d added to output", i+1)
        else:
            LOGGER.debug("Page %d skipped - no text content", i+1)
    
    result_text = "\n\n---\n\n".join(pages).strip()
    LOGGER.info("PDF text extraction completed: %d pages processed, %d pages with content, %d total chars", 
               len(reader.pages), len(pages), len(result_text))
    return (result_text, len(pages))


# --------------------------------------------------------------------------- #
# Per-file worker                                                              #
# --------------------------------------------------------------------------- #


def _process_single(path: Path, outdir: Path | None) -> list[Path]:
    """Process a single input file and convert it to Markdown."""
    LOGGER.info("Starting processing of file: %s", path)
    produced: list[Path] = []
    suffix = path.suffix.lower()

    try:
        if suffix in (".mhtml", ".mht"):
            LOGGER.info("Processing MHTML file: %s", path)
            LOGGER.debug("File size: %d bytes", path.stat().st_size)
            
            try:
                html_parts, resources = _build_resource_map_from_mhtml(path)
                LOGGER.info("MHTML parsing completed: %d HTML parts, %d resources", 
                           len(html_parts), len(resources))
                
                if resources:
                    LOGGER.debug("MHTML resources: %s", list(resources.keys())[:10])  # Log first 10 resource keys
                
            except Exception as exc:
                LOGGER.error("Failed to parse MHTML file %s: %s", path, exc, exc_info=True)
                raise RuntimeError(f"MHTML parsing failed for {path}: {exc}") from exc
            
            if not html_parts:
                LOGGER.error("No text/html parts found in MHTML file: %s", path)
                raise RuntimeError(f"No text/html parts found in {path}")
            
            _warn_better_format_guess_for_mhtml(html_parts, resources, path)
            
            for i, html in enumerate(html_parts):
                LOGGER.debug("Processing MHTML part %d/%d (length: %d chars)", 
                           i+1, len(html_parts), len(html))
                
                try:
                    md = dialogue_html_to_md(
                        html, resources=resources, log_prefix=f"[{path.name} part {i}] "
                    )
                except Exception as exc:
                    LOGGER.error("Dialogue extraction failed for %s part %d: %s", 
                               path, i, exc, exc_info=True)
                    raise RuntimeError(f"Dialogue extraction failed for {path} part {i}: {exc}") from exc
                
                if not md.strip():
                    LOGGER.error("No extractable content in %s part %d", path, i)
                    raise RuntimeError(f"No extractable content in {path} part {i}")
                
                out = (outdir or path.parent) / f"{path.stem}-part{i}.md"
                LOGGER.debug("Writing output to: %s (%d chars)", out, len(md))
                
                # Use system encoding for output files
                system_encoding = _get_system_encoding()
                try:
                    out.write_text(md, encoding=system_encoding)
                    LOGGER.info("Successfully wrote MHTML part %d to: %s", i, out)
                except UnicodeEncodeError as exc:
                    # If system encoding fails, raise error instead of falling back
                    LOGGER.critical("Cannot write output file %s with system encoding %s: %s", 
                                   out, system_encoding, exc, exc_info=True)
                    raise RuntimeError(
                        f"Cannot write output file {out} with system encoding {system_encoding}"
                    ) from exc
                produced.append(out)

        elif suffix in (".html", ".htm"):
            LOGGER.info("Processing HTML file: %s", path)
            LOGGER.debug("File size: %d bytes", path.stat().st_size)
            
            # Use system encoding for reading HTML files
            system_encoding = _get_system_encoding()
            try:
                html = path.read_text(
                    encoding=system_encoding
                )  # Strict reading - raise on decode errors
                LOGGER.debug("Successfully read HTML file with encoding %s: %d chars", 
                           system_encoding, len(html))
            except UnicodeDecodeError as exc:
                LOGGER.critical("Cannot read HTML file %s with system encoding %s: %s", 
                               path, system_encoding, exc, exc_info=True)
                raise ValueError(
                    f"Cannot read HTML file {path} with system encoding {system_encoding}: {exc}"
                ) from exc
            
            _warn_better_format_guess_for_html(html, path)
            
            try:
                md = dialogue_html_to_md(html, resources=None, log_prefix=f"[{path.name}] ")
            except Exception as exc:
                LOGGER.error("Dialogue extraction failed for HTML file %s: %s", 
                           path, exc, exc_info=True)
                raise RuntimeError(f"Dialogue extraction failed for {path}: {exc}") from exc
            
            if not md.strip():
                LOGGER.error("No extractable content in HTML file: %s", path)
                raise RuntimeError(f"No extractable content in {path}")
            
            out = (outdir or path.parent) / f"{path.stem}.md"
            LOGGER.debug("Writing HTML output to: %s (%d chars)", out, len(md))
            
            # Use system encoding for output files
            try:
                out.write_text(md, encoding=system_encoding)
                LOGGER.info("Successfully wrote HTML output to: %s", out)
            except UnicodeEncodeError as exc:
                LOGGER.critical("Cannot write output file %s with system encoding %s: %s", 
                               out, system_encoding, exc, exc_info=True)
                raise RuntimeError(
                    f"Cannot write output file {out} with system encoding {system_encoding}"
                ) from exc
            produced.append(out)

        elif suffix == ".pdf":
            LOGGER.info("Processing PDF file: %s", path)
            LOGGER.debug("File size: %d bytes", path.stat().st_size)
            
            try:
                text, pages = _pdf_to_text(path)
                LOGGER.info("PDF text extraction completed: %d pages, %d chars", pages, len(text))
            except Exception as exc:
                LOGGER.error("PDF text extraction failed for %s: %s", path, exc, exc_info=True)
                raise RuntimeError(f"PDF extraction failed for {path}: {exc}") from exc
            
            _warn_better_format_guess_for_pdf(pages_extracted=pages, text_len=len(text), path=path)
            
            if not text.strip():
                LOGGER.error("No extractable content in PDF file: %s", path)
                raise RuntimeError(f"No extractable content in {path}")
            
            out = (outdir or path.parent) / f"{path.stem}.md"
            LOGGER.debug("Writing PDF output to: %s (%d chars)", out, len(text))
            
            # Use system encoding for output files
            system_encoding = _get_system_encoding()
            try:
                out.write_text(text.strip() + "\n", encoding=system_encoding)
                LOGGER.info("Successfully wrote PDF output to: %s", out)
            except UnicodeEncodeError as exc:
                LOGGER.critical("Cannot write output file %s with system encoding %s: %s", 
                               out, system_encoding, exc, exc_info=True)
                raise RuntimeError(
                    f"Cannot write output file {out} with system encoding {system_encoding}"
                ) from exc
            produced.append(out)

        else:
            LOGGER.error("Unsupported file type '%s' for file: %s", suffix, path)
            raise RuntimeError(f"Unsupported file type: {path.suffix}")

    except Exception as exc:
        # Re-raise with additional context, but don't double-log
        if not isinstance(exc, RuntimeError):
            LOGGER.error("Unexpected error processing file %s: %s", path, exc, exc_info=True)
        raise

    LOGGER.info("File processing completed successfully: %s -> %d output(s)", 
               path, len(produced))
    return produced


# --------------------------------------------------------------------------- #
# Path expansion & batch processing                                            #
# --------------------------------------------------------------------------- #


def expand_paths(inputs: Sequence[str]) -> list[Path]:
    """Expand glob patterns in input paths and return deduplicated resolved paths."""
    LOGGER.debug("Expanding %d input patterns: %s", len(inputs), inputs)
    
    expanded: list[Path] = []
    for i, p in enumerate(inputs):
        LOGGER.debug("Processing input pattern %d/%d: %s", i+1, len(inputs), p)
        matches = glob.glob(p)
        LOGGER.debug("Pattern '%s' matched %d files: %s", p, len(matches), matches[:5])  # Show first 5 matches
        
        for m in matches:
            path_obj = Path(m).resolve()
            expanded.append(path_obj)
    
    LOGGER.debug("Total expanded paths before deduplication: %d", len(expanded))
    
    # de-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for path_item in expanded:
        if path_item not in seen:
            uniq.append(path_item)
            seen.add(path_item)
        else:
            LOGGER.debug("Skipping duplicate path: %s", path_item)
    
    LOGGER.debug("Final unique paths after deduplication: %d", len(uniq))
    return uniq


def process_many(inputs: Sequence[str], outdir: Path | None, jobs: int) -> list[Path]:
    """Process multiple input files concurrently and return list of output paths."""
    LOGGER.info("Starting batch processing: %d input patterns, output_dir=%s, jobs=%s", 
               len(inputs), outdir, jobs or "auto")
    
    files = expand_paths(inputs)
    LOGGER.info("Path expansion completed: %d input patterns -> %d files", len(inputs), len(files))
    
    if not files:
        LOGGER.warning("No files found after path expansion")
        return []

    LOGGER.debug("Files to process: %s", [str(f) for f in files])

    # group-level format advice
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        stem = p.stem.lower()
        by_stem.setdefault(stem, []).append(p)
    
    LOGGER.debug("Grouped files by stem: %d unique stems", len(by_stem))
    
    for stem, file_list in by_stem.items():
        exts = {f.suffix.lower() for f in file_list}
        LOGGER.debug("File group '%s' has extensions: %s", stem, exts)
        
        if any(e in exts for e in [".html", ".htm"]) and any(e in exts for e in [".mhtml", ".mht"]):
            paths_str = ", ".join(str(f) for f in file_list)
            LOGGER.warning(
                "Both HTML and MHTML present for files: %s. "
                "The tool will compare them; prefer the richer result.",
                paths_str,
            )
        if ".pdf" in exts and (any(e in exts for e in [".html", ".htm", ".mhtml", ".mht"])):
            paths_str = ", ".join(str(f) for f in file_list)
            LOGGER.warning(
                "PDF provided alongside HTML/MHTML for files: %s; "
                "prefer HTML/MHTML over PDF when possible.",
                paths_str,
            )

    if outdir:
        LOGGER.debug("Creating output directory: %s", outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Output directory ready: %s", outdir)

    # Calculate batch characteristics for executor selection
    total_size = 0
    for p in files:
        with contextlib.suppress(OSError):
            total_size += p.stat().st_size
    
    small_batch = (len(files) < 8) or (total_size < 8 * 1024 * 1024)
    max_workers = max(1, jobs or os.cpu_count() or 4)
    executor = ThreadPoolExecutor if small_batch else ProcessPoolExecutor
    
    LOGGER.info("Batch processing configuration: executor=%s, workers=%d, total_size=%d bytes, small_batch=%s", 
               executor.__name__, max_workers, total_size, small_batch)

    produced_total: list[Path] = []
    failures: list[str] = []
    
    LOGGER.info("Starting concurrent processing of %d files", len(files))
    with executor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_single, p, outdir): p for p in files}
        LOGGER.debug("Submitted %d processing tasks", len(futs))
        
        completed_count = 0
        for fut in as_completed(futs):
            src = futs[fut]
            completed_count += 1
            
            try:
                result = fut.result()
                produced_total.extend(result)
                LOGGER.info("Processing completed (%d/%d): %s -> %d output(s)", 
                           completed_count, len(files), src, len(result))
            except Exception as exc:
                failure_msg = f"{src}: {exc}"
                failures.append(failure_msg)
                LOGGER.error("Processing failed (%d/%d): %s", 
                           completed_count, len(files), src, exc_info=True)
    
    if failures:
        LOGGER.critical("Batch processing completed with %d failures out of %d files", 
                       len(failures), len(files))
        LOGGER.critical("Failed files:\n%s", "\n".join(failures))
        raise RuntimeError("Some files failed:\n" + "\n".join(failures))
    
    LOGGER.info("Batch processing completed successfully: %d files -> %d outputs", 
               len(files), len(produced_total))
    return produced_total
