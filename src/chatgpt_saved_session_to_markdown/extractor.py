# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""HTML/MHTML/PDF to Markdown extractor with role detection and warnings."""

from __future__ import annotations

import base64
import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup
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
            "%s: HTML references cid: resources; an MHTML export typically embeds those. Prefer MHTML if available.",
            path,
        )
    elif role_markers == 0 and img_http >= 5:
        LOGGER.warning(
            "%s: HTML has many external images but no clear chat role markers. "
            "An MHTML export often preserves inline assets better. Consider MHTML if available.",
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
    """Always warn that PDF is less preferred than HTML/MHTML."""
    LOGGER.warning(
        "%s: PDF text extraction is best-effort and loses structure. Prefer HTML or MHTML exports whenever available.",
        path,
    )


# --------------------------------------------------------------------------- #
# MHTML parsing & in-memory resource embedding                                 #
# --------------------------------------------------------------------------- #


def _build_resource_map_from_mhtml(path: Path) -> tuple[list[str], dict[str, tuple[str, bytes]]]:
    """Return (html_parts, resources) from an MHTML file; no temp files."""
    html_parts: list[str] = []
    resources: dict[str, tuple[str, bytes]] = {}
    with path.open("rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    if msg.is_multipart():
        for sub in msg.walk():
            if not isinstance(sub, Message):
                continue
            ctype = (sub.get_content_type() or "").lower()
            payload = sub.get_payload(decode=True) or b""
            if ctype.startswith("text/html"):
                try:
                    text = payload.decode(sub.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    text = payload.decode("utf-8", errors="replace")
                html_parts.append(text)
            else:
                cid = (sub.get("Content-ID") or "").strip().strip("<>").strip()
                loc = (sub.get("Content-Location") or "").strip()
                if cid:
                    resources[f"cid:{cid}"] = (ctype, payload)
                if loc:
                    resources[loc] = (ctype, payload)
    else:
        if (msg.get_content_type() or "").lower().startswith("text/html"):
            payload = msg.get_payload(decode=True) or b""
            html_parts.append(
                payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            )
    return html_parts, resources


def _to_data_uri(mime: str, data: bytes) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def _resolve_embeds(
    html: str, resources: dict[str, tuple[str, bytes]] | None, log_prefix: str = ""
) -> str:
    """Inline cid:/Content-Location resources as data: URIs so converter sees them."""
    if not resources:
        return html
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["img", "a", "source"]):
        attr = "src" if tag.name != "a" else "href"
        val = (tag.get(attr) or "").strip()
        if not val:
            continue
        if val in resources or (val.startswith("cid:") and val in resources):
            mime, data = resources[val]
            tag[attr] = _to_data_uri(mime, data)
        elif val.startswith("cid:"):
            LOGGER.warning("%sUnresolved CID resource: %s", log_prefix, val)
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
        convert=["a", "img", "table", "thead", "tbody", "tr", "th", "td", "pre", "code"],
    )
    # Normalize excessive blank lines a bit
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


# --------------------------------------------------------------------------- #
# Role extraction with BeautifulSoup                                           #
# --------------------------------------------------------------------------- #


def try_extract_messages_with_roles(html: str) -> list[tuple[str, str]] | None:
    """Use BeautifulSoup selectors to extract (role, inner_html) messages."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []

    # Structured exports (preferred)
    for el in soup.select("[data-message-author-role]"):
        role = (el.get("data-message-author-role") or "").strip().lower()
        if role in {"user", "assistant", "system", "gpt"}:
            content = (
                el.select_one(".markdown, .prose, .message-content, [data-message-content]") or el
            )
            body_html = content.decode_contents()
            out.append((role, body_html))
    if out:
        return out

    # Heuristic class-based
    candidates = soup.find_all(["div", "section", "article"], class_=True)
    for el in candidates:
        classes = " ".join(el.get("class", [])).lower()
        role = (
            "assistant"
            if any(k in classes for k in ("assistant", "gpt", "bot"))
            else ("user" if any(k in classes for k in ("user", "you")) else "unknown")
        )
        if role != "unknown":
            content = (
                el.select_one(".markdown, .prose, .message-content, [data-message-content]") or el
            )
            out.append((role, content.decode_contents()))
    if out:
        return out

    # ARIA/data-role hints
    for el in soup.select('[aria-label*="User" i], [aria-label*="Assistant" i], [data-role]'):
        aria = (el.get("aria-label") or "").lower()
        drole = (el.get("data-role") or "").lower()
        role = (
            "assistant"
            if "assistant" in aria or "assistant" in drole
            else ("user" if "user" in aria or "user" in drole else "unknown")
        )
        if role != "unknown":
            out.append((role, el.decode_contents()))
    return out or None


def dialogue_html_to_md(
    html: str, resources: dict[str, tuple[str, bytes]] | None = None, log_prefix: str = ""
) -> str:
    """Try role-bucketed rendering; fall back to full-page rendering."""
    html_inlined = _resolve_embeds(html, resources, log_prefix=log_prefix)

    msgs = try_extract_messages_with_roles(html_inlined)
    if msgs:
        blocks: list[str] = []
        for role, body in msgs:
            role_label = "User" if role == "user" else "Assistant"
            md = _html_to_markdown(body).strip()
            if md:
                blocks.append(f"### {role_label}\n\n{md}")
        if blocks:
            return ("\n\n".join(blocks)).strip() + "\n"

    return _html_to_markdown(html_inlined)


# --------------------------------------------------------------------------- #
# PDF extraction (pypdf)                                                       #
# --------------------------------------------------------------------------- #


def _pdf_to_text(path: Path) -> tuple[str, int]:
    """Extract text from PDF using pypdf (best-effort, structure lost)."""
    import pypdf  # runtime import to keep import cost low

    reader = pypdf.PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append(txt.strip())
    return ("\n\n---\n\n".join(pages).strip(), len(pages))


# --------------------------------------------------------------------------- #
# Per-file worker                                                              #
# --------------------------------------------------------------------------- #


def _process_single(path: Path, outdir: Path | None) -> list[Path]:
    produced: list[Path] = []
    suffix = path.suffix.lower()

    if suffix in (".mhtml", ".mht"):
        LOGGER.info("Processing MHTML: %s", path)
        html_parts, resources = _build_resource_map_from_mhtml(path)
        if not html_parts:
            raise RuntimeError(f"No text/html parts found in {path}")
        _warn_better_format_guess_for_mhtml(html_parts, resources, path)
        for i, html in enumerate(html_parts):
            md = dialogue_html_to_md(
                html, resources=resources, log_prefix=f"[{path.name} part {i}] "
            )
            if not md.strip():
                raise RuntimeError(f"No extractable content in {path} part {i}")
            out = (outdir or path.parent) / f"{path.stem}-part{i}.md"
            out.write_text(md, encoding="utf-8")
            produced.append(out)

    elif suffix in (".html", ".htm"):
        LOGGER.info("Processing HTML: %s", path)
        html = path.read_text(encoding="utf-8", errors="replace")
        _warn_better_format_guess_for_html(html, path)
        md = dialogue_html_to_md(html, resources=None, log_prefix=f"[{path.name}] ")
        if not md.strip():
            raise RuntimeError(f"No extractable content in {path}")
        out = (outdir or path.parent) / f"{path.stem}.md"
        out.write_text(md, encoding="utf-8")
        produced.append(out)

    elif suffix == ".pdf":
        LOGGER.info("Processing PDF: %s", path)
        text, pages = _pdf_to_text(path)
        _warn_better_format_guess_for_pdf(pages_extracted=pages, text_len=len(text), path=path)
        if not text.strip():
            raise RuntimeError(f"No extractable content in {path}")
        out = (outdir or path.parent) / f"{path.stem}.md"
        out.write_text(text.strip() + "\n", encoding="utf-8")
        produced.append(out)

    else:
        LOGGER.error("Unsupported file type: %s", path)
        raise RuntimeError(f"Unsupported file type: {path.suffix}")

    return produced


# --------------------------------------------------------------------------- #
# Path expansion & batch processing                                            #
# --------------------------------------------------------------------------- #


def expand_paths(inputs: Sequence[str]) -> list[Path]:
    import glob

    expanded: list[Path] = []
    for p in inputs:
        matches = glob.glob(p)
        for m in matches:
            expanded.append(Path(m).resolve())
    # de-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in expanded:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def process_many(inputs: Sequence[str], outdir: Path | None, jobs: int) -> list[Path]:
    files = expand_paths(inputs)
    if not files:
        return []

    # group-level format advice
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        stem = p.stem.lower()
        by_stem.setdefault(stem, []).append(p)
    for stem, file_list in by_stem.items():
        exts = {f.suffix.lower() for f in file_list}
        if any(e in exts for e in [".html", ".htm"]) and any(e in exts for e in [".mhtml", ".mht"]):
            paths_str = ", ".join(str(f) for f in file_list)
            LOGGER.warning(
                "Both HTML and MHTML present for files: %s. The tool will compare them; prefer the richer result.",
                paths_str,
            )
        if ".pdf" in exts and (any(e in exts for e in [".html", ".htm", ".mhtml", ".mht"])):
            paths_str = ", ".join(str(f) for f in file_list)
            LOGGER.warning(
                "PDF provided alongside HTML/MHTML for files: %s; prefer HTML/MHTML over PDF when possible.",
                paths_str,
            )

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    total_size = 0
    for p in files:
        try:
            total_size += p.stat().st_size
        except OSError:
            pass
    small_batch = (len(files) < 8) or (total_size < 8 * 1024 * 1024)
    max_workers = max(1, jobs or os.cpu_count() or 4)
    Executor = ThreadPoolExecutor if small_batch else ProcessPoolExecutor

    produced_total: list[Path] = []
    failures: list[str] = []
    with Executor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_single, p, outdir): p for p in files}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                produced_total.extend(fut.result())
            except Exception as exc:
                failures.append(f"{src}: {exc}")
                LOGGER.error("Failed: %s", src, exc_info=exc)
    if failures:
        raise RuntimeError("Some files failed:\n" + "\n".join(failures))
    return produced_total
