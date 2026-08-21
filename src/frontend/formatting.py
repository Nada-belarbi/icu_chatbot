"""Safe presentation helpers for the Streamlit frontend.

These helpers deliberately transform only display values.  They never inspect
or alter the response, safety decision, or sources returned by the pipeline.
"""

from __future__ import annotations

import html
import re
from pathlib import PurePosixPath


_ORGANISATIONS = {
    "AHRQ": "Agency for Healthcare Research and Quality",
    "ELSO": "ELSO",
    "ESICM": "ESICM",
    "ESPEN": "ESPEN",
    "GOV": "Service Public",
    "HAS": "Haute Autorité de Santé",
    "HAS_DROITS": "Haute Autorité de Santé",
    "ICUS": "Intensive Care Society",
    "KDIGO": "KDIGO",
    "NHS": "NHS",
    "NICE": "NICE",
    "NIH": "NIH / MedlinePlus",
    "RAW": "Sources hospitalières",
    "SCCM": "Society of Critical Care Medicine",
    "SRLF": "Société de Réanimation de Langue Française",
    "UCLH": "University College London Hospitals",
}


def content_to_html(content: str) -> str:
    """Render plain response text with a small, safe Markdown subset."""
    escaped = html.escape(str(content)).replace("\r\n", "\n")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", escaped):
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue
        if all(line.startswith(("• ", "- ", "* ")) for line in lines):
            items = "".join(f"<li>{line[2:]}</li>" for line in lines)
            blocks.append(f"<ul>{items}</ul>")
        else:
            blocks.append(f"<p>{'<br>'.join(lines)}</p>")
    return "".join(blocks) or "<p></p>"


def humanize_source(source: str) -> tuple[str, str]:
    """Convert an internal retrieval path into a person-readable source card."""
    normalized = str(source).replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    organisation_key = next((part for part in parts if part in _ORGANISATIONS), "")
    organisation = _ORGANISATIONS.get(organisation_key, "Source documentaire")
    filename = path.stem
    filename = re.sub(r"^[A-Z][A-Z0-9_-]*\d{2,4}_", "", filename)
    filename = filename.replace("_", " ").replace("-", " ")
    filename = re.sub(r"\s+", " ", filename).strip()
    title = filename[:1].upper() + filename[1:] if filename else "Document d'information"
    return organisation, title


def title_from_question(question: str) -> str:
    """Produce a bounded conversation title without invoking an external model."""
    cleaned = re.sub(r"\s+", " ", question).strip().rstrip("?.! ")
    if not cleaned:
        return "Nouvelle conversation"
    return cleaned[:77].rstrip() + ("…" if len(cleaned) > 77 else "")