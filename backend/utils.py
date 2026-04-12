# utils.py — Utility helpers for TalentLens AI
# Note: Skill extraction is now handled by skills.py (canonical taxonomy).
# This file provides text-cleaning helpers used by main.py.

import re


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from PDFs/DOCX.
    Removes excessive whitespace, weird unicode bullets, and null bytes.
    """
    if not text:
        return ""
    # Replace null bytes and common junk unicode
    text = text.replace("\x00", "").replace("\uf0b7", "•").replace("\uf0a7", "-")
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def truncate(text: str, max_chars: int = 8000) -> str:
    """
    Truncate text to max_chars to avoid embedding model overload.
    Sentence-transformers handles up to ~512 tokens (~2000 chars safely),
    but we allow more since we split by section internally.
    """
    return text[:max_chars] if len(text) > max_chars else text


def format_bytes(num_bytes: int) -> str:
    """Return a human-readable file size string."""
    if num_bytes >= 1_048_576:
        return f"{num_bytes / 1_048_576:.1f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"