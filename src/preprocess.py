"""Shared text normalization used before embedding and caching."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace without changing the actual document content."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



