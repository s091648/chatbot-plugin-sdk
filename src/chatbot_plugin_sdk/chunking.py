"""Default text chunking strategy — simple sliding window."""

from __future__ import annotations


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _chunk_text(text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using a simple character sliding window.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
