"""Chunking strategies for Practo Domain Support Agent.

Implements two distinct chunking strategies:
1. Fixed-size chunking with overlap (character/word-based sliding window)
2. Sentence-based chunking (semantic sentence boundary splitting)
"""

import re
from typing import List, Dict, Any

def fixed_size_chunking(
    docs: List[Dict[str, str]],
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[Dict[str, Any]]:
    """Chunk documents using a fixed character size with sliding overlap.

    Args:
        docs: List of documents containing doc_id, topic, and content.
        chunk_size: Maximum character length per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with id, text, doc_id, topic, and strategy.
    """
    chunks = []
    for doc in docs:
        text = doc["content"].strip()
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc.get("title", topic)

        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"fixed_{doc_id}_c{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "topic": topic,
                    "title": title,
                    "chunk_idx": chunk_idx,
                    "strategy": "fixed_size_overlap",
                })
                chunk_idx += 1

            if end >= text_len:
                break
            start += (chunk_size - overlap)

    return chunks


def sentence_based_chunking(
    docs: List[Dict[str, str]],
    sentences_per_chunk: int = 1,
) -> List[Dict[str, Any]]:
    """Chunk documents by individual or paired sentence boundaries.

    Args:
        docs: List of documents containing doc_id, topic, and content.
        sentences_per_chunk: Number of consecutive sentences per chunk.

    Returns:
        List of chunk dicts with id, text, doc_id, topic, and strategy.
    """
    chunks = []
    sentence_splitter = re.compile(r"(?<=[.!?])\s+")

    for doc in docs:
        text = doc["content"].strip()
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc.get("title", topic)

        raw_sentences = [s.strip() for s in sentence_splitter.split(text) if s.strip()]

        chunk_idx = 0
        for i in range(0, len(raw_sentences), sentences_per_chunk):
            grouped = raw_sentences[i : i + sentences_per_chunk]
            chunk_text = " ".join(grouped).strip()
            if chunk_text:
                chunk_id = f"sent_{doc_id}_c{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "topic": topic,
                    "title": title,
                    "chunk_idx": chunk_idx,
                    "strategy": "sentence_based",
                })
                chunk_idx += 1

    return chunks
