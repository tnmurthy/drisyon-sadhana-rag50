# Student implementation stub
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """TODO: Implement fixed-size or semantic text chunking with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """TODO: Calculate cosine similarity between two dense embedding vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
