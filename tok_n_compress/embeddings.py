"""
Embedding Layer for AHMS (Agentic Hybrid Memory System).

Provides fast local vector representations and similarity metrics.
Features:
- FastLocalEmbedding: Zero-external-dependency n-gram hash projection (256-dim, L2 normalized).
- Pluggable interface for custom embedders (SentenceTransformers, OpenAI, Ollama, etc.).
- Binary serialization/deserialization for efficient SQLite BLOB storage.
- Fast cosine distance calculation.
"""

import math
import re
import struct
from typing import List, Optional, Callable


class FastLocalEmbedding:
    """
    Zero-dependency fast local embedding generator.
    Produces 256-dimensional L2-normalized vector embeddings from text
    using hashed character and subword n-grams.
    """

    DIMENSION = 256

    def __init__(self, custom_embedder: Optional[Callable[[str], List[float]]] = None):
        self.custom_embedder = custom_embedder

    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized float vector for the given text."""
        if self.custom_embedder is not None:
            raw = self.custom_embedder(text)
            return self.normalize(raw)

        if not text or not text.strip():
            return [0.0] * self.DIMENSION

        vec = [0.0] * self.DIMENSION
        clean = text.lower()

        # Word tokens (unigrams and bigrams)
        words = re.findall(r"\b[a-zA-Z0-9_\-\./#]+\b", clean)
        for i, word in enumerate(words):
            # Unigram hash
            idx = abs(hash(word)) % self.DIMENSION
            vec[idx] += 1.0

            # Bigram hash
            if i > 0:
                bigram = f"{words[i-1]}_{word}"
                b_idx = abs(hash(bigram)) % self.DIMENSION
                vec[b_idx] += 1.5

            # Character 3-grams for subword and root matching
            if len(word) >= 3:
                for j in range(len(word) - 2):
                    trigram = word[j:j+3]
                    t_idx = abs(hash(trigram)) % self.DIMENSION
                    vec[t_idx] += 0.5

        return self.normalize(vec)

    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """L2-normalize a vector so dot product equals cosine similarity."""
        norm_sq = sum(x * x for x in vector)
        if norm_sq <= 1e-12:
            return [0.0] * len(vector)
        norm = math.sqrt(norm_sq)
        return [round(x / norm, 6) for x in vector]

    @staticmethod
    def serialize_vector(vector: List[float]) -> bytes:
        """Pack float vector into binary bytes for SQLite BLOB storage."""
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def deserialize_vector(data: bytes) -> List[float]:
        """Unpack binary bytes into a list of floats."""
        if not data:
            return []
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    @staticmethod
    def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine distance (0.0 = identical, 2.0 = opposite).
        Assumes vectors are L2-normalized; cosine similarity is dot product.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 1.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        # Clamp dot between -1.0 and 1.0 for numerical safety
        dot = max(min(dot, 1.0), -1.0)
        return max(0.0, 1.0 - dot)


# Global singleton
_embedding_engine: Optional[FastLocalEmbedding] = None


def get_embedding_engine(custom_embedder: Optional[Callable[[str], List[float]]] = None) -> FastLocalEmbedding:
    """Get or create the global embedding engine."""
    global _embedding_engine
    if _embedding_engine is None or custom_embedder is not None:
        _embedding_engine = FastLocalEmbedding(custom_embedder=custom_embedder)
    return _embedding_engine
