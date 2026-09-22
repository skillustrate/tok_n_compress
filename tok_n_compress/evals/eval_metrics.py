"""
Evaluation Metrics for Conversation Compressor Skill.

Calculates quantitative benchmarks including Faithfulness, Completeness,
Entity Retention, Compression Ratio, and Retrieval Recall@K.
"""

import time
import re
from typing import Dict, List, Any, Set, Tuple


class BenchmarkEvaluator:
    """Evaluates conversation compression and retrieval quality."""

    @staticmethod
    def calculate_compression_metrics(
        original_tokens: int,
        compressed_tokens: int,
        original_chars: int,
        compressed_chars: int
    ) -> Dict[str, float]:
        """Calculate compression ratios and percentage token reduction."""
        ratio = round(original_chars / max(compressed_chars, 1), 2)
        token_reduction = round(
            ((original_tokens - compressed_tokens) / max(original_tokens, 1)) * 100, 2
        )
        return {
            "compression_ratio": ratio,
            "token_reduction_pct": token_reduction,
            "tokens_saved": max(original_tokens - compressed_tokens, 0)
        }

    @staticmethod
    def calculate_entity_retention(original_text: str, summary_text: str, targets: List[str]) -> Dict[str, Any]:
        """
        Evaluate how many target critical entities (e.g., error codes, filenames, constants)
        were preserved in the summary or its metadata.
        """
        found = []
        missing = []
        combined_text = summary_text.lower()

        for target in targets:
            if target.lower() in combined_text:
                found.append(target)
            else:
                missing.append(target)

        retention_rate = len(found) / max(len(targets), 1)
        return {
            "retention_rate": round(retention_rate, 4),
            "found_entities": found,
            "missing_entities": missing,
            "total_targets": len(targets)
        }

    @staticmethod
    def evaluate_retrieval_recall(
        retrieved_results: List[Dict[str, Any]],
        expected_needle: str,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate if expected needle appears in top-K retrieved raw segments.
        """
        top_k = retrieved_results[:k]
        found_rank = None

        for rank, match in enumerate(top_k, start=1):
            if expected_needle.lower() in match.get("content", "").lower():
                found_rank = rank
                break

        return {
            "hit_at_k": found_rank is not None,
            "rank": found_rank,
            "k": k,
            "total_candidates": len(retrieved_results)
        }

    @staticmethod
    def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time in milliseconds."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return result, round(duration_ms, 2)
