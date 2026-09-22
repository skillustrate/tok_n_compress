"""
Retrieval Layer for Hybrid Memory Model.

Enables search and "re-hydration" of compressed details from deep memory into the active context window.
"""

import re
from typing import Dict, List, Any, Optional

from .database import DatabaseManager, get_database_manager


class RetrievalEngine:
    """Handles retrieval and re-hydration of compressed conversation segments."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_database_manager()

    def query_memory(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search compressed memory for relevant segments matching user query.
        Uses a hybrid strategy: raw segment search with fallback/enrichment from checkpoint metadata.

        Args:
            query: The user's query or search terms.
            limit: Maximum number of results to return.

        Returns:
            List of matching segments with metadata and relevance scores.
        """
        trimmed_query = query.strip()
        if not trimmed_query:
            return []

        query_terms = [t.lower() for t in re.findall(r"\b\w+\b", trimmed_query) if len(t) > 1]

        scored_results: Dict[int, Dict[str, Any]] = {}

        # Step 1: Search raw_segments via exact query and terms
        raw_matches = self.db.search_raw_segments(trimmed_query, limit=limit * 2)
        for match in raw_matches:
            seg_id = match["id"]
            content = match["content"]
            score = self._compute_relevance(content, trimmed_query, query_terms)
            scored_results[seg_id] = {
                "segment_id": seg_id,
                "checkpoint_id": match["checkpoint_id"],
                "content": content,
                "timestamp": match.get("timestamp"),
                "relevance_score": score,
                "source": "raw_segment"
            }

        # Step 2: Fallback or enrichment via Checkpoints (summaries, key prompts, files)
        if len(scored_results) < limit:
            all_checkpoints = self.db.get_all_checkpoints(limit=limit * 3)
            for cp in all_checkpoints:
                cp_id = cp["id"]
                raw_segments = self.db.get_raw_segments_by_checkpoint(cp_id)
                if not raw_segments:
                    continue

                # Search within checkpoint summary
                summary_text = cp.get("summary", "")
                summary_score = self._compute_relevance(summary_text, trimmed_query, query_terms)

                # Search within key prompts
                prompts = cp.get("key_prompts", [])
                prompts_text = " ".join(
                    f"{p.get('prompt', '')} {p.get('outcome', '')}" for p in prompts
                )
                prompts_score = self._compute_relevance(prompts_text, trimmed_query, query_terms)

                # Search within files metadata
                files = cp.get("files_metadata", [])
                files_text = " ".join(
                    f"{f.get('filename', '')} {f.get('summary', '')}" for f in files
                )
                files_score = self._compute_relevance(files_text, trimmed_query, query_terms)

                best_cp_score = max(summary_score, prompts_score, files_score)
                if best_cp_score > 0.15:
                    combined_content = "\n\n".join(raw_segments)
                    # Use a synthetic ID or first segment ID
                    refs = cp.get("raw_segment_refs", [])
                    seg_id = refs[0] if refs else 0
                    if seg_id not in scored_results or best_cp_score > scored_results[seg_id]["relevance_score"]:
                        scored_results[seg_id] = {
                            "segment_id": seg_id,
                            "checkpoint_id": cp_id,
                            "content": combined_content,
                            "timestamp": cp.get("timestamp"),
                            "relevance_score": round(best_cp_score, 4),
                            "source": "checkpoint_metadata"
                        }

        sorted_matches = sorted(
            scored_results.values(),
            key=lambda x: x["relevance_score"],
            reverse=True
        )
        return sorted_matches[:limit]

    def _compute_relevance(self, text: str, full_query: str, query_terms: List[str]) -> float:
        """Compute relevance score (0.0 to 1.0) of text against query."""
        if not text:
            return 0.0

        lower_text = text.lower()
        score = 0.0

        # Exact phrase match gives high boost
        if full_query.lower() in lower_text:
            score += 0.6

        # Term overlap
        if query_terms:
            matched_terms = [t for t in query_terms if t in lower_text]
            term_ratio = len(matched_terms) / len(query_terms)
            score += 0.4 * term_ratio

        return min(round(score, 4), 1.0)

    def rehydrate_segment(
        self,
        segment: str,
        user_query: Optional[str] = None
    ) -> str:
        """
        Format a raw segment into a clean, structured context block for injection into active LLM context.

        Args:
            segment: The raw conversation content to inject.
            user_query: Current query for context awareness.

        Returns:
            Formatted string ready for context window insertion.
        """
        header = "[REHYDRATED CONVERSATION SEGMENT]"
        footer = "[END REHYDRATED SEGMENT]"

        if user_query:
            query_line = f"Context Query: {user_query.strip()}\n"
        else:
            query_line = ""

        formatted = (
            f"{header}\n"
            f"{query_line}"
            f"---\n"
            f"{segment.strip()}\n"
            f"---\n"
            f"{footer}"
        )
        return formatted

    def get_checkpoint_details(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Get full details of a specific checkpoint along with all its raw segments."""
        cp = self.db.get_checkpoint(checkpoint_id)
        if not cp:
            return None

        raw_segments = self.db.get_raw_segments_by_checkpoint(checkpoint_id)
        children = self.db.get_child_checkpoints(checkpoint_id)

        return {
            "checkpoint": cp,
            "raw_segments": raw_segments,
            "child_checkpoint_ids": children
        }


# Singleton instance
_retrieval_engine: Optional[RetrievalEngine] = None


def get_retrieval_engine(db_manager: Optional[DatabaseManager] = None) -> RetrievalEngine:
    """Get or create the global retrieval engine instance."""
    global _retrieval_engine
    if _retrieval_engine is None or db_manager is not None:
        _retrieval_engine = RetrievalEngine(db_manager=db_manager)
    return _retrieval_engine
