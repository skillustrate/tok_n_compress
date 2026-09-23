"""
Retrieval Layer for AHMS (Agentic Hybrid Memory System).

Features:
- Query Expansion for ambiguous, casual, or pronoun-heavy user queries.
- Hybrid Search combining SQLite FTS5 (Keyword) and vector similarity via Reciprocal Rank Fusion (RRF).
- Structured XML-enveloped Context Re-hydration (<retrieved_context>...</retrieved_context>).
"""

import re
from typing import Dict, List, Any, Optional

from .database import DatabaseManager, get_database_manager
from .embeddings import get_embedding_engine, FastLocalEmbedding


class RetrievalEngine:
    """Handles query expansion, hybrid RRF search, and XML context re-hydration."""

    # Common technical synonym expansions
    SYNONYM_MAP = {
        "error": ["error", "exception", "failure", "traceback", "failed", "crash"],
        "bug": ["bug", "defect", "issue", "failure", "fix"],
        "timeout": ["timeout", "timed out", "deadline", "connection timeout"],
        "auth": ["auth", "authentication", "login", "token", "jwt", "credentials"],
        "db": ["db", "database", "sqlite", "query", "schema", "table"],
        "test": ["test", "pytest", "assertion", "mock", "unit test"],
        "deploy": ["deploy", "deployment", "docker", "podman", "container"],
        "api": ["api", "endpoint", "route", "http", "request", "response"]
    }

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedder: Optional[FastLocalEmbedding] = None
    ):
        self.db = db_manager or get_database_manager()
        self.embedder = embedder or get_embedding_engine()

    def expand_query(self, query: str, recent_context: Optional[str] = None) -> str:
        """
        Expand casual or vague queries into richer technical terms using synonym mapping
        and recent contextual signals.
        """
        trimmed = query.strip()
        if not trimmed:
            return ""

        words = re.findall(r"\b[a-zA-Z0-9_\-\./#]+\b", trimmed.lower())
        expanded_terms = list(words)

        for w in words:
            if w in self.SYNONYM_MAP:
                expanded_terms.extend(self.SYNONYM_MAP[w])

        # If query is very short or vague (e.g. "that error"), pull technical clues from recent context
        if len(words) <= 3 and recent_context:
            context_words = re.findall(r"\b[a-zA-Z0-9_\-\./#]{4,}\b", recent_context.lower())
            # Pick up to 3 distinctive technical terms from context
            for cw in context_words[:10]:
                if cw not in expanded_terms:
                    expanded_terms.append(cw)
                    if len(expanded_terms) >= 8:
                        break

        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                deduped.append(term)

        return " ".join(deduped)

    def query_memory(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True,
        expand_vague_query: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search compressed memory using Hybrid RRF (FTS5 + Vector Cosine).

        Args:
            query: The user's search query or conversational question.
            limit: Maximum number of results to return.
            use_hybrid: If True, uses Hybrid Reciprocal Rank Fusion; otherwise keyword search.
            expand_vague_query: If True, automatically expands short or vague queries.

        Returns:
            Ranked list of matching segments with relevance scores and metadata.
        """
        trimmed_query = query.strip()
        if not trimmed_query:
            return []

        search_query = self.expand_query(trimmed_query) if expand_vague_query else trimmed_query

        if use_hybrid:
            q_vec = self.embedder.embed_text(search_query)
            hybrid_results = self.db.search_hybrid_rrf(search_query, query_vector=q_vec, limit=limit)
            if hybrid_results:
                return hybrid_results

        # Fallback / Enrichment via Checkpoints metadata if needed
        return self._search_with_checkpoint_fallback(search_query, limit=limit)

    def _search_with_checkpoint_fallback(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search raw segments with fallback to checkpoint summaries and metadata."""
        query_terms = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 1]
        scored_results: Dict[int, Dict[str, Any]] = {}

        raw_matches = self.db.search_fts(query, limit=limit * 2)
        for match in raw_matches:
            seg_id = match["id"]
            content = match["content"]
            score = self._compute_relevance(content, query, query_terms)
            scored_results[seg_id] = {
                "id": seg_id,
                "segment_id": seg_id,
                "checkpoint_id": match["checkpoint_id"],
                "content": content,
                "timestamp": match.get("timestamp"),
                "relevance_score": score,
                "source": "raw_segment"
            }

        if len(scored_results) < limit:
            all_checkpoints = self.db.get_all_checkpoints(limit=limit * 3)
            for cp in all_checkpoints:
                cp_id = cp["id"]
                raw_segments = self.db.get_raw_segments_by_checkpoint(cp_id)
                if not raw_segments:
                    continue

                summary_score = self._compute_relevance(cp.get("summary", ""), query, query_terms)
                prompts = cp.get("key_prompts", [])
                prompts_text = " ".join(f"{p.get('prompt', '')} {p.get('outcome', '')}" for p in prompts)
                prompts_score = self._compute_relevance(prompts_text, query, query_terms)

                best_cp_score = max(summary_score, prompts_score)
                if best_cp_score > 0.15:
                    combined_content = "\n\n".join(raw_segments)
                    refs = cp.get("raw_segment_refs", [])
                    seg_id = refs[0] if refs else 0
                    if seg_id not in scored_results or best_cp_score > scored_results[seg_id]["relevance_score"]:
                        scored_results[seg_id] = {
                            "id": seg_id,
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
        """Compute basic text relevance score."""
        if not text:
            return 0.0

        lower_text = text.lower()
        score = 0.0

        if full_query.lower() in lower_text:
            score += 0.6

        if query_terms:
            matched_terms = [t for t in query_terms if t in lower_text]
            term_ratio = len(matched_terms) / len(query_terms)
            score += 0.4 * term_ratio

        return min(round(score, 4), 1.0)

    def rehydrate_segment(
        self,
        segment: str,
        user_query: Optional[str] = None,
        segment_id: Optional[int] = None,
        checkpoint_id: Optional[int] = None,
        timestamp: Optional[str] = None,
        relevance_score: Optional[float] = None,
        format_style: str = "legacy"
    ) -> str:
        """
        Format a raw segment into a clean, structured context block for injection into active LLM context.
        
        Supports both modern XML-enveloped format (<retrieved_context>) and legacy text tags.
        """
        content = segment.strip()
        ts_attr = f' timestamp="{timestamp}"' if timestamp else ""
        seg_attr = f' segment_id="{segment_id}"' if segment_id is not None else ""
        cp_attr = f' checkpoint_id="{checkpoint_id}"' if checkpoint_id is not None else ""
        rel_attr = f' relevance="{relevance_score:.2f}"' if relevance_score is not None else ""

        if format_style == "xml":
            query_tag = f"  <query>{user_query.strip()}</query>\n" if user_query else ""
            return (
                f'<retrieved_context{seg_attr}{cp_attr}{ts_attr}{rel_attr}>\n'
                f"{query_tag}"
                f"  <content>\n{content}\n  </content>\n"
                f"</retrieved_context>"
            )

        # Legacy bracketed format
        header = "[REHYDRATED CONVERSATION SEGMENT]"
        footer = "[END REHYDRATED SEGMENT]"
        query_line = f"Context Query: {user_query.strip()}\n" if user_query else ""
        return (
            f"{header}\n"
            f"{query_line}"
            f"---\n"
            f"{content}\n"
            f"---\n"
            f"{footer}"
        )

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
