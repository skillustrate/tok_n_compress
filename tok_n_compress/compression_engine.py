"""
Compression Engine for AHMS (Agentic Hybrid Memory System).

Orchestrates conversation compression with:
- Dynamic sliding percentage budgeting via ContextWindowManager.
- Pin Protection (preserves system prompts, schemas, and pinned turns).
- Semantic boundary snapping.
- Large log and traceback collapsing.
- Atomic dual-persistence (SQLite checkpoints + vector embeddings).
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .database import DatabaseManager, get_database_manager
from .summarizer import SummarizationEngine, get_summarizer
from .checkpoint_generator import CheckpointGenerator, get_checkpoint_generator
from .retrieval import RetrievalEngine, get_retrieval_engine
from .context_window_manager import ContextWindowManager, get_context_window_manager


class CompressionEngine:
    """Orchestrates conversation compression based on triggers, dynamic budgets, and pin protection."""

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "out", "off", "over", "under",
        "then", "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
        "should", "now", "is", "was", "are", "were", "be", "been", "have", "has", "had"
    }

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        summarizer: Optional[SummarizationEngine] = None,
        checkpoint_gen: Optional[CheckpointGenerator] = None,
        retrieval: Optional[RetrievalEngine] = None,
        window_manager: Optional[ContextWindowManager] = None,
        token_threshold: Optional[int] = None,
        topic_shift_sensitivity: Optional[float] = None,
        model_context_window: int = 32000
    ):
        self.db = db_manager or get_database_manager()
        self.summarizer = summarizer or get_summarizer()
        self.checkpoint_gen = checkpoint_gen or get_checkpoint_generator(self.db, self.summarizer)
        self.retrieval = retrieval or get_retrieval_engine(self.db)
        self.window_manager = window_manager or get_context_window_manager(model_context_window=model_context_window)

        # Thresholds
        env_threshold = os.environ.get("TOKEN_THRESHOLD")
        self.token_threshold = token_threshold or (int(env_threshold) if env_threshold else self.window_manager.eviction_threshold)

        env_shift = os.environ.get("TOPIC_SHIFT_SENSITIVITY")
        self.topic_shift_sensitivity = topic_shift_sensitivity or (float(env_shift) if env_shift else 0.5)

    def get_token_count(self, text: str) -> int:
        """Estimate token count for a string using standard word/char heuristics."""
        return self.window_manager.estimate_tokens(text)

    def count_history_tokens(self, history: List[Dict[str, Any]]) -> int:
        """Calculate total tokens in a list of conversation messages."""
        return self.window_manager.count_turns_tokens(history)

    def detect_topic_shift(
        self,
        current_segment: str,
        previous_summary: Optional[str] = None
    ) -> bool:
        """
        Detect if a significant topic shift has occurred compared to the previous context.
        """
        if not previous_summary or not current_segment:
            return False

        def stem_word(w: str) -> str:
            w = w.lower()
            for suffix in ("ing", "ed", "es", "s", "ment", "tion", "ation", "able", "ible", "ive", "al", "er", "or", "ly"):
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    return w[:-len(suffix)]
            return w

        def extract_roots(text: str) -> set:
            tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
            return {stem_word(t) for t in tokens if t not in self.STOP_WORDS}

        curr_roots = extract_roots(current_segment)
        prev_roots = extract_roots(previous_summary)

        if not curr_roots or not prev_roots:
            return False

        matches = 0
        for w1 in curr_roots:
            if any(w1 in w2 or w2 in w1 or (len(w1) >= 4 and len(w2) >= 4 and w1[:4] == w2[:4]) for w2 in prev_roots):
                matches += 1

        continuity = min(matches / 2.0, 1.0)
        dissimilarity = 1.0 - continuity
        return dissimilarity > self.topic_shift_sensitivity

    def should_compress(
        self,
        current_token_count: int,
        segment_length: int = 0,
        topic_shift_detected: bool = False,
        active_history: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Determine whether compression should be triggered dynamically."""
        if active_history:
            dynamic_trigger = self.window_manager.should_trigger_eviction(active_history)
        else:
            dynamic_trigger = current_token_count >= self.token_threshold

        topic_trigger = topic_shift_detected and (segment_length > 300)
        return bool(dynamic_trigger or topic_trigger)

    def compress_conversation(
        self,
        raw_history: List[Dict[str, Any]],
        current_token_count: Optional[int] = None,
        user_query: Optional[str] = None,
        parent_id: Optional[int] = None,
        keep_recent_n: int = 2
    ) -> Dict[str, Any]:
        """
        Compress conversation history using dynamic budgeting, pin protection,
        and semantic boundary snapping.
        """
        if not raw_history:
            raise ValueError("Cannot compress an empty conversation history.")

        # If parent_id is not specified, auto-link to the most recent checkpoint if one exists
        if parent_id is None:
            recent_cps = self.db.get_all_checkpoints(limit=1)
            if recent_cps:
                parent_id = recent_cps[0]["id"]

        # Partition unpinned candidate turns from pinned and recent turns
        to_compress, to_preserve = self.window_manager.partition_eviction_candidates(
            raw_history,
            keep_recent_n=keep_recent_n
        )

        # Fallback if no unpinned candidate turns exist
        if not to_compress:
            return {
                "checkpoint_id": None,
                "parent_checkpoint_id": parent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_segment_length": 0,
                "compressed_summary_length": 0,
                "compression_ratio": 1.0,
                "tokens_before": current_token_count or self.count_history_tokens(raw_history),
                "tokens_after": current_token_count or self.count_history_tokens(raw_history),
                "tokens_saved": 0,
                "updated_history": raw_history,
                "retrieval_available": True,
                "message": "All historical turns are pinned or protected."
            }

        raw_segment_content = self._join_segment(to_compress)

        # Log & Payload Collapsing: collapse giant tracebacks or logs
        collapsed_content, was_collapsed, _ = self.window_manager.collapse_large_outputs(raw_segment_content)

        compressed_tokens = self.count_history_tokens(to_compress)
        total_tokens_before = current_token_count or (compressed_tokens + self.count_history_tokens(to_preserve))

        # Generate and atomically persist checkpoint and vector embeddings
        timestamp = datetime.now(timezone.utc).isoformat()
        checkpoint_ref = self.checkpoint_gen.generate_checkpoint(
            segment_content=raw_segment_content,  # Save full raw content to Layer 3
            timestamp=timestamp,
            parent_id=parent_id
        )

        # Update active conversation history
        updated_history = self._update_active_history(
            checkpoint_ref=checkpoint_ref,
            preserved_messages=to_preserve,
            original_tokens=compressed_tokens
        )

        tokens_after = self.count_history_tokens(updated_history)
        raw_len = max(len(raw_segment_content), 1)
        summary_len = max(len(checkpoint_ref["summary"]), 1)
        ratio = round(raw_len / summary_len, 2)

        return {
            "checkpoint_id": checkpoint_ref["id"],
            "parent_checkpoint_id": checkpoint_ref["parent_checkpoint_id"],
            "timestamp": timestamp,
            "raw_segment_length": raw_len,
            "compressed_summary_length": summary_len,
            "compression_ratio": ratio,
            "tokens_before": int(total_tokens_before),
            "tokens_after": int(tokens_after),
            "tokens_saved": max(int(total_tokens_before - tokens_after), 0),
            "updated_history": updated_history,
            "retrieval_available": True,
            "logs_collapsed": was_collapsed
        }

    def _join_segment(self, history: List[Dict[str, Any]]) -> str:
        """Format a list of message dicts into a unified conversation transcript."""
        lines = []
        for msg in history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "").strip()
            ts = msg.get("timestamp", "")
            time_tag = f" {ts}:" if ts else ":"
            lines.append(f"[{role}]{time_tag} {content}")
        return "\n\n".join(lines)

    def _update_active_history(
        self,
        checkpoint_ref: Dict[str, Any],
        preserved_messages: List[Dict[str, Any]],
        original_tokens: int
    ) -> List[Dict[str, Any]]:
        """Construct new active conversation history with the checkpoint reference and preserved turns."""
        files = checkpoint_ref.get("files_metadata", [])
        files_str = f"\nFiles: {', '.join(f.get('filename', '') for f in files)}" if files else ""

        checkpoint_msg = {
            "role": "system",
            "content": (
                f"[COMPRESSED CHECKPOINT #{checkpoint_ref['id']}]\n"
                f"Summary: {checkpoint_ref['summary']}"
                f"{files_str}\n"
                f"(Compressed ~{original_tokens:,} tokens into memory. "
                f"Use memory retrieval tool to query details from this segment.)"
            ),
            "checkpoint_id": checkpoint_ref["id"],
            "timestamp": checkpoint_ref["timestamp"]
        }

        return [checkpoint_msg] + preserved_messages


# Singleton instance
_compression_engine: Optional[CompressionEngine] = None


def get_compression_engine(
    db_manager: Optional[DatabaseManager] = None,
    summarizer: Optional[SummarizationEngine] = None,
    checkpoint_gen: Optional[CheckpointGenerator] = None,
    retrieval: Optional[RetrievalEngine] = None,
    window_manager: Optional[ContextWindowManager] = None
) -> CompressionEngine:
    """Get or create the global compression engine instance."""
    global _compression_engine
    if _compression_engine is None or any(arg is not None for arg in (db_manager, summarizer, checkpoint_gen, retrieval, window_manager)):
        _compression_engine = CompressionEngine(
            db_manager=db_manager,
            summarizer=summarizer,
            checkpoint_gen=checkpoint_gen,
            retrieval=retrieval,
            window_manager=window_manager
        )
    return _compression_engine
