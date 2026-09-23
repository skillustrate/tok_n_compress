"""
Conversation Compressor Skill Interface for AHMS (Agentic Hybrid Memory System).

Provides a unified high-level interface for agent and workflow integration,
exposing dynamic compression, hybrid RRF memory querying, XML re-hydration,
pin protection, and episodic hierarchy inspection.
"""

from typing import Dict, List, Any, Optional, Union

from .database import DatabaseManager, get_database_manager
from .summarizer import SummarizationEngine, get_summarizer
from .checkpoint_generator import CheckpointGenerator, get_checkpoint_generator
from .retrieval import RetrievalEngine, get_retrieval_engine
from .compression_engine import CompressionEngine, get_compression_engine
from .context_window_manager import ContextWindowManager, get_context_window_manager


class ConversationCompressorSkill:
    """Production Skill interface for AHMS v2.0."""

    def __init__(
        self,
        db_path: str = "./checkpoints.db",
        llm_client: Optional[Any] = None,
        model_context_window: int = 32000,
        working_memory_ratio: float = 0.30,
        token_threshold: Optional[int] = None,
        topic_shift_sensitivity: float = 0.5
    ):
        self.db = get_database_manager(db_path)
        self.summarizer = get_summarizer(llm_client=llm_client)
        self.checkpoint_gen = get_checkpoint_generator(self.db, self.summarizer)
        self.retrieval = get_retrieval_engine(self.db)
        self.window_manager = get_context_window_manager(
            model_context_window=model_context_window,
            working_memory_ratio=working_memory_ratio
        )
        self.compression_engine = get_compression_engine(
            db_manager=self.db,
            summarizer=self.summarizer,
            checkpoint_gen=self.checkpoint_gen,
            retrieval=self.retrieval,
            window_manager=self.window_manager
        )
        if token_threshold is not None:
            self.compression_engine.token_threshold = token_threshold
        self.compression_engine.topic_shift_sensitivity = topic_shift_sensitivity

    def configure_model_window(self, context_tokens: int, working_ratio: float = 0.30):
        """Dynamically reconfigure working memory budget according to target model capacity."""
        self.window_manager.model_context_window = context_tokens
        self.window_manager.working_memory_ratio = working_ratio
        self.compression_engine.token_threshold = self.window_manager.eviction_threshold

    def compress(
        self,
        raw_history: List[Dict[str, Any]],
        current_token_count: Optional[int] = None,
        user_query: Optional[str] = None,
        parent_id: Optional[int] = None,
        keep_recent_n: int = 2
    ) -> Dict[str, Any]:
        """
        Compress conversation history into a checkpoint and return updated context.
        Respects pin protection and semantic turn boundaries.
        """
        return self.compression_engine.compress_conversation(
            raw_history=raw_history,
            current_token_count=current_token_count,
            user_query=user_query,
            parent_id=parent_id,
            keep_recent_n=keep_recent_n
        )

    def query(
        self,
        query_text: str,
        limit: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query compressed long-term memory using Hybrid RRF (FTS5 + Vector Cosine).
        """
        return self.retrieval.query_memory(query_text, limit=limit, use_hybrid=use_hybrid)

    def rehydrate(
        self,
        target: Union[int, str],
        user_query: Optional[str] = None,
        format_style: str = "legacy"
    ) -> str:
        """
        Rehydrate a raw conversation segment into active context format (default: XML).
        """
        segment_id = None
        checkpoint_id = None
        timestamp = None

        if isinstance(target, int):
            segment_id = target
            content = self.db.get_raw_segment(target)
            if not content:
                content = self.db.get_raw_segment_by_checkpoint(target)
                checkpoint_id = target
            if not content:
                raise ValueError(f"No conversation content found for ID {target}")
        else:
            content = str(target)

        return self.retrieval.rehydrate_segment(
            segment=content,
            user_query=user_query,
            segment_id=segment_id,
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            format_style=format_style
        )

    def pin_segment(self, segment_id: int, pinned: bool = True) -> bool:
        """Protect or unprotect a segment from automated eviction."""
        return self.db.set_pin_status(segment_id, is_pinned=pinned)

    def get_checkpoint_map(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get the high-level Layer 2 episodic summary map of the conversation.
        Returns a lightweight tree structure of all checkpoints.
        """
        checkpoints = self.db.get_all_checkpoints(limit=limit)
        map_entries = []
        for cp in checkpoints:
            map_entries.append({
                "checkpoint_id": cp["id"],
                "parent_id": cp["parent_checkpoint_id"],
                "timestamp": cp["timestamp"],
                "summary": cp["summary"],
                "files": [f.get("filename") for f in cp.get("files_metadata", []) if isinstance(f, dict)],
                "is_pinned": bool(cp.get("is_pinned", False)),
                "segment_count": cp.get("segment_count", 0)
            })
        return map_entries

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve checkpoint metadata and raw segment references."""
        return self.retrieval.get_checkpoint_details(checkpoint_id)

    def get_hierarchy(self, checkpoint_id: int) -> List[Dict[str, Any]]:
        """Retrieve full ancestral lineage of checkpoints from root to leaf."""
        return self.checkpoint_gen.get_checkpoint_hierarchy(checkpoint_id)

    def get_all_checkpoints(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all stored checkpoints."""
        return self.db.get_all_checkpoints(limit=limit)

    def should_compress(
        self,
        current_token_count: int,
        segment_length: int = 0,
        topic_shift_detected: bool = False,
        active_history: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Check if memory compression trigger criteria are satisfied."""
        return self.compression_engine.should_compress(
            current_token_count=current_token_count,
            segment_length=segment_length,
            topic_shift_detected=topic_shift_detected,
            active_history=active_history
        )

    def evaluate_quality(self, original_text: str, summary_text: str) -> Dict[str, Any]:
        """Evaluate faithfulness and completeness of a summary against original content."""
        return self.summarizer.evaluate_summary_faithfulness(original_text, summary_text)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the conversation database."""
        all_cps = self.db.get_all_checkpoints(limit=1000)
        total_segments = sum(cp.get("segment_count", 0) for cp in all_cps)
        pinned_cps = sum(1 for cp in all_cps if cp.get("is_pinned", False))
        return {
            "total_checkpoints": len(all_cps),
            "pinned_checkpoints": pinned_cps,
            "total_raw_segments": total_segments,
            "model_context_window": self.window_manager.model_context_window,
            "working_memory_budget": self.window_manager.working_memory_budget,
            "eviction_threshold": self.window_manager.eviction_threshold,
            "token_threshold": self.compression_engine.token_threshold,
            "topic_shift_sensitivity": self.compression_engine.topic_shift_sensitivity
        }


# Backwards compatibility class
class CompressTokenCheckpoint(ConversationCompressorSkill):
    """Compatibility wrapper preserving the legacy class name."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        super().__init__(
            db_path=cfg.get("db_path", "./checkpoints.db"),
            model_context_window=cfg.get("model_context_window", 32000),
            token_threshold=cfg.get("token_threshold", 80000),
            topic_shift_sensitivity=cfg.get("topic_shift_sensitivity", 0.5)
        )

    def compress(
        self,
        raw_history_or_path: Any,
        **kwargs
    ) -> Any:
        if isinstance(raw_history_or_path, list):
            return super().compress(raw_history_or_path, **kwargs)
        return raw_history_or_path
