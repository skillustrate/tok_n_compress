"""
Conversation Compressor Skill Interface.

Provides a unified high-level interface for agent and workflow integration,
exposing compression, memory querying, re-hydration, and hierarchy inspection tools.
"""

from typing import Dict, List, Any, Optional, Union

from .database import DatabaseManager, get_database_manager
from .summarizer import SummarizationEngine, get_summarizer
from .checkpoint_generator import CheckpointGenerator, get_checkpoint_generator
from .retrieval import RetrievalEngine, get_retrieval_engine
from .compression_engine import CompressionEngine, get_compression_engine


class ConversationCompressorSkill:
    """Production Skill interface for the Hybrid Memory Model."""

    def __init__(
        self,
        db_path: str = "./checkpoints.db",
        llm_client: Optional[Any] = None,
        token_threshold: int = 80000,
        topic_shift_sensitivity: float = 0.5
    ):
        self.db = get_database_manager(db_path)
        self.summarizer = get_summarizer(llm_client=llm_client)
        self.checkpoint_gen = get_checkpoint_generator(self.db, self.summarizer)
        self.retrieval = get_retrieval_engine(self.db)
        self.compression_engine = get_compression_engine(
            db_manager=self.db,
            summarizer=self.summarizer,
            checkpoint_gen=self.checkpoint_gen,
            retrieval=self.retrieval
        )
        self.compression_engine.token_threshold = token_threshold
        self.compression_engine.topic_shift_sensitivity = topic_shift_sensitivity

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

        Args:
            raw_history: List of conversation messages.
            current_token_count: Optional pre-calculated token count.
            user_query: Current query context.
            parent_id: Optional parent checkpoint ID for explicit branching/chaining.
            keep_recent_n: Number of recent messages to preserve in active context.

        Returns:
            Dictionary containing compression ratio, token savings, and updated history.
        """
        return self.compression_engine.compress_conversation(
            raw_history=raw_history,
            current_token_count=current_token_count,
            user_query=user_query,
            parent_id=parent_id,
            keep_recent_n=keep_recent_n
        )

    def query(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query compressed long-term memory for relevant past conversation segments.

        Args:
            query_text: Search terms, topic, or question.
            limit: Maximum number of relevant segments to return.

        Returns:
            List of matching segments with relevance scores.
        """
        return self.retrieval.query_memory(query_text, limit=limit)

    def rehydrate(
        self,
        target: Union[int, str],
        user_query: Optional[str] = None
    ) -> str:
        """
        Rehydrate a raw conversation segment into active context format.

        Args:
            target: Either a segment ID (int) or raw segment content string.
            user_query: Optional user question providing context for injection.

        Returns:
            Formatted context block ready for inclusion in LLM prompt.
        """
        if isinstance(target, int):
            content = self.db.get_raw_segment(target)
            if not content:
                # Try fetching via checkpoint_id
                content = self.db.get_raw_segment_by_checkpoint(target)
            if not content:
                raise ValueError(f"No conversation content found for ID {target}")
        else:
            content = str(target)

        return self.retrieval.rehydrate_segment(content, user_query=user_query)

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
        topic_shift_detected: bool = False
    ) -> bool:
        """Check if memory compression trigger criteria are satisfied."""
        return self.compression_engine.should_compress(
            current_token_count=current_token_count,
            segment_length=segment_length,
            topic_shift_detected=topic_shift_detected
        )

    def evaluate_quality(self, original_text: str, summary_text: str) -> Dict[str, Any]:
        """Evaluate faithfulness and completeness of a summary against original content."""
        return self.summarizer.evaluate_summary_faithfulness(original_text, summary_text)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the conversation database."""
        all_cps = self.db.get_all_checkpoints(limit=1000)
        total_segments = sum(cp.get("segment_count", 0) for cp in all_cps)
        return {
            "total_checkpoints": len(all_cps),
            "total_raw_segments": total_segments,
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
        # If legacy path argument is passed
        return raw_history_or_path
