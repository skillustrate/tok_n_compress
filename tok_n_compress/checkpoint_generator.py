"""
Checkpoint Generation Module for AHMS (Agentic Hybrid Memory System).

Connects the summarization engine to the database and creates Checkpoint objects with hierarchy.
Ensures atomic persistence of the checkpoint, raw conversation segments, and vector embeddings.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set

from .database import DatabaseManager, get_database_manager
from .summarizer import SummarizationEngine, get_summarizer
from .embeddings import get_embedding_engine


class CheckpointGenerator:
    """Generates and persists checkpoints for conversation compression."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        summarizer: Optional[SummarizationEngine] = None
    ):
        self.db = db_manager or get_database_manager()
        self.summarizer = summarizer or get_summarizer()
        self.embedder = get_embedding_engine()

    def generate_checkpoint(
        self,
        segment_content: str,
        timestamp: Optional[str] = None,
        parent_id: Optional[int] = None,
        is_pinned: bool = False,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a checkpoint from a raw conversation segment and persist it atomically
        along with its vector embedding and metadata.

        Args:
            segment_content: The full text of the conversation segment to compress.
            timestamp: ISO format timestamp (defaults to current UTC time).
            parent_id: ID of the previous checkpoint in the conversation hierarchy.
            is_pinned: Whether this checkpoint and segment are protected with pin status.
            embedding: Precomputed embedding vector or None (computed automatically).

        Returns:
            Dictionary containing the checkpoint reference data and database IDs.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Validate parent checkpoint if specified
        if parent_id is not None:
            parent = self.db.get_checkpoint(parent_id)
            if parent is None:
                raise ValueError(f"Parent checkpoint with id {parent_id} does not exist.")

        # Step 1: Summarize the segment
        summary_data = self.summarizer.summarize_segment(segment_content)

        # Step 2: Generate embedding if not supplied
        if embedding is None:
            embedding = self.embedder.embed_text(segment_content)

        # Step 3: Save raw segment, embedding, and checkpoint atomically in SQLite
        checkpoint_id, raw_segment_id = self.db.save_checkpoint_atomic(
            parent_id=parent_id,
            timestamp=timestamp,
            summary=summary_data["summary"],
            files_metadata=summary_data["files_metadata"],
            key_prompts=summary_data["key_prompts"],
            raw_segment_content=segment_content,
            embedding=embedding,
            is_pinned=is_pinned
        )

        # Step 4: Build checkpoint reference object for active context
        checkpoint_ref = {
            "id": checkpoint_id,
            "parent_checkpoint_id": parent_id,
            "timestamp": timestamp,
            "summary": summary_data["summary"],
            "files_metadata": summary_data["files_metadata"],
            "key_prompts": summary_data["key_prompts"],
            "raw_segment_refs": [raw_segment_id],
            "is_pinned": is_pinned
        }

        return checkpoint_ref

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve checkpoint data by ID."""
        return self.db.get_checkpoint(checkpoint_id)

    def get_checkpoint_hierarchy(self, checkpoint_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve a checkpoint and all its ancestors up to the root.

        Returns:
            List of checkpoints ordered from root to the specified leaf.
        """
        hierarchy = []
        current_id: Optional[int] = checkpoint_id
        visited: Set[int] = set()

        while current_id is not None:
            if current_id in visited:
                break  # Cycle protection
            visited.add(current_id)

            cp = self.db.get_checkpoint(current_id)
            if not cp:
                break

            hierarchy.append(cp)
            current_id = cp.get("parent_checkpoint_id")

        return list(reversed(hierarchy))  # Return from root to leaf


# Singleton instance
_generator: Optional[CheckpointGenerator] = None


def get_checkpoint_generator(
    db_manager: Optional[DatabaseManager] = None,
    summarizer: Optional[SummarizationEngine] = None
) -> CheckpointGenerator:
    """Get or create the global checkpoint generator instance."""
    global _generator
    if _generator is None or db_manager is not None or summarizer is not None:
        _generator = CheckpointGenerator(db_manager=db_manager, summarizer=summarizer)
    return _generator
