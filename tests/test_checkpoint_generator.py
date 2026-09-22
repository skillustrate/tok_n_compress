"""Tests for CheckpointGenerator hierarchy, atomic storage, and validation."""

import os
import tempfile
import pytest
from tok_n_compress.database import DatabaseManager
from tok_n_compress.summarizer import SummarizationEngine
from tok_n_compress.checkpoint_generator import CheckpointGenerator


@pytest.fixture
def generator():
    temp_dir = tempfile.mkdtemp()
    db = DatabaseManager(os.path.join(temp_dir, "gen_test.db"))
    summarizer = SummarizationEngine()
    return CheckpointGenerator(db_manager=db, summarizer=summarizer)


def test_generate_checkpoint_atomic(generator):
    """Verify generate_checkpoint creates distinct IDs and links raw segment."""
    cp_ref = generator.generate_checkpoint(
        segment_content="[USER]: Need to refactor api router.\n[ASSISTANT]: Router refactored.",
        parent_id=None
    )

    assert cp_ref["id"] > 0
    assert cp_ref["parent_checkpoint_id"] is None
    assert len(cp_ref["raw_segment_refs"]) == 1

    stored = generator.db.get_checkpoint(cp_ref["id"])
    assert stored is not None
    assert stored["id"] == cp_ref["id"]


def test_checkpoint_hierarchy_chain(generator):
    """Verify parent-child hierarchy navigation."""
    cp1 = generator.generate_checkpoint("Step 1", parent_id=None)
    cp2 = generator.generate_checkpoint("Step 2", parent_id=cp1["id"])
    cp3 = generator.generate_checkpoint("Step 3", parent_id=cp2["id"])

    assert cp2["parent_checkpoint_id"] == cp1["id"]
    assert cp3["parent_checkpoint_id"] == cp2["id"]

    hierarchy = generator.get_checkpoint_hierarchy(cp3["id"])
    chain_ids = [cp["id"] for cp in hierarchy]
    assert chain_ids == [cp1["id"], cp2["id"], cp3["id"]]


def test_invalid_parent_raises_error(generator):
    """Verify providing an unrecorded parent_id raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        generator.generate_checkpoint("Orphan step", parent_id=99999)
