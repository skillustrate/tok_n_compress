"""Tests for RetrievalEngine memory querying and re-hydration."""

import os
import tempfile
import pytest
from tok_n_compress.database import DatabaseManager
from tok_n_compress.checkpoint_generator import CheckpointGenerator
from tok_n_compress.retrieval import RetrievalEngine


@pytest.fixture
def retrieval_setup():
    temp_dir = tempfile.mkdtemp()
    db = DatabaseManager(os.path.join(temp_dir, "retrieval_test.db"))
    gen = CheckpointGenerator(db_manager=db)
    retrieval = RetrievalEngine(db_manager=db)
    return db, gen, retrieval


def test_query_memory_direct_raw_segment(retrieval_setup):
    """Verify querying memory returns matching raw segment."""
    db, gen, retrieval = retrieval_setup

    gen.generate_checkpoint(
        "[USER]: We are getting error FATAL_KAFKA_DISCONNECT in producer pool."
    )

    matches = retrieval.query_memory("FATAL_KAFKA_DISCONNECT", limit=3)
    assert len(matches) >= 1
    assert "FATAL_KAFKA_DISCONNECT" in matches[0]["content"]
    assert matches[0]["relevance_score"] >= 0.6


def test_rehydrate_formatting(retrieval_setup):
    """Verify rehydration formats output with query context."""
    db, gen, retrieval = retrieval_setup

    raw = "[USER]: What is the secret code?\n[ASSISTANT]: CODE_7721"
    rehydrated = retrieval.rehydrate_segment(raw, user_query="What was the secret code?")

    assert "[REHYDRATED CONVERSATION SEGMENT]" in rehydrated
    assert "[END REHYDRATED SEGMENT]" in rehydrated
    assert "Context Query: What was the secret code?" in rehydrated
    assert "CODE_7721" in rehydrated


def test_get_checkpoint_details(retrieval_setup):
    """Verify retrieving full checkpoint details with raw segments."""
    db, gen, retrieval = retrieval_setup

    cp_ref = gen.generate_checkpoint("Important conversation block.")
    details = retrieval.get_checkpoint_details(cp_ref["id"])

    assert details is not None
    assert details["checkpoint"]["id"] == cp_ref["id"]
    assert len(details["raw_segments"]) == 1
    assert "Important conversation block." in details["raw_segments"][0]
