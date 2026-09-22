"""Tests for ConversationCompressorSkill top-level interface."""

import os
import tempfile
import pytest
from tok_n_compress.skill import ConversationCompressorSkill, CompressTokenCheckpoint


@pytest.fixture
def skill():
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "skill_test.db")
    return ConversationCompressorSkill(db_path=db_file)


def test_skill_end_to_end(skill):
    """Verify end-to-end skill operations: compress, query, rehydrate, stats."""
    history = [
        {"role": "user", "content": "Deploying build v3.8.1 with hash 8f9b2a."},
        {"role": "assistant", "content": "Build deployed successfully."},
        {"role": "user", "content": "Now checking logs for errors."},
        {"role": "assistant", "content": "No errors detected."}
    ]

    # Compress
    res = skill.compress(history, keep_recent_n=2)
    assert res["checkpoint_id"] > 0
    assert len(res["updated_history"]) == 3

    # Query
    matches = skill.query("8f9b2a", limit=2)
    assert len(matches) >= 1
    assert "8f9b2a" in matches[0]["content"]

    # Rehydrate by ID
    rehydrated = skill.rehydrate(matches[0]["segment_id"], user_query="What was the build hash?")
    assert "8f9b2a" in rehydrated
    assert "[REHYDRATED CONVERSATION SEGMENT]" in rehydrated

    # Stats
    stats = skill.get_stats()
    assert stats["total_checkpoints"] == 1
    assert stats["total_raw_segments"] == 1


def test_legacy_wrapper():
    """Verify backwards compatibility wrapper CompressTokenCheckpoint."""
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "compat_test.db")
    wrapper = CompressTokenCheckpoint({"db_path": db_file})

    history = [
        {"role": "user", "content": "Legacy message 1"},
        {"role": "assistant", "content": "Legacy reply 1"}
    ]
    res = wrapper.compress(history)
    assert "checkpoint_id" in res
