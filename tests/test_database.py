"""Tests for DatabaseManager persistence, atomic transactions, and search."""

import os
import tempfile
import pytest
from tok_n_compress.database import DatabaseManager


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_checkpoints.db")
    db = DatabaseManager(db_file)
    yield db


def test_table_creation(temp_db):
    """Verify that tables and indexes are created successfully."""
    with temp_db._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "checkpoints" in tables
        assert "raw_segments" in tables


def test_atomic_save(temp_db):
    """Verify atomic transaction saves both checkpoint and raw segment together."""
    cp_id, seg_id = temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-22T10:00:00Z",
        summary="Test segment summary",
        files_metadata=[{"filename": "app.py", "summary": "main entry point"}],
        key_prompts=[{"prompt": "deploy app", "outcome": "deployed"}],
        raw_segment_content="[USER]: deploy app\n[ASSISTANT]: deployed"
    )

    assert cp_id > 0
    assert seg_id > 0

    cp = temp_db.get_checkpoint(cp_id)
    assert cp is not None
    assert cp["summary"] == "Test segment summary"
    assert cp["files_metadata"] == [{"filename": "app.py", "summary": "main entry point"}]
    assert cp["raw_segment_refs"] == [seg_id]

    seg = temp_db.get_raw_segment(seg_id)
    assert seg == "[USER]: deploy app\n[ASSISTANT]: deployed"


def test_atomic_rollback_on_failure(temp_db):
    """Simulate failure to verify atomic rollback leaves no orphaned records."""
    # Try invalid foreign key insert or malformed state
    with pytest.raises(Exception):
        # Passing an invalid type that fails database execute will trigger rollback
        temp_db.save_checkpoint_atomic(
            parent_id=None,
            timestamp=None,  # NOT NULL constraint violation
            summary="Invalid",
            files_metadata=[],
            key_prompts=[],
            raw_segment_content="Should rollback"
        )

    all_cps = temp_db.get_all_checkpoints()
    assert len(all_cps) == 0


def test_get_all_checkpoints_columns(temp_db):
    """Verify get_all_checkpoints returns all columns without KeyError."""
    temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-22T10:00:00Z",
        summary="Summary 1",
        files_metadata=[{"filename": "test.txt", "summary": "docs"}],
        key_prompts=[{"prompt": "p1", "outcome": "o1"}],
        raw_segment_content="segment text"
    )

    all_cps = temp_db.get_all_checkpoints()
    assert len(all_cps) == 1
    cp = all_cps[0]
    assert "files_metadata" in cp
    assert "key_prompts" in cp
    assert "raw_segment_refs" in cp
    assert cp["segment_count"] == 1


def test_cascading_delete(temp_db):
    """Verify that deleting a checkpoint cascades to delete its raw segments."""
    cp_id, seg_id = temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-22T10:00:00Z",
        summary="Summary to delete",
        files_metadata=[],
        key_prompts=[],
        raw_segment_content="Raw text to cascade delete"
    )

    assert temp_db.get_raw_segment(seg_id) is not None
    deleted = temp_db.delete_checkpoint(cp_id)
    assert deleted is True

    assert temp_db.get_checkpoint(cp_id) is None
    assert temp_db.get_raw_segment(seg_id) is None


def test_search_raw_segments(temp_db):
    """Verify keyword and phrase search in raw segments."""
    temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-22T10:00:00Z",
        summary="DB work",
        files_metadata=[],
        key_prompts=[],
        raw_segment_content="Fixed deadlock in postgres migration worker"
    )
    temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-22T11:00:00Z",
        summary="Frontend",
        files_metadata=[],
        key_prompts=[],
        raw_segment_content="Updated CSS styles for modal popup"
    )

    results = temp_db.search_raw_segments("postgres migration")
    assert len(results) >= 1
    assert "deadlock" in results[0]["content"]

    no_match = temp_db.search_raw_segments("kubernetes helm")
    assert len(no_match) == 0
