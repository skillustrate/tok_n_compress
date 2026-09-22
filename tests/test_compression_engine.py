"""Tests for CompressionEngine orchestrator, token calculation, and triggers."""

import os
import tempfile
import pytest
from tok_n_compress.database import DatabaseManager
from tok_n_compress.compression_engine import CompressionEngine


@pytest.fixture
def engine():
    temp_dir = tempfile.mkdtemp()
    db = DatabaseManager(os.path.join(temp_dir, "comp_test.db"))
    return CompressionEngine(db_manager=db)


def test_token_counting(engine):
    """Verify token estimation returns integers."""
    tokens = engine.get_token_count("This is a simple sentence with several words.")
    assert isinstance(tokens, int)
    assert tokens > 0

    assert engine.get_token_count("") == 0


def test_compress_conversation_workflow(engine):
    """Verify compression saves checkpoint and returns structured updated history."""
    messages = [
        {"role": "user", "content": "How do we scale the service across multiple availability zones and handle failovers?"},
        {"role": "assistant", "content": "We should use horizontal pod autoscaling with a minimum of 3 replicas per zone, and set up an AWS Application Load Balancer with health checks on /healthz."},
        {"role": "user", "content": "What about the database connection pooling?"},
        {"role": "assistant", "content": "We will configure PgBouncer in transaction pooling mode with max client connections set to 500."},
        {"role": "user", "content": "Can we also enable SSL encryption on the connection pool?"},
        {"role": "assistant", "content": "Yes, SSL verification is enabled with verify-full mode using our CA bundle."},
        {"role": "user", "content": "Let's test it with load using k6 script test_load.js."},
        {"role": "assistant", "content": "Load test initiated across all availability zones."}
    ]

    result = engine.compress_conversation(messages, keep_recent_n=2)

    assert result["checkpoint_id"] > 0
    assert result["compression_ratio"] >= 1.5
    assert result["tokens_before"] > result["tokens_after"]
    assert len(result["updated_history"]) == 3  # 1 checkpoint message + 2 kept messages
    assert result["updated_history"][0]["role"] == "system"
    assert f"[COMPRESSED CHECKPOINT #{result['checkpoint_id']}]" in result["updated_history"][0]["content"]
    assert result["updated_history"][1]["content"] == "Let's test it with load using k6 script test_load.js."


def test_empty_history_raises_error(engine):
    """Verify attempting to compress empty history raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compress an empty conversation"):
        engine.compress_conversation([])


def test_topic_shift_detection(engine):
    """Verify topic shift sensitivity distinguishes same vs different topics."""
    t1 = "Working on React frontend UI buttons and Tailwind CSS styles."
    t1_cont = "Updating the button hover state and primary theme colors."
    t2 = "Debugging Linux kernel memory page faults in C driver."

    assert engine.detect_topic_shift(t1_cont, t1) is False
    assert engine.detect_topic_shift(t2, t1) is True
