"""
Comprehensive tests for AHMS v2.0 (Agentic Hybrid Memory System).

Tests:
1. FastLocalEmbedding & Cosine Distance calculations.
2. SQLite unified persistence with FTS5 and vector cosine distance.
3. Hybrid Search with Reciprocal Rank Fusion (RRF).
4. Dynamic Token Budgeting via ContextWindowManager.
5. Pin Protection (shielding critical turns from eviction).
6. Large output & log collapsing.
7. Semantic boundary snapping during eviction.
8. Query expansion for vague queries.
9. XML-enveloped Context Re-hydration.
10. Layer 2 Episodic Checkpoint Map.
11. MCP Server v2.0 tools and resources.
"""

import json
import pytest
from pathlib import Path

from tok_n_compress.embeddings import FastLocalEmbedding, get_embedding_engine
from tok_n_compress.database import DatabaseManager
from tok_n_compress.context_window_manager import ContextWindowManager
from tok_n_compress.retrieval import RetrievalEngine
from tok_n_compress.skill import ConversationCompressorSkill
from tok_n_compress.mcp_server import MCPServer


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "ahms_v2_test.db"
    manager = DatabaseManager(str(db_file))
    yield manager
    manager.clear_all()


# 1. Embeddings and Cosine Distance
def test_fast_local_embedding_properties():
    embedder = FastLocalEmbedding()
    vec1 = embedder.embed_text("PostgreSQL connection timeout error")
    vec2 = embedder.embed_text("Database connection timeout exception")
    vec3 = embedder.embed_text("Baking strawberry cheesecake dessert")

    assert len(vec1) == FastLocalEmbedding.DIMENSION
    assert len(vec2) == FastLocalEmbedding.DIMENSION

    # Serialization roundtrip
    packed = FastLocalEmbedding.serialize_vector(vec1)
    unpacked = FastLocalEmbedding.deserialize_vector(packed)
    assert len(unpacked) == len(vec1)
    assert pytest.approx(vec1[0], abs=1e-4) == unpacked[0]

    # Similar text should have smaller cosine distance than unrelated text
    dist_similar = FastLocalEmbedding.cosine_distance(vec1, vec2)
    dist_unrelated = FastLocalEmbedding.cosine_distance(vec1, vec3)
    assert dist_similar < dist_unrelated


# 2. SQLite FTS5 and Vector Search
def test_sqlite_fts5_and_vector_persistence(temp_db):
    embedder = get_embedding_engine()
    vec1 = embedder.embed_text("Kubernetes ingress deployment failure with 502 bad gateway")
    vec2 = embedder.embed_text("React frontend button styling with Tailwind CSS")

    cp_id, seg1_id = temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-23T10:00:00Z",
        summary="Kubernetes ingress troubleshooting",
        files_metadata=[{"filename": "ingress.yaml"}],
        key_prompts=[{"prompt": "Fix 502", "outcome": "Resolved"}],
        raw_segment_content="Kubernetes ingress deployment failure with 502 bad gateway",
        embedding=vec1
    )

    _, seg2_id = temp_db.save_checkpoint_atomic(
        parent_id=cp_id,
        timestamp="2026-09-23T10:15:00Z",
        summary="React frontend button update",
        files_metadata=[{"filename": "Button.tsx"}],
        key_prompts=[{"prompt": "Style button", "outcome": "Styled"}],
        raw_segment_content="React frontend button styling with Tailwind CSS",
        embedding=vec2
    )

    # Test FTS5 exact search
    fts_results = temp_db.search_fts("Kubernetes ingress")
    assert len(fts_results) >= 1
    assert fts_results[0]["id"] == seg1_id
    assert "Kubernetes" in fts_results[0]["content"]

    # Test Vector cosine similarity inside SQLite
    query_vec = embedder.embed_text("k8s gateway crash")
    vec_results = temp_db.search_vector(query_vec, limit=2)
    assert len(vec_results) == 2
    assert vec_results[0]["id"] == seg1_id  # Closest semantic match


# 3. Hybrid Search with Reciprocal Rank Fusion (RRF)
def test_hybrid_rrf_search(temp_db):
    embedder = get_embedding_engine()
    temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-23T10:00:00Z",
        summary="User Authentication Setup",
        files_metadata=[{"filename": "auth.py"}],
        key_prompts=[],
        raw_segment_content="Implemented JWT token verification and OAuth2 login flow",
        embedding=embedder.embed_text("Implemented JWT token verification and OAuth2 login flow")
    )

    temp_db.save_checkpoint_atomic(
        parent_id=None,
        timestamp="2026-09-23T10:10:00Z",
        summary="Payment Gateway Setup",
        files_metadata=[{"filename": "stripe.py"}],
        key_prompts=[],
        raw_segment_content="Configured Stripe checkout webhook handlers",
        embedding=embedder.embed_text("Configured Stripe checkout webhook handlers")
    )

    rrf_matches = temp_db.search_hybrid_rrf("OAuth login token", limit=2)
    assert len(rrf_matches) >= 1
    assert "JWT" in rrf_matches[0]["content"]
    assert "rrf_score" in rrf_matches[0]
    assert rrf_matches[0]["rrf_score"] > 0


# 4. Dynamic Token Budgeting via ContextWindowManager
def test_dynamic_token_budgeting():
    # 8k model
    mgr_8k = ContextWindowManager(model_context_window=8000, working_memory_ratio=0.30)
    assert mgr_8k.working_memory_budget == 2400
    assert mgr_8k.eviction_threshold == int(2400 * 0.70)

    # 128k model
    mgr_128k = ContextWindowManager(model_context_window=128000, working_memory_ratio=0.30)
    assert mgr_128k.working_memory_budget == 38400
    assert mgr_128k.eviction_threshold == int(38400 * 0.70)


# 5. Pin Protection (shielding critical turns)
def test_pin_protection(temp_db):
    skill = ConversationCompressorSkill(db_path=str(temp_db.db_path))

    history = [
        {"role": "system", "content": "CORE ARCHITECTURE: Use FastAPI and SQLite.", "pinned": True},
        {"role": "user", "content": "How do we structure endpoints?"},
        {"role": "assistant", "content": "Create routers in api/ directory."},
        {"role": "user", "content": "Now add user authentication."},
        {"role": "assistant", "content": "Added auth router with JWT."}
    ]

    res = skill.compress(history, keep_recent_n=2)
    updated = res["updated_history"]

    # Verify pinned turn is preserved in updated history
    pinned_turns = [t for t in updated if t.get("pinned") or "CORE ARCHITECTURE" in t.get("content", "")]
    assert len(pinned_turns) == 1
    assert "CORE ARCHITECTURE" in pinned_turns[0]["content"]


# 6. Large Output / Log Collapsing
def test_log_collapsing():
    mgr = ContextWindowManager(max_log_lines=20)
    long_log = "\n".join([f"Line {i}: test runner execution trace" for i in range(50)])
    collapsed, was_collapsed, raw = mgr.collapse_large_outputs(long_log)

    assert was_collapsed is True
    assert "[COLLAPSED LOG/PAYLOAD: 50 lines." in collapsed
    assert raw == long_log


# 7. Semantic Boundary Snapping
def test_semantic_boundary_snapping():
    mgr = ContextWindowManager()
    history = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Turn 2"},
        {"role": "tool", "content": "Turn 3"},
        {"role": "user", "content": "Turn 4"},
        {"role": "assistant", "content": "Turn 5"}
    ]
    # keep_recent_n = 2 means Turns 4 and 5 are preserved; candidate turns 1, 2, 3 end on 'tool'
    to_evict, to_preserve = mgr.partition_eviction_candidates(history, keep_recent_n=2)
    assert len(to_evict) == 3
    assert to_evict[-1]["role"] == "tool"
    assert len(to_preserve) == 2


# 8. Query Expansion for Vague Queries
def test_query_expansion():
    retrieval = RetrievalEngine()
    expanded = retrieval.expand_query("that error", recent_context="PostgreSQL database connection refused")
    assert "error" in expanded
    assert "exception" in expanded or "failure" in expanded
    assert "postgresql" in expanded or "database" in expanded


# 9. XML-Enveloped Re-hydration
def test_xml_rehydration():
    retrieval = RetrievalEngine()
    raw = "[USER]: What is the DB schema?\n[ASSISTANT]: CREATE TABLE users(id INT);"
    xml_out = retrieval.rehydrate_segment(
        raw,
        user_query="DB schema?",
        segment_id=42,
        checkpoint_id=1,
        timestamp="2026-09-23T10:00:00Z",
        relevance_score=0.95,
        format_style="xml"
    )

    assert '<retrieved_context segment_id="42" checkpoint_id="1"' in xml_out
    assert 'relevance="0.95"' in xml_out
    assert '<query>DB schema?</query>' in xml_out
    assert 'CREATE TABLE users' in xml_out
    assert '</retrieved_context>' in xml_out


# 10. Layer 2 Checkpoint Map
def test_checkpoint_map(temp_db):
    skill = ConversationCompressorSkill(db_path=str(temp_db.db_path))
    skill.compress([
        {"role": "user", "content": "Editing config.py and main.py"},
        {"role": "assistant", "content": "Updated config settings."}
    ])

    chk_map = skill.get_checkpoint_map()
    assert len(chk_map) == 1
    assert "checkpoint_id" in chk_map[0]
    assert "summary" in chk_map[0]


# 11. MCP Server v2.0 Protocol, Tools, and Resources
def test_mcp_server_v2_features(temp_db):
    server = MCPServer(db_path=str(temp_db.db_path))

    # Test initialize exposes version 2.0.0 and resources capability
    init_res = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_res["result"]["serverInfo"]["version"] == "2.0.0"
    assert "resources" in init_res["result"]["capabilities"]

    # Test tools list includes get_checkpoint_map and pin_segment
    tools_res = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in tools_res["result"]["tools"]]
    assert "get_checkpoint_map" in names
    assert "pin_segment" in names

    # Test resources list and read
    res_list = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}})
    assert res_list["result"]["resources"][0]["uri"] == "memory://checkpoint-map"

    res_read = server.handle_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "resources/read",
        "params": {"uri": "memory://checkpoint-map"}
    })
    assert "contents" in res_read["result"]
    assert res_read["result"]["contents"][0]["uri"] == "memory://checkpoint-map"
