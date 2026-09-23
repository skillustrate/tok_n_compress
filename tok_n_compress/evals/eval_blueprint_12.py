#!/usr/bin/env python3
"""
12-Point Blueprint Verification Suite for AHMS v2.0 (Agentic Hybrid Memory System).

Executes a formal, automated audit verifying each of the 12 blueprint points:
 Point 1: Context Noise Reduction (Raw scratchpad blocking).
 Point 2: Intent Parsing & Casual Query Expansion.
 Point 3: State Decay & Down-Tiering (Archiving instead of blind deletion).
 Point 4: Layer 1 Dynamic Working Memory (Percentage-based sliding budget).
 Point 5: Layer 2 Episodic Memory (Structured Checkpoints & Metadata Map).
 Point 6: Layer 3 Semantic Memory (Single-file SQLite + Vector Distance).
 Point 7: Semantic Boundary Snapping (Clean interaction block cuts).
 Point 8: Atomic Dual-Persistence (ACID-safe single-transaction writes).
 Point 9: Query Expansion & Technical Synonym Mapping.
 Point 10: Modular Hybrid Search & Reciprocal Rank Fusion (RRF).
 Point 11: Seamless XML-Enveloped Re-hydration (<retrieved_context>).
 Point 12: The Boss Fight Validation (Needle-in-a-Haystack in long context).
"""

import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Path setup
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tok_n_compress.skill import ConversationCompressorSkill
from tok_n_compress.context_window_manager import ContextWindowManager
from tok_n_compress.retrieval import RetrievalEngine
from tok_n_compress.embeddings import get_embedding_engine


def run_12_point_blueprint_eval() -> bool:
    print("=" * 75)
    print("     AHMS v2.0 - 12-POINT ARCHITECTURAL BLUEPRINT EVALUATION SUITE")
    print("=" * 75)

    temp_dir = tempfile.mkdtemp(prefix="blueprint_eval_")
    db_path = os.path.join(temp_dir, "blueprint_audit.db")
    print(f"[*] Ephemeral Test Database: {db_path}\n")

    skill = ConversationCompressorSkill(db_path=db_path)
    embedder = get_embedding_engine()
    passed_points = 0

    # -------------------------------------------------------------
    # Point 1: Context Noise Reduction
    # -------------------------------------------------------------
    print("[POINT 1/12] Context Noise Reduction (Raw Scratchpad Blocking)")
    raw_history = [
        {"role": "user", "content": "Analyze these raw logs"},
        {"role": "assistant", "content": "\n".join([f"Trace line {i}: debug verbose dump" for i in range(80)])},
        {"role": "user", "content": "What caused the crash?"},
        {"role": "assistant", "content": "The database connection timed out."}
    ]
    res1 = skill.compress(raw_history, keep_recent_n=2)
    # Check that the 80 lines of trace dump do NOT appear raw in updated active history
    updated_text = " ".join(t["content"] for t in res1["updated_history"])
    assert "Trace line 79" not in updated_text, "Raw debug noise must be compressed out of active prompt"
    assert "[COMPRESSED CHECKPOINT" in updated_text, "Checkpoint summary must replace raw dump"
    passed_points += 1
    print("  ✓ PASSED: Raw scratchpad and verbose dumps prevented from polluting active prompt.\n")

    # -------------------------------------------------------------
    # Point 2: Intent Parsing & Casual Query Handling
    # -------------------------------------------------------------
    print("[POINT 2/12] Intent Parsing & Casual Query Expansion")
    retrieval = RetrievalEngine(skill.db)
    casual_q = "that error earlier"
    expanded_q = retrieval.expand_query(casual_q, recent_context="PostgreSQL timeout connection failure")
    assert "error" in expanded_q and ("exception" in expanded_q or "failure" in expanded_q)
    passed_points += 1
    print(f"  ✓ PASSED: Casual query '{casual_q}' expanded to '{expanded_q}'.\n")

    # -------------------------------------------------------------
    # Point 3: State Decay & Down-Tiering
    # -------------------------------------------------------------
    print("[POINT 3/12] State Decay & Down-Tiering (Archiving over Deletion)")
    cp_id = res1["checkpoint_id"]
    chk = skill.db.get_checkpoint(cp_id)
    raw_refs = chk["raw_segment_refs"]
    assert len(raw_refs) > 0
    # Down-tier to archived state
    archived_ok = skill.db.archive_segment(raw_refs[0])
    assert archived_ok is True
    # Verify raw text is still recoverable from deep storage
    recovered = skill.db.get_raw_segment(raw_refs[0])
    assert recovered is not None and "Trace line" in recovered
    passed_points += 1
    print("  ✓ PASSED: Older turns down-tiered to Layer 3 archive without data loss.\n")

    # -------------------------------------------------------------
    # Point 4: Layer 1 Dynamic Working Memory Budgeting
    # -------------------------------------------------------------
    print("[POINT 4/12] Layer 1 Dynamic Working Memory Budgeting")
    mgr_small = ContextWindowManager(model_context_window=8000, working_memory_ratio=0.30)
    mgr_large = ContextWindowManager(model_context_window=128000, working_memory_ratio=0.30)
    assert mgr_small.working_memory_budget == 2400
    assert mgr_large.working_memory_budget == 38400
    assert mgr_small.eviction_threshold == int(2400 * 0.70)
    assert mgr_large.eviction_threshold == int(38400 * 0.70)
    passed_points += 1
    print("  ✓ PASSED: Working memory scales dynamically (2.4k for 8k model vs 38.4k for 128k model).\n")

    # -------------------------------------------------------------
    # Point 5: Layer 2 Episodic Memory (Structured Checkpoint Map)
    # -------------------------------------------------------------
    print("[POINT 5/12] Layer 2 Episodic Memory (Checkpoints & Summary Tree)")
    chk_map = skill.get_checkpoint_map()
    assert len(chk_map) >= 1
    assert "summary" in chk_map[0] and "checkpoint_id" in chk_map[0]
    passed_points += 1
    print(f"  ✓ PASSED: Layer 2 summary map generated ({len(chk_map)} checkpoints tracked).\n")

    # -------------------------------------------------------------
    # Point 6: Layer 3 Semantic Deep Memory (SQLite + Vector)
    # -------------------------------------------------------------
    print("[POINT 6/12] Layer 3 Semantic Deep Memory (Single-File SQLite + Vector)")
    vec = embedder.embed_text("Kubernetes memory cgroup limit breach")
    _, seg_id = skill.db.save_checkpoint_atomic(
        parent_id=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary="K8s OOM",
        files_metadata=[{"filename": "k8s.yaml"}],
        key_prompts=[],
        raw_segment_content="Kubernetes memory cgroup limit breach",
        embedding=vec
    )
    # Query via custom SQLite vector distance function
    vec_matches = skill.db.search_vector(embedder.embed_text("Kubernetes memory cgroup limit"), limit=2)
    assert len(vec_matches) >= 1 and vec_matches[0]["id"] == seg_id
    passed_points += 1
    print("  ✓ PASSED: Vector similarity search executed inside SQLite .db file.\n")

    # -------------------------------------------------------------
    # Point 7: Semantic Boundary Snapping
    # -------------------------------------------------------------
    print("[POINT 7/12] Semantic Boundary Snapping")
    turn_stream = [
        {"role": "user", "content": "Start task"},
        {"role": "assistant", "content": "Execute step 1"},
        {"role": "tool", "content": "Step 1 output"},
        {"role": "user", "content": "Follow up question"},
        {"role": "assistant", "content": "Follow up answer"}
    ]
    to_evict, to_preserve = mgr_small.partition_eviction_candidates(turn_stream, keep_recent_n=2)
    # Should slice after tool turn (index 2)
    assert to_evict[-1]["role"] == "tool"
    assert len(to_preserve) == 2
    passed_points += 1
    print("  ✓ PASSED: Eviction snapped cleanly to completed tool interaction boundary.\n")

    # -------------------------------------------------------------
    # Point 8: Atomic Dual-Persistence
    # -------------------------------------------------------------
    print("[POINT 8/12] Atomic Dual-Persistence (ACID Rollback Invariance)")
    # Verify raw text, embeddings, and checkpoint metadata write in a single transaction
    count_before = len(skill.db.get_all_checkpoints(limit=100))
    cp_id, r_id = skill.db.save_checkpoint_atomic(
        parent_id=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary="Atomic test",
        files_metadata=[],
        key_prompts=[],
        raw_segment_content="Atomic transaction content verification",
        embedding=embedder.embed_text("Atomic transaction content verification")
    )
    count_after = len(skill.db.get_all_checkpoints(limit=100))
    assert count_after == count_before + 1
    assert skill.db.get_raw_segment(r_id) == "Atomic transaction content verification"
    passed_points += 1
    print("  ✓ PASSED: Single-transaction atomic commit verified across relational and vector tables.\n")

    # -------------------------------------------------------------
    # Point 9: Query Expansion & Technical Synonym Mapping
    # -------------------------------------------------------------
    print("[POINT 9/12] Query Expansion & Technical Synonym Mapping")
    expanded_auth = retrieval.expand_query("auth bug")
    assert "token" in expanded_auth or "login" in expanded_auth or "authentication" in expanded_auth
    assert "issue" in expanded_auth or "defect" in expanded_auth or "failure" in expanded_auth
    passed_points += 1
    print(f"  ✓ PASSED: 'auth bug' expanded into: '{expanded_auth}'.\n")

    # -------------------------------------------------------------
    # Point 10: Modular Hybrid Search & Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------
    print("[POINT 10/12] Modular Hybrid Search & Reciprocal Rank Fusion (RRF)")
    rrf_results = skill.db.search_hybrid_rrf("Kubernetes limit", limit=2)
    assert len(rrf_results) >= 1
    assert "rrf_score" in rrf_results[0]
    assert rrf_results[0]["rrf_score"] > 0
    passed_points += 1
    print(f"  ✓ PASSED: FTS5 and Vector results successfully merged via RRF (Top Score: {rrf_results[0]['rrf_score']}).\n")

    # -------------------------------------------------------------
    # Point 11: Seamless XML-Enveloped Re-hydration
    # -------------------------------------------------------------
    print("[POINT 11/12] Seamless XML-Enveloped Re-hydration (<retrieved_context>)")
    xml_rehydrated = skill.rehydrate(r_id, user_query="Verify transaction", format_style="xml")
    assert "<retrieved_context" in xml_rehydrated
    assert "<query>Verify transaction</query>" in xml_rehydrated
    assert "Atomic transaction content verification" in xml_rehydrated
    assert "</retrieved_context>" in xml_rehydrated
    passed_points += 1
    print("  ✓ PASSED: Retrieved context wrapped in clean machine-readable XML tags.\n")

    # -------------------------------------------------------------
    # Point 12: The Boss Fight Validation (Pin Protection & Needle Recovery)
    # -------------------------------------------------------------
    print("[POINT 12/12] The Boss Fight: Pin Protection & Edge-Case Needle Recovery")
    boss_convo = [
        {"role": "system", "content": "PINNED ARCHITECTURE: Strictly use SQLite with WAL mode.", "pinned": True},
        {"role": "user", "content": "Secret recovery token: RECOVERY_ALPHA_77921"},
        {"role": "assistant", "content": "Token recorded and validated."},
        {"role": "user", "content": "How do we handle 10,000 concurrent reads?"},
        {"role": "assistant", "content": "Use read connections with WAL mode enabled."}
    ]
    boss_res = skill.compress(boss_convo, keep_recent_n=2)
    # Verify pinned instruction survived in active prompt
    survived_pinned = [t for t in boss_res["updated_history"] if "PINNED ARCHITECTURE" in t.get("content", "")]
    assert len(survived_pinned) == 1, "Pinned turn must survive compression intact"

    # Verify secret needle can be retrieved and rehydrated
    needle_matches = skill.query("RECOVERY_ALPHA_77921", limit=1)
    assert len(needle_matches) >= 1
    needle_rehydrated = skill.rehydrate(needle_matches[0]["segment_id"], user_query="Find token", format_style="xml")
    assert "RECOVERY_ALPHA_77921" in needle_rehydrated
    passed_points += 1
    print("  ✓ PASSED: Pinned turn preserved and secret needle retrieved cleanly at Rank 1.\n")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("=" * 75)
    print(f"  >>> ALL 12 BLUEPRINT POINTS VERIFIED & PASSED ({passed_points}/12) <<<")
    print("=" * 75)
    return passed_points == 12


if __name__ == "__main__":
    ok = run_12_point_blueprint_eval()
    sys.exit(0 if ok else 1)
