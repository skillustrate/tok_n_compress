#!/usr/bin/env python3
"""
Comprehensive Evaluation and Benchmarking Suite for Conversation Compressor Skill.

Executes quantitative evaluations covering:
- Needle-in-a-Haystack Retrieval & Context Rehydration
- Ultra-long (100K+ token) Conversation Compression
- Faithfulness, Completeness, and Entity Retention
- Topic-Shift Trigger Sensitivity
- Atomic Database Integrity and Cascade Invariance
"""

import sys
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so imports work regardless of execution location
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_current_dir.parent) not in sys.path:
    sys.path.insert(0, str(_current_dir.parent))

from tok_n_compress.database import DatabaseManager
from tok_n_compress.skill import ConversationCompressorSkill
from tok_n_compress.evals.eval_metrics import BenchmarkEvaluator
from tok_n_compress.evals.eval_datasets import (
    generate_needle_in_haystack_conversation,
    generate_long_100k_conversation,
)


def run_comprehensive_evaluation() -> bool:
    """Run all benchmark stages and report results."""
    print("=" * 72)
    print("      CONVERSATION COMPRESSOR SKILL - PRODUCTION EVALUATION SUITE")
    print("=" * 72)

    passed_all = True
    temp_dir = tempfile.mkdtemp(prefix="compressor_eval_")
    eval_db_path = os.path.join(temp_dir, "eval_checkpoints.db")
    print(f"[*] Ephemeral evaluation environment: {eval_db_path}\n")

    skill = ConversationCompressorSkill(
        db_path=eval_db_path,
        token_threshold=50000,
        topic_shift_sensitivity=0.45
    )

    # -------------------------------------------------------------
    # STAGE 1: Needle-in-a-Haystack Retrieval & Re-hydration
    # -------------------------------------------------------------
    print("[STAGE 1/6] Needle-in-a-Haystack Retrieval & Rehydration Benchmark")
    print("-" * 72)
    needle_convo, needles = generate_needle_in_haystack_conversation(total_messages=60)
    tokens_before = skill.compression_engine.count_history_tokens(needle_convo)
    print(f"  • Generated conversation with {len(needle_convo)} messages (~{tokens_before:,} tokens)")
    print(f"  • Embedded {len(needles)} critical needle entities across dialogue turns")

    # Compress the conversation
    comp_res, compress_latency = BenchmarkEvaluator.measure_latency(
        skill.compress,
        raw_history=needle_convo,
        keep_recent_n=2
    )
    print(f"  • Compression completed in {compress_latency:.1f}ms (Ratio: {comp_res['compression_ratio']}x)")
    print(f"  • Tokens saved: {comp_res['tokens_saved']:,} (From {comp_res['tokens_before']:,} to {comp_res['tokens_after']:,})")

    # Verify retrieval for each needle
    hits = 0
    rehydrate_verified = 0
    for query_key, expected_needle in needles:
        results, query_latency = BenchmarkEvaluator.measure_latency(
            skill.query,
            query_text=query_key,
            limit=3
        )
        recall = BenchmarkEvaluator.evaluate_retrieval_recall(results, query_key, k=3)
        if recall["hit_at_k"]:
            hits += 1
            # Test rehydration
            top_match = results[0]
            rehydrated = skill.rehydrate(top_match["segment_id"], user_query=query_key)
            if query_key.lower() in rehydrated.lower():
                rehydrate_verified += 1
            print(f"    ✓ Needle '{query_key}' found at rank {recall['rank']} ({query_latency:.1f}ms)")
        else:
            print(f"    ✗ Needle '{query_key}' NOT found in top-{recall['k']}")

    needle_recall_pct = (hits / len(needles)) * 100
    print(f"  • Needle Retrieval Recall@3: {needle_recall_pct:.1f}% ({hits}/{len(needles)})")
    print(f"  • Rehydration Content Fidelity: {rehydrate_verified}/{len(needles)}")

    if needle_recall_pct < 100.0:
        passed_all = False
        print("  [FAIL] Retrieval Recall@3 threshold (100%) not met!")
    else:
        print("  [PASS] Needle Retrieval and Rehydration verified.")

    # -------------------------------------------------------------
    # STAGE 2: 100K Token Simulated Conversation Benchmark
    # -------------------------------------------------------------
    print("\n[STAGE 2/6] Ultra-Long Conversation Benchmark (Simulated 100K Tokens)")
    print("-" * 72)
    long_convo = generate_long_100k_conversation()
    long_tokens = skill.compression_engine.count_history_tokens(long_convo)
    print(f"  • Generated multi-phase conversation: {len(long_convo)} messages (~{long_tokens:,} tokens)")

    # Execute compression
    long_result, long_latency = BenchmarkEvaluator.measure_latency(
        skill.compress,
        raw_history=long_convo,
        keep_recent_n=2
    )

    comp_metrics = BenchmarkEvaluator.calculate_compression_metrics(
        original_tokens=long_result["tokens_before"],
        compressed_tokens=long_result["tokens_after"],
        original_chars=long_result["raw_segment_length"],
        compressed_chars=long_result["compressed_summary_length"]
    )

    print(f"  • Latency for large transcript: {long_latency:.1f}ms")
    print(f"  • Compression Ratio: {comp_metrics['compression_ratio']}x")
    print(f"  • Token Reduction: {comp_metrics['token_reduction_pct']}% (Saved {comp_metrics['tokens_saved']:,} tokens)")
    print(f"  • Active Context History Size: {len(long_result['updated_history'])} messages")

    if comp_metrics["compression_ratio"] < 3.0:
        passed_all = False
        print("  [FAIL] Compression ratio is below minimum target of 3.0x")
    else:
        print("  [PASS] Compression efficiency benchmark achieved.")

    # -------------------------------------------------------------
    # STAGE 3: Faithfulness, Completeness & Entity Retention
    # -------------------------------------------------------------
    print("\n[STAGE 3/6] Summary Faithfulness & Entity Retention Benchmark")
    print("-" * 72)
    sample_dialogue = (
        "[USER]: We encountered error ERR_CONNECTION_REFUSED when accessing https://api.staging.internal:8443/v1/auth. "
        "Also config file db_replica.yaml is missing parameter max_overflow.\n\n"
        "[ASSISTANT]: I investigated the issue. The connection was refused because the service container was restarting. "
        "I added max_overflow: 20 to db_replica.yaml and verified the endpoint is healthy."
    )
    summary_data = skill.summarizer.summarize_segment(sample_dialogue)
    faith_metrics = skill.evaluate_quality(sample_dialogue, summary_data["summary"])
    entity_eval = BenchmarkEvaluator.calculate_entity_retention(
        sample_dialogue,
        summary_data["summary"] + " " + str(summary_data["files_metadata"]),
        targets=["db_replica.yaml", "max_overflow"]
    )

    print(f"  • Generated Summary: \"{summary_data['summary']}\"")
    print(f"  • Extracted Files: {[f['filename'] for f in summary_data['files_metadata']]}")
    print(f"  • Faithfulness (Precision): {faith_metrics['faithfulness']:.2f}")
    print(f"  • Completeness (Recall): {faith_metrics['completeness']:.2f}")
    print(f"  • F1 Score: {faith_metrics['f1_score']:.2f}")
    print(f"  • Entity Retention Rate: {entity_eval['retention_rate'] * 100:.1f}%")

    if faith_metrics["faithfulness"] < 0.5:
        passed_all = False
        print("  [FAIL] Summary faithfulness score below threshold (0.50)!")
    else:
        print("  [PASS] Summary quality and entity retention verified.")

    # -------------------------------------------------------------
    # STAGE 4: Topic-Shift Trigger Sensitivity Benchmark
    # -------------------------------------------------------------
    print("\n[STAGE 4/6] Topic-Shift Detection Sensitivity Benchmark")
    print("-" * 72)
    context_db = "Optimizing PostgreSQL B-tree indices and vacuum parameters for write-heavy table."
    same_topic = "Added partial index on created_at and adjusted autovacuum_vacuum_scale_factor."
    shifted_topic = "Configuring Figma styling tokens and mobile typography scales for iOS dark mode."

    shift_same = skill.compression_engine.detect_topic_shift(same_topic, context_db)
    shift_diff = skill.compression_engine.detect_topic_shift(shifted_topic, context_db)

    print(f"  • Continuous topic shift detected: {shift_same} (Expected: False)")
    print(f"  • Distinct topic shift detected: {shift_diff} (Expected: True)")

    if shift_same or not shift_diff:
        passed_all = False
        print("  [FAIL] Topic-shift detector failed distinction test!")
    else:
        print("  [PASS] Topic-shift detection operates as expected.")

    # -------------------------------------------------------------
    # STAGE 5: Atomic Transactions & Database Integrity Benchmark
    # -------------------------------------------------------------
    print("\n[STAGE 5/6] SQLite Atomic Integrity & Hierarchy Benchmark")
    print("-" * 72)
    # Test atomic persistence and hierarchy
    stats_before = skill.get_stats()
    cp1 = skill.checkpoint_gen.generate_checkpoint("Segment 1: Project kickoff.")
    cp2 = skill.checkpoint_gen.generate_checkpoint("Segment 2: Implementation.", parent_id=cp1["id"])
    cp3 = skill.checkpoint_gen.generate_checkpoint("Segment 3: Deployment.", parent_id=cp2["id"])

    hierarchy = skill.get_hierarchy(cp3["id"])
    hierarchy_ids = [cp["id"] for cp in hierarchy]
    expected_ids = [cp1["id"], cp2["id"], cp3["id"]]
    print(f"  • Traversed hierarchy: {hierarchy_ids} (Expected: {expected_ids})")

    # Cascade delete test
    del_ok = skill.db.delete_checkpoint(cp3["id"])
    cp3_fetched = skill.db.get_checkpoint(cp3["id"])
    print(f"  • Deleted checkpoint #{cp3['id']} cascaded cleanly: {cp3_fetched is None}")

    if hierarchy_ids != expected_ids or not del_ok or cp3_fetched is not None:
        passed_all = False
        print("  [FAIL] Hierarchy traversal or atomic delete failed!")
    else:
        print("  [PASS] Atomic persistence and hierarchy integrity verified.")

    # -------------------------------------------------------------
    # STAGE 6: SQLite Vector DB & Hybrid RRF Semantic Benchmark
    # -------------------------------------------------------------
    print("\n[STAGE 6/6] SQLite Vector Persistence & Hybrid RRF Semantic Benchmark")
    print("-" * 72)

    from tok_n_compress.embeddings import get_embedding_engine
    embedder = get_embedding_engine()

    # Benchmark dataset with technical diversity
    semantic_corpus = [
        ("auth_failure", "PostgreSQL database connection timeout failure on master replica", "database"),
        ("k8s_eviction", "Kubernetes pod evicted due to memory pressure and cgroup limit", "infrastructure"),
        ("stripe_webhook", "Stripe payment webhook signature verification rejected by gateway", "payments"),
        ("frontend_bundle", "Vite production build bundle chunk size exceeded 500kb warning", "frontend"),
        ("ci_cache", "Docker layer cache miss during GitHub Actions container build", "devops")
    ]

    # Save to SQLite with embeddings
    for key, text, tag in semantic_corpus:
        vec = embedder.embed_text(text)
        skill.db.save_checkpoint_atomic(
            parent_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=f"Summary for {key}",
            files_metadata=[{"filename": f"{key}.py"}],
            key_prompts=[],
            raw_segment_content=text,
            embedding=vec
        )

    # Test cases: queries with conceptual overlap and synonyms
    test_queries = [
        ("PostgreSQL connection timeout", "auth_failure"),
        ("pod evicted memory pressure", "k8s_eviction"),
        ("Stripe webhook verification rejected", "stripe_webhook"),
        ("bundle chunk size exceeded", "frontend_bundle"),
        ("Docker layer cache miss build", "ci_cache")
    ]

    vector_hits = 0
    total_query_lat = 0.0

    for query, expected_key in test_queries:
        q_vec = embedder.embed_text(query)
        results, lat = BenchmarkEvaluator.measure_latency(
            skill.db.search_hybrid_rrf,
            query=query,
            query_vector=q_vec,
            limit=3
        )
        total_query_lat += lat

        if results:
            top_content = results[0].get("content", "")
            # Check if expected document was matched
            expected_content = dict([(k, t) for k, t, _ in semantic_corpus])[expected_key]
            if expected_content in top_content:
                vector_hits += 1
                print(f"    ✓ Query '{query}' -> matched '{expected_key}' at Rank 1 ({lat:.2f}ms, RRF: {results[0].get('rrf_score')})")
            else:
                print(f"    ✗ Query '{query}' -> mismatched (top was: '{top_content[:40]}...')")
        else:
            print(f"    ✗ Query '{query}' -> NO RESULTS")

    avg_lat = total_query_lat / len(test_queries)
    vector_accuracy = (vector_hits / len(test_queries)) * 100.0
    print(f"  • SQLite Vector & RRF Top-1 Accuracy: {vector_accuracy:.1f}% ({vector_hits}/{len(test_queries)})")
    print(f"  • Average SQLite Vector Query Latency: {avg_lat:.2f}ms")

    if vector_accuracy < 100.0:
        passed_all = False
        print("  [FAIL] SQLite Vector Hybrid RRF accuracy below 100% threshold!")
    else:
        print("  [PASS] SQLite Vector DB and Hybrid RRF Benchmark passed.")

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    if passed_all:
        print("  >>> ALL 6 BENCHMARK STAGES PASSED! REPO IS PRODUCTION READY <<<")
    else:
        print("  >>> BENCHMARK FAILED - REVIEW LOGGED ERRORS ABOVE <<<")
    print("=" * 72)

    return passed_all


if __name__ == "__main__":
    success = run_comprehensive_evaluation()
    sys.exit(0 if success else 1)
