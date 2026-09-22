"""
Evaluation Datasets and Synthesizers for Conversation Compressor Skill.

Generates realistic multi-topic conversations, including a simulated 100K+ token conversation,
file-tracking scenarios, and needle-in-a-haystack verification tests.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple


def generate_needle_in_haystack_conversation(
    total_messages: int = 100,
    needles: List[Tuple[str, str]] = None
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Generate an extensive conversation with specific 'needles' (critical facts/errors)
    embedded at defined intervals.

    Args:
        total_messages: Total messages to generate.
        needles: List of (query, needle_content) pairs.

    Returns:
        Tuple of (messages, needles)
    """
    if needles is None:
        needles = [
            ("ERROR_CODE_9841", "Critical failure during payment processing: ERROR_CODE_9841_PAYLOAD_CORRUPT"),
            ("staging secret key", "Database staging secret key set to: SEC_KEY_99812_ALPHA_STAGING"),
            ("migration version", "Schema rollback executed successfully to migration version v2.14.9-final"),
            ("quarterly report", "Uploaded quarterly report data in finance_2026_q1_audited.csv")
        ]

    topics = [
        ("Architecture", "Discussing microservice boundaries and event queues for message streaming."),
        ("Infrastructure", "Terraform configuration updates for kubernetes cluster autoscaling."),
        ("Frontend", "React state management migration from Redux to Zustand store."),
        ("CI/CD", "Optimizing GitHub Actions workflows and Docker build caching layers."),
        ("Security", "Auditing OAuth token lifecycle and CORS origin policies across domains.")
    ]

    messages: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)

    # Calculate placement indices for needles
    needle_step = total_messages // (len(needles) + 1)
    needle_indices = { (i + 1) * needle_step: needles[i] for i in range(len(needles)) }

    for i in range(total_messages):
        current_time = (base_time + timedelta(minutes=i * 10)).isoformat()
        role = "user" if i % 2 == 0 else "assistant"

        if i in needle_indices:
            query, needle_content = needle_indices[i]
            if role == "user":
                content = f"Please check this critical issue: {needle_content}"
            else:
                content = f"Acknowledged. I have logged and recorded: {needle_content}"
        else:
            topic_name, topic_desc = topics[i % len(topics)]
            filler = (
                f"Turn {i}: {topic_name} discussion. {topic_desc} "
                "Evaluating performance benchmarks, latency histograms, and concurrency bottlenecks under load. "
                "The team reviewed telemetry traces and identified several IO wait times. "
            ) * 5
            content = filler

        messages.append({
            "role": role,
            "content": content,
            "timestamp": current_time
        })

    return messages, needles


def generate_long_100k_conversation() -> List[Dict[str, Any]]:
    """
    Generate a simulated ultra-long conversation approaching or exceeding 100K tokens
    across multiple distinct task phases.
    """
    phases = [
        ("Phase 1: Project Scoping", "Defining requirements for real-time document collaboration engine.", 30),
        ("Phase 2: Data Modeling", "Designing PostgreSQL schema, migrations, and indexing strategies.", 30),
        ("Phase 3: Protocol Design", "Specifying WebSocket framing protocol and Operational Transformation logic.", 30),
        ("Phase 4: Implementation", "Writing Go backend services, connection pools, and Redis pubsub handlers.", 40),
        ("Phase 5: Performance Testing", "Running k6 load tests with 50,000 concurrent WebSocket connections.", 30),
        ("Phase 6: Deployment & Monitoring", "Deploying Helm charts and Grafana dashboards for cluster alerts.", 30)
    ]

    messages: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(days=5)
    global_index = 0

    for phase_name, phase_desc, count in phases:
        for i in range(count):
            global_index += 1
            current_time = (base_time + timedelta(minutes=global_index * 15)).isoformat()
            role = "user" if i % 2 == 0 else "assistant"

            # Repeat paragraphs to simulate substantial token mass per message (~500 tokens/msg)
            body = (
                f"[{phase_name}] Detail block {i+1}. {phase_desc} "
                "Investigating edge cases in partition tolerance and distributed lock leases. "
                "Ensuring idempotency tokens prevent duplicate transaction commits across retries. "
                "Telemetry indicates p99 latency remains bounded under 12 milliseconds across all worker nodes. "
            ) * 12

            messages.append({
                "role": role,
                "content": body,
                "timestamp": current_time
            })

    return messages
