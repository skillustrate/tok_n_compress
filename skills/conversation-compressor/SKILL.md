---
name: conversation-compressor
description: >-
  Memory management system using a Hybrid Memory Model for handling ultra-long conversations
  (100K+ tokens). Compresses older dialogue into summary checkpoints while persisting raw
  segments in SQLite for deep on-demand retrieval and context re-hydration.
---

# Conversation Compressor Skill (Hybrid Memory Model)

The **Conversation Compressor** skill enables AI agents to handle arbitrarily long conversation histories without context-window exhaustion or critical information loss. It transitions from an unbounded linear message list to a **Hybrid Memory Model**:
- **Summary Layer (Active Context)**: Distilled high-level summaries of older dialogue blocks, retaining goals, entities, and outcomes.
- **Detail Layer (Deep Memory)**: Uncompressed raw messages persisted in a structured SQLite database (`checkpoints.db`) that can be searched and re-hydrated on demand.

---

## When to Use This Skill

Activate and use this skill when:
1. **Context Window Saturation**: The active context token count approaches 70-80% of the model's limit (e.g. 80,000+ tokens).
2. **Topic Shifts**: The user shifts to a distinct new project phase, debugging thread, or task, and previous dialogue history is no longer needed in full detail.
3. **Information Re-hydration**: The user asks for specific past details (e.g., "What was that exact error code from earlier?", "What was the JSON schema we agreed on?") that were compressed away.

---

## Architecture & Data Flow

```
[ Active Conversation History ]
               │
               ▼ (Trigger: Token threshold or topic shift)
┌───────────────────────────────────────────────┐
│              Compression Engine               │
│  - Extracts summary, files, and key prompts   │
│  - Saves raw transcript to SQLite atomically  │
│  - Returns Checkpoint Reference               │
└──────────────────────┬────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[ Updated Active Context ]    [ SQLite Long-term DB ]
 - Checkpoint #N Summary       - checkpoints table (metadata & refs)
 - Recent uncompressed turns   - raw_segments table (full text)
                                     ▲
                                     │
                             [ query_memory ]
                           (Deep Re-hydration)
```

---

## Python API Quickstart

### 1. Initializing the Skill

```python
from tok_n_compress import ConversationCompressorSkill

skill = ConversationCompressorSkill(
    db_path="./checkpoints.db",
    token_threshold=80000,
    topic_shift_sensitivity=0.5
)
```

### 2. Checking Triggers & Compressing Conversation

```python
# Check if compression is needed
should_run = skill.should_compress(
    current_token_count=85000,
    segment_length=1200,
    topic_shift_detected=False
)

if should_run:
    result = skill.compress(
        raw_history=messages,
        current_token_count=85000,
        keep_recent_n=2  # Keep last 2 messages uncompressed
    )
    # Replace conversation messages with updated history:
    messages = result["updated_history"]
    print(f"Compressed {result['tokens_saved']} tokens (Ratio: {result['compression_ratio']}x)")
```

### 3. Querying Memory & Re-hydrating Context

When the user asks for historical details:

```python
# 1. Search compressed memory
matches = skill.query("TypeError in database migration", limit=3)

# 2. Re-hydrate matched raw segment into active prompt
for match in matches:
    context_block = skill.rehydrate(
        match["segment_id"],
        user_query="How did we resolve the database migration error?"
    )
    # Inject context_block into system instructions or current prompt
```

### 4. Navigating History Hierarchy

Checkpoints maintain parent-child links to allow traversing conversation branches:

```python
lineage = skill.get_hierarchy(checkpoint_id=3)
for cp in lineage:
    print(f"Checkpoint #{cp['id']}: {cp['summary']}")
```

---

## Database Schema Reference

The database (`checkpoints.db`) enforces atomic persistence with foreign key constraints:

- **`checkpoints`**:
  - `id`: Auto-incrementing primary key.
  - `parent_checkpoint_id`: Foreign key to previous checkpoint.
  - `timestamp`: UTC ISO timestamp.
  - `summary`: Compressed text summary.
  - `files_metadata`: JSON list of referenced files.
  - `key_prompts`: JSON list of user intents and outcomes.
  - `raw_segment_refs`: JSON array of IDs in `raw_segments`.

- **`raw_segments`**:
  - `id`: Auto-incrementing primary key.
  - `checkpoint_id`: Foreign key linking to parent checkpoint.
  - `content`: Exact uncompressed conversation text.
  - `timestamp`: Original creation timestamp.

---

## Evaluation & Quality Benchmarks

To run the automated test suite and benchmarks:

```bash
# Run unit & integration tests
pytest tests/ -v

# Run the comprehensive evaluation benchmark
python3 tok_n_compress/evals/run.py
```
