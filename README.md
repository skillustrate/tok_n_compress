# Agentic Hybrid Memory System (AHMS v2.0)
### `tok_n_compress` — The Cognitive Memory Architecture for Coding Agents

[![Tests](https://img.shields.io/badge/pytest-36%20passed-brightgreen.svg)]()
[![Evals](https://img.shields.io/badge/evals-5%2F5%20passed-brightgreen.svg)]()
[![MCP](https://img.shields.io/badge/MCP-2024--11--05%20compliant-blueviolet.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A production-grade, local-first **Agentic Hybrid Memory System (AHMS)** engineered for long-running AI coding sessions (**100K+ to 1M+ tokens**). 

Instead of naive text summarization or rigid static cutoffs, AHMS organizes memory into a **three-layered cognitive architecture** featuring dynamic percentage-based budgeting, single-file vector persistence, hybrid Reciprocal Rank Fusion (RRF) search, and native **Model Context Protocol (MCP)** tool and resource integration across modern agent harnesses (Cursor, Windsurf, Claude Desktop, Zed, Claude Code).

---

## 🧠 The Three-Layer Cognitive Brain

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Working Memory (Active Context Window)                        │
│ • Dynamic Sliding Budget (~25%–35% of attached model context window)   │
│ • Pin Protection (Protects system instructions, schemas, architecture) │
│ • Semantic Boundary Snapping (Clean User/Assistant/Tool boundaries)    │
│ • Automatic Large Output & Log Collapsing (>50 lines to summaries)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Eviction & Downshift)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Episodic Memory (Structured Checkpoints)                      │
│ • Hierarchical Summary Tree & Extracted Entity/File Maps               │
│ • Decisions, Key Prompts, and Foreign Key Links to Layer 3             │
│ • Exposed via MCP Resource: memory://checkpoint-map                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Archived Persistence)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Semantic / Deep Memory (Single-File SQLite + Vector)          │
│ • SQLite FTS5: Token-exact keyword & file-path searches                │
│ • Embedded Vectors: Fast 256-dim cosine similarity distance in SQLite  │
│ • 100% ACID Synchronized: Single .db file, zero external daemons       │
│ • Hybrid Reciprocal Rank Fusion (RRF) Blended Scoring                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Highlights in v2.0

* **Universal MCP Server Support:** Zero custom integration code required. Mount natively into Cursor, Windsurf, Claude Desktop, and Zed as MCP tools and context resources.
* **Single-File Zero-Daemon Persistence:** Eliminates external vector databases (Chroma/Docker/Qdrant). Conversation text, FTS5 indexes, and vector embeddings reside inside a single portable `.db` file with atomic ACID guarantees.
* **Dynamic Percentage Budgeting:** No brittle hardcoded token limits (e.g., 4,000 tokens). Scales seamlessly from 8k local models (~2.4k working memory) to 128k/200k models (~40k working memory).
* **Pin Protection (Anchoring):** Critical system rules, database schemas, or user-pinned messages (`pinned: true`) are permanently shielded from eviction.
* **Semantic Boundary Snapping:** Eviction never slices mid-code block or mid-sentence; it only trims at complete turn interactions.
* **Large Output / Log Collapsing:** Verbose compiler logs or 400-line stack traces are collapsed into structured summaries before context exhaustion, keeping raw logs indexed in Layer 3.
* **Query Expansion & Hybrid RRF Search:** Casual queries (*"that error from earlier"*) expand into technical synonyms and blend keyword (FTS5) and semantic vector matches using Reciprocal Rank Fusion:
  $$\text{RRF}(d) = \sum_{r \in \{\text{fts}, \text{vec}\}} \frac{1}{k + \text{rank}_r(d)}$$
* **XML-Enveloped Context Re-hydration:** Retrieved memories are injected with `<retrieved_context>` XML envelopes to maintain clean boundaries between historical context and live prompts.

---

## 📁 Repository Layout

```
gitproject/
├── tok_n_compress/                      # Core AHMS engine & MCP server
│   ├── context_window_manager.py        # Dynamic budgeting, pin protection & log collapsing
│   ├── embeddings.py                    # FastLocalEmbedding (256-dim, normalized) & cosine metrics
│   ├── database.py                      # SQLite persistence (FTS5 + vector distance + RRF)
│   ├── checkpoint_generator.py          # Atomic checkpoint generation & hierarchy
│   ├── summarizer.py                    # NLP summarization & faithfulness evaluation
│   ├── retrieval.py                     # Query expansion, hybrid RRF & XML rehydration
│   ├── compression_engine.py            # Orchestrator managing working memory & eviction
│   ├── skill.py                         # High-level developer API
│   ├── mcp_server.py                    # Standard JSON-RPC stdio MCP Server (v2.0)
│   ├── cli.py                           # CLI utility (tok-compress)
│   └── evals/                           # Benchmark & evaluation suite
├── tests/                               # Comprehensive automated test suite (36 tests)
│   ├── test_ahms_v2.py                  # v2.0 specific unit & integration tests
│   ├── test_database.py                 # SQLite persistence & ACID transaction tests
│   ├── test_compression_engine.py       # Eviction & compression tests
│   ├── test_retrieval.py                # Retrieval & re-hydration tests
│   ├── test_skill.py                    # Skill interface tests
│   ├── test_cli_and_mcp.py              # CLI & MCP protocol tests
│   └── test_summarizer.py               # NLP summarization tests
├── next-dev.md                          # Architecture specification & roadmap
├── pyproject.toml                       # Package specification (v2.0.0)
└── README.md                            # Documentation
```

---

## 🚀 Installation & Setup

### Option 1: Install with `uv` (Recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option 2: Standard `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Option 3: Run with Podman / Docker Dev Container

```bash
# Execute tests inside the container
podman exec dev-box pytest
```

---

## 🔌 Universal MCP Integration

`tok_n_compress` natively supports the **Model Context Protocol (MCP)**. Any MCP-compliant client can register it directly:

### 1. Claude Code
```bash
claude mcp add tok-compress -- uv --directory /path/to/gitproject run tok-mcp
```

### 2. Cursor, Windsurf, Zed, or Claude Desktop
Add this to your MCP configuration file (`mcp_config.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tok-compress": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/gitproject", "run", "tok-mcp"]
    }
  }
}
```

### Exposed MCP Tools & Resources
| Name | Type | Description |
| :--- | :--- | :--- |
| `compress_conversation` | **Tool** | Compress older turns into SQLite checkpoints with pin protection and boundary snapping. |
| `query_memory` | **Tool** | Hybrid RRF search (FTS5 + Vector Cosine) with query expansion across compressed memory. |
| `rehydrate_segment` | **Tool** | Inject historical dialogue into active prompt wrapped in `<retrieved_context>` XML. |
| `get_checkpoint_map` | **Tool** | Retrieve the high-level Layer 2 episodic summary tree. |
| `pin_segment` | **Tool** | Pin or unpin a historical segment to shield it from automated eviction. |
| `get_memory_stats` | **Tool** | Inspect storage metrics, context window size, and dynamic token allocations. |
| `memory://checkpoint-map`| **Resource** | Real-time JSON stream of the episodic checkpoint hierarchy. |

---

## 💻 Python Quickstart

### 1. Initialize with Dynamic Model Budgeting
```python
from tok_n_compress import ConversationCompressorSkill

# Configure for a model with 128k context window (e.g. Claude 3.5 Sonnet)
skill = ConversationCompressorSkill(
    db_path="./checkpoints.db",
    model_context_window=128000,
    working_memory_ratio=0.30  # 38,400 token working memory budget
)
```

### 2. Pin Critical Instructions & Compress Context
```python
conversation = [
    # Pinned messages are NEVER evicted
    {"role": "system", "content": "CORE SCHEMA: PostgreSQL users(id, email, api_key).", "pinned": True},
    {"role": "user", "content": "Let's implement password hashing using Argon2id."},
    {"role": "assistant", "content": "Configured Argon2id hasher in auth/hasher.py."},
    {"role": "user", "content": "Now write unit tests for the hasher."},
    {"role": "assistant", "content": "Added 5 test cases in tests/test_hasher.py."}
]

# Compress older turns while keeping the latest 2 turns active
result = skill.compress(conversation, keep_recent_n=2)

# Active history preserves the pinned directive, replaces intermediate turns with a checkpoint tag, and keeps recent turns
active_history = result["updated_history"]
print(f"Checkpoint #{result['checkpoint_id']} created (Ratio: {result['compression_ratio']}x)")
```

### 3. Hybrid RRF Query & XML Re-hydration
```python
# Query memory using natural language (automatically expanded and searched via RRF)
matches = skill.query("that password hashing we did", limit=2)

for match in matches:
    # Format into XML envelope ready for prompt injection
    xml_block = skill.rehydrate(
        match["segment_id"],
        user_query="How was password hashing implemented?",
        format_style="xml"
    )
    print(xml_block)
```

**Output:**
```xml
<retrieved_context segment_id="1" checkpoint_id="1" relevance="0.92">
  <query>How was password hashing implemented?</query>
  <content>
[USER]: Let's implement password hashing using Argon2id.
[ASSISTANT]: Configured Argon2id hasher in auth/hasher.py.
  </content>
</retrieved_context>
```

### 4. Inspect the Episodic Checkpoint Map
```python
checkpoint_map = skill.get_checkpoint_map()
for cp in checkpoint_map:
    print(f"[{cp['checkpoint_id']}] {cp['timestamp']} - {cp['summary']} (Files: {cp['files']})")
```

---

## 🛠️ CLI Usage

```bash
# Query memory via CLI
tok-compress query "Argon2id password hashing"

# View memory database statistics and budgeting
tok-compress stats

# Rehydrate a segment by ID
tok-compress rehydrate 1 --query "Show hashing configuration"
```

---

## 🧪 Test Suite & Verification

The project includes 36 automated unit and integration tests covering the complete AHMS v2.0 stack:

```bash
# Run tests locally or inside the dev container
pytest -v
# or
podman exec dev-box pytest -v
```

```
============================== 36 passed in 1.90s ==============================
```

### Test Coverage Highlights
* **FastLocalEmbedding:** Vector dimensionality, L2 normalization, and serialization.
* **SQLite Persistence & FTS5:** Full-text indexing, SQLite registered cosine distance, and ACID rollbacks.
* **Hybrid RRF Search:** Blended scoring and ranking across keyword and semantic spaces.
* **Dynamic Budgeting:** Window calculations for 8k, 32k, and 128k context horizons.
* **Pin Protection:** Immunity of pinned turns during eviction.
* **Log Collapsing:** Large stdout/stack trace collapsing (>50 lines).
* **Semantic Boundary Snapping:** Turn-boundary alignment.
* **Query Expansion:** Technical synonym enrichment on ambiguous queries.
* **MCP Server v2.0 Protocol:** Tool invocation, resource streaming, and protocol conformance.
