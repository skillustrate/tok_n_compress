# Next Development Phase: Agentic Hybrid Memory System (AHMS)

**Status:** Planned (Post-Architectural Audit)  
**Target Version:** v2.0 (The "Brain" Upgrade)  
**Objective:** Transform `tok_n_compress` from a keyword-based utility into an open-source, **MCP-Compliant Agentic Hybrid Memory System (AHMS)**. This evolution moves beyond simple summarization into a three-layered cognitive architecture featuring dynamic percentage-based budgeting, single-file vector persistence (`sqlite-vec`), and plug-and-play compatibility across IDEs (Claude Desktop, Cursor, Windsurf, Zed).

---

## 1. The Problem: Why We Are Upgrading
The v1.0 "Hybrid Model" was a massive leap forward, but architectural audits and real-world agentic workflows identified four critical failure points:

1. **The "Context Injection" Noise:** Simply dumping raw text back into prompts breaks conversational flow and confuses the model about the timeline of events.
2. **The "Query Ambiguity" Gap:** Users search with vague phrases (e.g., *"that error from earlier"*). Keyword-only search fails on synonyms and casual language.
3. **The Static Token Trap:** Hardcoding static cutoffs (e.g., 4,000 tokens) breaks small-window models (overflowing 8k contexts) and severely underutilizes massive models (e.g., 128k–200k contexts).
4. **The Split-Brain Persistence Risk:** Managing external vector databases alongside SQLite risks desynchronization and adds heavy container/daemon dependencies.

---

## 2. The Solution: The Three-Layer Brain

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Working Memory (Active Context Window)             │
│ • Dynamic Sliding Budget (25%-35% of Model Window)          │
│ • Pin Protection (System Prompts, Schemas, Key Rules)       │
│ • Semantic Boundary Trimming & Large Log Collapsing         │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Eviction & Downshift)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Episodic Memory (Structured Checkpoints)           │
│ • Hierarchical Summary Tree & Extracted Metadata            │
│ • Decisions, Entity Maps, and Pointers to Raw Data          │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Archived Persistence)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Semantic / Deep Memory (Single-File SQLite + Vec)  │
│ • SQLite FTS5 (Keyword / Metadata Search)                   │
│ • sqlite-vec (Local Vector Embeddings & Similarity Search)  │
│ • Zero External Daemons, 100% ACID Synchronized Storage     │
└─────────────────────────────────────────────────────────────┘
```

### **Layer 1: The Active Layer (Working Memory)**
* **Content:** The active conversational turns and tool outputs.
* **Management:** Governed by the `ContextWindowManager` using **Dynamic Token Allocation**:
  * **Sliding Percentage Window:** Default working budget scales dynamically to ~25%–35% of the attached LLM’s context window (e.g., ~2.5k tokens for an 8k model; ~40k tokens for a 128k model).
  * **Pin Protection:** Turns explicitly marked with `pinned: true` (core instructions, active schema, architectural decisions) are permanently shielded from eviction.
  * **Log & Payload Collapsing:** Verbose terminal logs or code outputs (>50 lines) are collapsed into summary pointers, preventing technical noise from blowing through the physical context ceiling.
  * **Semantic Boundary Snapping:** Eviction never cuts mid-code or mid-sentence; it only slices cleanly at completed User/Assistant/Tool interaction blocks.

### **Layer 2: The Summary Layer (Episodic Memory)**
* **Content:** Hierarchical checkpoints created upon Layer 1 eviction.
* **Structure:**
  * **Summary:** Distilled intent and outcome of the evicted block.
  * **Metadata:** Extracted entities (modified files, technical decisions, bug IDs).
  * **Pointers:** Foreign keys linking directly to the raw segments stored in Layer 3.
* **Purpose:** Provides a low-token "Bird's-Eye Map" of long sessions.

### **Layer 3: The Detail Layer (Semantic / Deep Memory)**
* **Content:** Full raw transcripts, code segments, and vector embeddings.
* **Storage Engine (`sqlite-vec` + SQLite):**
  * **Relational / FTS5 (SQLite):** Exact keyword matches, file path queries, and metadata lookups.
  * **Embedded Vector (`sqlite-vec`):** Meaning-based semantic similarity search residing inside the **same** SQLite `.db` file.
  * **Zero Split-Brain:** Checkpoints, raw logs, and embeddings commit together under a single ACID transaction—no external Chroma/Docker dependencies required.

---

## 3. The Intelligence Pipeline

### **A. The Compression & Eviction Loop (The "Write" Path)**
When active turns exceed the dynamic threshold (e.g., 70% of working memory budget):
1. **Candidate Identification:** Select the oldest unpinned turns up to the nearest clean semantic boundary.
2. **Analysis & Summarization:** `SummarizationEngine` creates a structured checkpoint (intent, decisions, entities).
3. **Log Collapsing:** Large outputs/logs are collapsed to summaries, with full outputs pushed down to Layer 3.
4. **Atomic Ingestion:** Raw turns and their vector embeddings are inserted into `raw_segments` and `vec_items` inside SQLite within one transaction.
5. **Context Replacement:** Evicted turns in Layer 1 are replaced with a lightweight checkpoint reference tag.

### **B. The Retrieval Loop (The "Read" Path)**
When past context is needed:
1. **Query Expansion:** The agent expands casual or vague user questions (*"that error from earlier"*) into technical terms based on recent conversational signals.
2. **Hybrid Search:**
   * **Exact Search (FTS5):** Matches exact keywords, function names, and file paths.
   * **Semantic Search (`sqlite-vec`):** Finds conceptually related turns via cosine distance.
3. **Reciprocal Rank Fusion (RRF):** Blends and ranks keyword and vector matches into a unified relevance score.
4. **Structured Re-hydration:** Retrieved context is wrapped into clean, machine-readable XML tags (`<retrieved_context>...</retrieved_context>`) with timestamps and relevance scores to preserve conversational coherence.

---

## 4. MCP (Model Context Protocol) Integration

To allow zero-config integration into IDEs and harnesses (Cursor, Windsurf, Claude Desktop, Zed), AHMS exposes a standardized MCP server interface:

* **Tools:**
  * `store_memory`: Allows agents to explicitly save key facts, decisions, or pin critical turns.
  * `search_memory`: Hybrid semantic + keyword retrieval returning ranked results.
  * `get_checkpoint_map`: Returns the high-level Layer 2 episodic summary tree.
* **Resources:**
  * `memory://active-summary`: Real-time episodic map exposed as an MCP context resource.

---

## 5. Implementation Roadmap

### **Milestone 1: Unified Storage & `sqlite-vec` Foundation**
* [ ] Implement SQLite schema with FTS5 and `sqlite-vec` extension integration.
* [ ] Verify single-file `.db` portability and atomic write transactions.
* [ ] Benchmark vector write/search operations locally with zero background daemons.

### **Milestone 2: Dynamic Budgeting & Context Window Manager**
* [ ] Build `ContextWindowManager` with dynamic percentage calculation based on target model window.
* [ ] Implement `is_pinned` message protection.
* [ ] Implement Semantic Boundary detection (snapping eviction to clean interaction blocks).
* [ ] Add automated large output / log collapsing.

### **Milestone 3: Summarization & Chunking Pipeline**
* [ ] Implement `SummarizationEngine` for Layer 2 checkpoint creation.
* [ ] Integrate local embedding model (e.g., `sentence-transformers` / fast ONNX runtime).
* [ ] Implement Recursive Character Chunking with syntax boundary awareness.

### **Milestone 4: Hybrid Search & Fusion Engine**
* [ ] Implement Contextual Query Expansion for ambiguous queries.
* [ ] Build Hybrid Search querying both FTS5 text and `sqlite-vec` embeddings.
* [ ] Implement Reciprocal Rank Fusion (RRF) scoring algorithm.
* [ ] Implement XML-enveloped `<retrieved_context>` re-hydration.

### **Milestone 5: MCP Server Interface**
* [ ] Implement MCP server layer using Python MCP SDK.
* [ ] Expose `store_memory`, `search_memory`, and `get_checkpoint_map` tools.
* [ ] Provide plug-and-play configuration templates for Cursor, Windsurf, and Claude Desktop.

### **Milestone 6: Validation & Stress Testing**
* [ ] **Needle-in-a-Haystack Test:** Retrieve specific obscure facts across 100k+ historical tokens.
* [ ] **Dynamic Eviction Integrity:** Verify zero truncation of pinned turns and clean boundary cuts.
* [ ] **Synonym & Ambiguity Benchmark:** Validate query expansion recall on vague phrasing.
* [ ] **MCP Harness Test:** End-to-end multi-turn testing inside Claude Desktop and Cursor.
