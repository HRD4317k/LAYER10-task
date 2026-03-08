# Layer10 Memory Graph — Take-Home 2026

A production-grade pipeline that transforms GitHub issue discussions into a **grounded, temporal, queryable memory graph** with robust deduplication, conflict detection, and an interactive visualization layer.

## Corpus

**FastAPI GitHub Issues** — ~300 issues with comments, timeline events, labels, assignments, and state transitions from the [fastapi/fastapi](https://github.com/fastapi/fastapi) repository.

- **Source**: GitHub REST API v3
- **How to reproduce**: Run `python ingestion/github_ingest.py` (or set `GITHUB_TOKEN` for higher rate limits)
- Pre-fetched data included at `data/raw/fastapi_issues.json`

---

## Architecture Overview

```
Corpus (GitHub API)
  ↓
1. Ingestion         → data/raw/fastapi_issues.json      (idempotent, incremental)
  ↓
2. Transformation    → data/processed/events.json         (canonical event stream)
  ↓
3. Temporal Model    → data/processed/issue_state_history.json  (bi-temporal)
  ↓
4. Claim Extraction  → data/processed/claims.json         (LLM + validation + grounding)
  ↓
5. Deduplication     → data/processed/entities.json       (3-layer: artifact/entity/claim)
                     → data/processed/canonical_claims.json
                     → data/processed/merge_log.json      (reversible audit trail)
  ↓
6. Conflict Detection → data/processed/state_conflicts.json  (state + claim + temporal)
  ↓
7. Graph Construction → data/processed/memory_graph.json  (nodes + edges)
                      → data/processed/issue_graph.html   (PyVis)
  ↓
8. Retrieval API      → data/processed/example_context_packs.json
  ↓
9. Visualization UI   → http://localhost:8000/
```

Each stage is independently re-runnable and produces persisted artifacts.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with `llama3` model (for extraction stage only)

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama & pull model for LLM extraction
ollama pull llama3
```

### Run the Full Pipeline

```bash
# Run all stages (skip extraction if Ollama isn't running)
python run_pipeline.py --skip-extraction

# Run all stages including LLM extraction
python run_pipeline.py

# Run specific stages
python run_pipeline.py transform temporal dedup conflicts graph

# Include fresh data ingestion from GitHub
python run_pipeline.py --include-ingest --skip-extraction
```

### Launch the API & Visualization

```bash
python -m api.server
# → Open http://localhost:8000/
```

---

## Ontology / Schema

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| **Person** | GitHub user | tiangolo, dependabot |
| **Issue** | GitHub issue | issue_15028 |
| **PullRequest** | Pull request | issue_15027 |
| **Label** | Issue label | bug, enhancement |
| **Milestone** | Project milestone | v0.100.0 |
| **Component** | Software component | FastAPI, Starlette |
| **Team** | Organizational team | fastapi-maintainers |

### Claim Types

| Type | Description |
|------|-------------|
| **StatusChange** | Issue opened/closed/reopened |
| **Assignment** | Assigned/unassigned to person |
| **Reference** | Cross-reference to another issue/PR |
| **Decision** | Explicit decision stated |
| **DependsOn** | Dependency relationship |
| **Duplicates** | Duplicate issue relationship |
| **Blocks** | Blocking relationship |
| **Fixes** | PR fixes an issue |
| **LabelAction** | Label added/removed |
| **Performance** | Performance-related claim |
| **OwnershipChange** | Ownership transferred |
| **Generic** | Catch-all for unclassified claims |

### Evidence

Every claim carries one or more **Evidence** records:
- `source_id` — deterministic ID
- `excerpt` — exact text span supporting the claim
- `char_offset_start` / `char_offset_end` — character offsets in source
- `timestamp` — when the evidence was created
- `source_url` — direct link to GitHub

### Extraction Metadata

Every extracted claim carries version info:
- `model_name` / `model_version`
- `prompt_hash` — SHA-256 of the prompt template
- `schema_version` — ontology version (semver)
- `confidence` — 0.0–1.0

---

## Structured Extraction

- **Prompt**: Typed JSON extraction prompt with explicit field specifications
- **Validation**: Pydantic validation → invalid types normalized, confidence gated
- **Retry**: Up to 2 retries on parse failure
- **Quality gates**: Minimum confidence threshold (0.3), evidence required, human review flag below 0.4
- **Signal filtering**: Only processes comments with cross-references or decision keywords
- **Versioning**: Prompt hash + schema version stored per claim for reproducible backfill

---

## Deduplication & Canonicalization

Three-layer dedup:

1. **Artifact dedup** — near-identical comments within the same issue (cosine similarity ≥ 0.95)
2. **Entity canonicalization** — merge similar person names, maintain alias maps
3. **Claim dedup** — cluster semantically equivalent claims (similarity ≥ 0.85), merge evidence sets

### Reversibility

- Every merge recorded in `merge_log.json` with:
  - Pre-merge snapshot for rollback
  - Reason, similarity score, timestamp, who triggered it
- `/api/merges/{id}/reverse` endpoint to undo merges
- Merge status: `active` or `reversed`

---

## Memory Graph Design

### Bi-Temporal Model

| Time Axis | Description |
|-----------|-------------|
| **event_time** | When the real-world event happened |
| **valid_from** | When the system started considering this current |
| **valid_until** | When superseded (`null` = still current) |
| **recorded_at** | When we ingested it |

### State Query

`GET /api/state/{issue_id}?as_of=2024-01-15T00:00:00Z` returns the believed state at that time.

### Updates & Idempotency

- All source IDs are deterministic (content-addressed hashes)
- Re-running any stage produces the same output
- Incremental ingestion merges with existing data
- Checkpoint files track progress

### Permissions (Conceptual)

The `MemoryGraphStore` supports access-control filtering: each entity/claim carries `source_ids` that map back to original artifacts. A permission layer would filter results to only include items where the user has access to all underlying sources.

### Observability

`GET /api/metrics` returns:
- Entity/claim/event counts
- Average confidence
- Evidence coverage ratio
- Merge count & conflict count

---

## Retrieval & Grounding

`POST /api/retrieve` with `{"question": "..."}` returns a **ContextPack**:

1. **Keyword extraction** from question
2. **Entity resolution** — names → canonical entities (keyword + semantic)
3. **Claim search** — type/subject/predicate matching + embedding similarity
4. **Evidence expansion** — follow evidence pointers, include related events
5. **Conflict surfacing** — show both sides of contradictions
6. **Citation formatting** — every item grounded with source_id + excerpt + URL

### Ambiguity Handling

When entity resolution is uncertain, multiple interpretations are returned with relevance scores. Conflicting claims are surfaced explicitly rather than hidden.

---

## Visualization

Interactive web UI at `http://localhost:8000/`:

- **Graph view** — vis.js force-directed graph with color-coded node types
- **Entity browser** — searchable sidebar with type filters
- **Evidence panel** — click any entity/claim to see evidence, citations, confidence
- **Retrieval tab** — ask questions, get grounded answers with citations
- **Conflicts tab** — browse state irregularities, contradictions, ownership changes
- **Merges tab** — inspect all merges, reverse any merge with one click

---

## Conflict Detection

Four conflict detectors:

1. **State irregularities** — reopens, redundant transitions, rapid flapping
2. **Contradictory claims** — same subject+predicate with different values
3. **Temporal anomalies** — out-of-order events
4. **Ownership changes** — reassignment patterns

Each conflict carries severity (high/medium/low) and linked evidence.

---

## Layer10 Adaptation

### Ontology Changes

| Current (GitHub) | Layer10 Target |
|-------------------|----------------|
| Issue | Jira/Linear ticket |
| Comment | Slack message, email body |
| Person (GitHub login) | Person (email + Slack + Jira identity) |
| Label | Jira component, Linear label |
| PullRequest | Code review, merge event |

New entity types: **Email Thread**, **Slack Channel**, **Document**, **Meeting**, **Customer**.

New claim types: **Escalation**, **SLA Violation**, **Requirement Change**, **Approval**.

### Extraction Contract

- Same Pydantic schema, different prompt templates per source type
- Email: extract decisions, action items, participants
- Slack: thread-aware extraction with reaction signals
- Jira: leverage structured fields (status, assignee, priority) as high-confidence claims without LLM

### Dedup Strategy

- **Cross-source identity resolution**: match `user@company.com` ↔ `@user` (Slack) ↔ `user.name` (Jira) using organizational directory
- **Artifact dedup**: email forwarding/quoting detection; Slack → Jira cross-posts
- **Claim dedup**: same fact stated in email + codified in Jira ticket

### Grounding & Safety

- **Provenance chain**: evidence pointers include source system + message ID + timestamp
- **Deletions/redactions**: soft-delete claims when source is deleted; quarantine pending review
- **PII**: redaction-aware extraction; claims derived from redacted sources are flagged

### Permissions

- Each claim's evidence carries source ACLs (channel membership, Jira project access, email recipients)
- Retrieval filters results: user sees only claims grounded in accessible sources
- Shared claims (grounded in multiple sources) remain visible if user has access to at least one source

### Operational Reality

- **Scaling**: Event-driven ingestion (webhooks) → Kafka → extraction workers; graph in Postgres with adjacency tables or Neo4j
- **Cost**: LLM extraction batched; embedding model cached; incremental processing only on deltas
- **Incremental updates**: Webhook listeners for Slack/Jira/email; idempotent upsert into graph
- **Evaluation**: Golden-set of manually labeled claims for regression testing; confidence drift monitoring; alert on evidence coverage drops

---

## Project Structure

```
├── run_pipeline.py            # Pipeline orchestrator
├── requirements.txt           # Python dependencies
├── api/
│   └── server.py              # FastAPI REST API
├── ingestion/
│   └── github_ingest.py       # GitHub data fetching
├── transformation/
│   └── github_to_events.py    # Raw → canonical events
├── extraction/
│   └── claim_extractor.py     # LLM-based claim extraction
├── dedup/
│   └── identity_resolution.py # 3-layer dedup + merge tracking
├── graph/
│   ├── temporal_model.py      # Bi-temporal state history
│   ├── conflict_detection.py  # Multi-type conflict detection
│   └── build_graph.py         # Graph construction + PyVis
├── memory/
│   ├── schema.py              # Pydantic ontology models
│   └── graph_store.py         # In-memory graph store
├── retrieval/
│   └── retrieval_engine.py    # Question → ContextPack
├── ui/
│   └── index.html             # Interactive visualization
├── data/
│   ├── raw/                   # Ingested GitHub data
│   └── processed/             # All pipeline outputs
└── docs/
    └── ARCHITECTURE.md        # Technical architecture
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM | Ollama (llama3, local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Validation | Pydantic v2 |
| API | FastAPI + Uvicorn |
| Graph viz | vis.js (browser), PyVis (static) |
| Clustering | scikit-learn (cosine similarity) |
| Storage | JSON files (portable, inspectable) |
