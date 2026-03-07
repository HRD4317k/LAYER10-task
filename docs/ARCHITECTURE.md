# Architecture — Layer10 Memory Pipeline

## Execution Model

The system models GitHub discussions as an **event-sourced workflow** with a
bi-temporal memory graph. Each stage is a deterministic transformation that
consumes structured input artifacts and produces persisted output artifacts.

```
Raw Data (GitHub API)
  → Canonical Events             (transformation)
    → Bi-temporal State History   (temporal_model)
      → Claim Extraction          (claim_extractor — LLM bounded)
        → 3-Layer Deduplication   (identity_resolution)
          → Conflict Detection    (conflict_detection)
            → Memory Graph        (build_graph)
              → Retrieval API     (retrieval_engine)
                → Visualization   (ui/index.html)
```

## Design Principles

### Determinism
- Every stage produces identical output for identical input
- Source IDs are content-addressed (SHA-256 hashes)
- No hidden state between stages

### Grounding
- Every claim traces to ≥1 Evidence record
- Evidence carries: source_id, excerpt, char_offsets, timestamp, URL
- Retrieval always returns citations

### Temporal Correctness
- Bi-temporal: event_time (when it happened) vs validity interval (when we consider it current)
- State-at-time queries: "What was the status of issue X on date Y?"
- Claims carry valid_from/valid_until for "it used to be true" vs "it is true now"

### Safe Deduplication
- 3 layers: artifact → entity → claim
- Every merge recorded with pre-merge snapshot
- Merges are reversible via API
- Similarity thresholds are configurable

### Bounded LLM Usage
The LLM is treated as a **bounded extraction component** inside a larger
structured workflow. It is never autonomous:
- JSON-only extraction with explicit schema
- Post-generation Pydantic validation
- Retry on parse failure (max 2)
- Confidence gating (min 0.3)
- Human review flag (below 0.4)

### Observability
- Each stage writes structured output to disk
- Extraction log tracks processed/skipped/failed per run
- Merge log is a full audit trail
- `/api/metrics` endpoint for health monitoring
- Evidence coverage ratio tracked

## Data Flow

### Events (Canonical)
```json
{
  "event_type": "CommentAdded",
  "subject": "issue_15028",
  "actor": "tiangolo",
  "timestamp": "2024-01-15T10:30:00Z",
  "source_id": "a1b2c3d4e5f6",
  "evidence": "...",
  "evidence_full": "...",
  "source_url": "https://github.com/...",
  "char_offset_start": 0,
  "char_offset_end": 200
}
```

### Claims (Extracted + Validated)
```json
{
  "claim_id": "f7e8d9c0b1a2",
  "claim_type": "Reference",
  "subject": "issue_15028",
  "object": "#13399",
  "predicate": "references",
  "value": null,
  "confidence": 0.85,
  "status": "active",
  "evidence": [
    {
      "source_id": "a1b2c3d4e5f6",
      "excerpt": "This relates to #13399",
      "issue_id": "issue_15028"
    }
  ],
  "extraction_meta": {
    "model": "llama3",
    "schema_version": "2.0.0",
    "prompt_hash": "abc123...",
    "extracted_at": "2024-01-15T12:00:00Z"
  }
}
```

### Entities (Canonicalized)
```json
{
  "entity_id": "b3c4d5e6f7a8",
  "entity_type": "Person",
  "canonical_name": "tiangolo",
  "aliases": ["tiangolo", "Sebastián Ramírez"],
  "source_ids": ["..."],
  "first_seen": "2024-01-01T00:00:00Z",
  "last_seen": "2024-01-15T10:30:00Z"
}
```

### Merge Records
```json
{
  "merge_id": "m1n2o3p4q5r6",
  "merge_type": "entity",
  "source_ids": ["entity_a"],
  "target_id": "entity_b",
  "reason": "Similar person names (sim=0.95)",
  "similarity_score": 0.95,
  "merged_at": "2024-01-15T12:00:00Z",
  "status": "active",
  "snapshot_before": { ... }
}
```

## Conflict Taxonomy

| Type | Detection | Severity |
|------|-----------|----------|
| State irregularity | Reopen cycles, redundant transitions, rapid flapping | medium–high |
| Contradictory claims | Same subject+predicate, different values | high |
| Temporal anomaly | Out-of-order event timestamps | low |
| Ownership change | Multiple reassignments | low–medium |

## Permission Model (Conceptual)

```
User → has_access_to → [Source_A, Source_B, ...]
Claim → grounded_in → [Evidence_1, Evidence_2, ...]
Evidence → from_source → Source_X

retrieval_filter(user, claim):
  return any(
    user.has_access_to(ev.source) for ev in claim.evidence
  )
```

## Scaling Considerations

For Layer10's production environment:

| Concern | Approach |
|---------|----------|
| Ingestion | Webhook listeners → message queue (Kafka) → idempotent workers |
| Storage | Postgres with adjacency tables + JSONB for flexibility; optional Neo4j for graph queries |
| Embeddings | Pre-computed, cached in vector store (pgvector / Qdrant) |
| LLM cost | Batch extraction; smaller model for triage, larger for complex claims |
| Updates | Delta processing; only re-extract changed/new artifacts |
| Evaluation | Golden set of ~500 manually labeled claims; CI regression tests |
