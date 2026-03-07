"""
Multi-Level Deduplication & Canonicalization — Layer10 Memory Pipeline
======================================================================
Three dedup layers:
  1. Artifact dedup — identical / near-identical messages
  2. Entity canonicalization — people, issues, labels (aliases, renames)
  3. Claim dedup — merge semantically equivalent claims, keep evidence sets

Merge tracking:
  - Every merge is recorded with reason, similarity score, timestamp
  - Pre-merge snapshots are stored for reversibility
  - MergeRecord objects support undo

Uses sentence-transformer embeddings + cosine similarity.
"""

import hashlib
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAIMS_FILE = "data/processed/claims.json"
EVENTS_FILE = "data/processed/events.json"
ENTITIES_OUTPUT = "data/processed/entities.json"
CANONICAL_CLAIMS_OUTPUT = "data/processed/canonical_claims.json"
MERGE_LOG_OUTPUT = "data/processed/merge_log.json"

ENTITY_SIMILARITY_THRESHOLD = 0.90
CLAIM_SIMILARITY_THRESHOLD = 0.85
ARTIFACT_SIMILARITY_THRESHOLD = 0.95

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Merge record
# ---------------------------------------------------------------------------

def _merge_record(
    merge_type: str,
    source_ids: List[str],
    target_id: str,
    reason: str,
    score: Optional[float] = None,
    snapshot: Optional[Dict] = None,
) -> Dict[str, Any]:
    return {
        "merge_id": str(uuid.uuid4())[:12],
        "merge_type": merge_type,
        "source_ids": source_ids,
        "target_id": target_id,
        "reason": reason,
        "similarity_score": score,
        "merged_at": datetime.utcnow().isoformat() + "Z",
        "merged_by": "system",
        "status": "active",
        "snapshot_before": snapshot,
    }


# ---------------------------------------------------------------------------
# 1. Artifact dedup (near-identical comments / events)
# ---------------------------------------------------------------------------

def dedup_artifacts(events: List[Dict], model: SentenceTransformer) -> Tuple[List[Dict], List[Dict]]:
    """Remove near-duplicate events (same subject + very similar evidence)."""
    log.info("Running artifact dedup on %d events …", len(events))
    merges: List[Dict] = []

    # Group by subject for efficiency
    by_subject: Dict[str, List[int]] = defaultdict(list)
    for i, evt in enumerate(events):
        by_subject[evt.get("subject", "")].append(i)

    keep_mask = [True] * len(events)

    for subject, indices in by_subject.items():
        if len(indices) < 2:
            continue

        # Only dedup CommentAdded within same subject
        comment_indices = [i for i in indices if events[i]["event_type"] == "CommentAdded"]
        if len(comment_indices) < 2:
            continue

        texts = [events[i].get("evidence", "") for i in comment_indices]
        if not any(texts):
            continue

        embeddings = model.encode(texts)
        sim = cosine_similarity(embeddings)

        for a in range(len(comment_indices)):
            if not keep_mask[comment_indices[a]]:
                continue
            for b in range(a + 1, len(comment_indices)):
                if not keep_mask[comment_indices[b]]:
                    continue
                if sim[a][b] >= ARTIFACT_SIMILARITY_THRESHOLD:
                    # Keep the earlier one
                    drop_idx = comment_indices[b]
                    keep_idx = comment_indices[a]
                    keep_mask[drop_idx] = False
                    merges.append(_merge_record(
                        "artifact",
                        [events[drop_idx]["source_id"]],
                        events[keep_idx]["source_id"],
                        f"Near-duplicate comment (sim={sim[a][b]:.3f})",
                        score=float(sim[a][b]),
                        snapshot=events[drop_idx],
                    ))

    deduped = [e for i, e in enumerate(events) if keep_mask[i]]
    log.info("Artifact dedup: %d → %d events (%d duplicates removed)",
             len(events), len(deduped), len(events) - len(deduped))
    return deduped, merges


# ---------------------------------------------------------------------------
# 2. Entity canonicalization
# ---------------------------------------------------------------------------

def canonicalize_entities(
    events: List[Dict],
    claims: List[Dict],
    model: SentenceTransformer,
) -> Tuple[List[Dict], Dict[str, str], List[Dict]]:
    """
    Extract and canonicalize entities (persons, issues, labels).
    Returns: (entities, alias_map, merge_records)
    """
    log.info("Running entity canonicalization …")
    merges: List[Dict] = []

    # Collect raw entity mentions
    raw_entities: Dict[str, Dict] = {}  # key → entity info

    # Persons from events
    for evt in events:
        actor = evt.get("actor", "")
        if actor and actor != "system" and actor != "unknown":
            key = f"Person::{actor.lower()}"
            if key not in raw_entities:
                raw_entities[key] = {
                    "entity_type": "Person",
                    "canonical_name": actor,
                    "aliases": set(),
                    "source_ids": set(),
                    "first_seen": evt.get("timestamp"),
                    "last_seen": evt.get("timestamp"),
                }
            raw_entities[key]["aliases"].add(actor)
            raw_entities[key]["source_ids"].add(evt.get("source_id", ""))
            ts = evt.get("timestamp")
            if ts:
                if not raw_entities[key]["first_seen"] or ts < raw_entities[key]["first_seen"]:
                    raw_entities[key]["first_seen"] = ts
                if not raw_entities[key]["last_seen"] or ts > raw_entities[key]["last_seen"]:
                    raw_entities[key]["last_seen"] = ts

    # Issues from events
    # First pass: determine which subjects are pull requests
    pr_subjects = set()
    for evt in events:
        subj = evt.get("subject", "")
        if subj.startswith("issue_") and evt.get("extra", {}).get("is_pull_request"):
            pr_subjects.add(subj)

    # Issues / PullRequests from events
    for evt in events:
        subj = evt.get("subject", "")
        if subj.startswith("issue_"):
            etype = "PullRequest" if subj in pr_subjects else "Issue"
            key = f"{etype}::{subj}"
            if key not in raw_entities:
                raw_entities[key] = {
                    "entity_type": etype,
                    "canonical_name": subj,
                    "aliases": set(),
                    "source_ids": set(),
                    "first_seen": evt.get("timestamp"),
                    "last_seen": evt.get("timestamp"),
                    "properties": {},
                }
            raw_entities[key]["source_ids"].add(evt.get("source_id", ""))
            # Capture labels from extra
            if evt.get("extra", {}).get("labels"):
                raw_entities[key].setdefault("properties", {})["labels"] = evt["extra"]["labels"]
            ts = evt.get("timestamp")
            if ts:
                if not raw_entities[key].get("first_seen") or ts < raw_entities[key]["first_seen"]:
                    raw_entities[key]["first_seen"] = ts
                if not raw_entities[key].get("last_seen") or ts > raw_entities[key]["last_seen"]:
                    raw_entities[key]["last_seen"] = ts

    # Labels from events
    for evt in events:
        extra = evt.get("extra", {})
        label = extra.get("label", "")
        if label:
            key = f"Label::{label.lower()}"
            if key not in raw_entities:
                raw_entities[key] = {
                    "entity_type": "Label",
                    "canonical_name": label,
                    "aliases": set(),
                    "source_ids": set(),
                    "first_seen": evt.get("timestamp"),
                    "last_seen": evt.get("timestamp"),
                }
            raw_entities[key]["aliases"].add(label)
            raw_entities[key]["source_ids"].add(evt.get("source_id", ""))

    # Convert sets to lists and assign IDs
    entities_list: List[Dict] = []
    alias_map: Dict[str, str] = {}  # alias → canonical entity_id

    for key, ent in raw_entities.items():
        eid = hashlib.sha256(key.encode()).hexdigest()[:12]
        entity = {
            "entity_id": eid,
            "entity_type": ent["entity_type"],
            "canonical_name": ent["canonical_name"],
            "aliases": sorted(ent.get("aliases", set()) if isinstance(ent.get("aliases"), set) else []),
            "source_ids": sorted(ent.get("source_ids", set()) if isinstance(ent.get("source_ids"), set) else []),
            "first_seen": ent.get("first_seen"),
            "last_seen": ent.get("last_seen"),
            "properties": ent.get("properties", {}),
        }
        entities_list.append(entity)
        for alias in entity["aliases"]:
            alias_map[alias.lower()] = eid
        alias_map[entity["canonical_name"].lower()] = eid

    # Semantic merge pass: find similar person names
    person_entities = [e for e in entities_list if e["entity_type"] == "Person"]
    if len(person_entities) > 1:
        names = [e["canonical_name"] for e in person_entities]
        embeddings = model.encode(names)
        sim = cosine_similarity(embeddings)

        merged_into: Dict[int, int] = {}
        for i in range(len(person_entities)):
            if i in merged_into:
                continue
            for j in range(i + 1, len(person_entities)):
                if j in merged_into:
                    continue
                if sim[i][j] >= ENTITY_SIMILARITY_THRESHOLD:
                    target = person_entities[i]
                    source = person_entities[j]
                    merged_into[j] = i

                    # Merge aliases
                    target["aliases"] = sorted(set(target["aliases"]) | set(source["aliases"]))
                    target["source_ids"] = sorted(set(target["source_ids"]) | set(source["source_ids"]))

                    for alias in source["aliases"]:
                        alias_map[alias.lower()] = target["entity_id"]
                    alias_map[source["canonical_name"].lower()] = target["entity_id"]

                    merges.append(_merge_record(
                        "entity",
                        [source["entity_id"]],
                        target["entity_id"],
                        f"Similar person names: '{source['canonical_name']}' ≈ '{target['canonical_name']}'",
                        score=float(sim[i][j]),
                        snapshot=source,
                    ))

        # Remove merged entities
        merged_indices = set(merged_into.keys())
        person_entities = [e for i, e in enumerate(person_entities) if i not in merged_indices]

    # Rebuild entities list
    non_persons = [e for e in entities_list if e["entity_type"] != "Person"]
    entities_list = person_entities + non_persons

    log.info("Canonicalized %d entities (%d merges)", len(entities_list), len(merges))
    return entities_list, alias_map, merges


# ---------------------------------------------------------------------------
# 3. Claim dedup
# ---------------------------------------------------------------------------

def dedup_claims(
    claims: List[Dict], model: SentenceTransformer
) -> Tuple[List[Dict], List[Dict]]:
    """
    Merge semantically equivalent claims, preserving evidence sets.
    Returns: (canonical_claims, merge_records)
    """
    log.info("Running claim dedup on %d claims …", len(claims))
    merges: List[Dict] = []

    if not claims:
        return [], []

    # Build text representations
    texts = []
    for c in claims:
        text = f"{c.get('claim_type', '')} {c.get('subject', '')} {c.get('predicate', '')} {c.get('object', '')} {c.get('value', '')}"
        texts.append(text.strip())

    embeddings = model.encode(texts)
    sim = cosine_similarity(embeddings)

    # Greedy clustering
    visited: Set[int] = set()
    clusters: List[List[int]] = []

    for i in range(len(claims)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(claims)):
            if j in visited:
                continue
            if sim[i][j] >= CLAIM_SIMILARITY_THRESHOLD:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    # Merge each cluster into a canonical claim
    canonical_claims: List[Dict] = []
    for cluster_indices in clusters:
        representative = claims[cluster_indices[0]].copy()

        # Collect all evidence from cluster members
        all_evidence = []
        all_sources = set()
        max_confidence = 0.0

        for idx in cluster_indices:
            c = claims[idx]
            # Collect evidence from nested evidence array (rule-based claims)
            for ev in c.get("evidence", []):
                if ev.get("excerpt"):
                    all_evidence.append({
                        "source_id": ev.get("source_id", c.get("source_id", "")),
                        "source_url": ev.get("source_url", c.get("source_url", "")),
                        "issue_id": ev.get("issue_id", c.get("subject", "")),
                        "excerpt": ev.get("excerpt", ""),
                        "timestamp": ev.get("timestamp", ""),
                        "char_offset_start": ev.get("char_offset_start", 0),
                        "char_offset_end": ev.get("char_offset_end", 0),
                    })
            # Also check flat excerpt field (LLM-based claims)
            if c.get("excerpt") and not c.get("evidence"):
                all_evidence.append({
                    "source_id": c.get("source_id", ""),
                    "issue_id": c.get("issue_id", c.get("subject", "")),
                    "excerpt": c.get("excerpt", ""),
                })
            for sid in c.get("source_ids", []):
                all_sources.add(sid)
            all_sources.add(c.get("source_id", ""))
            max_confidence = max(max_confidence, c.get("confidence", 0.5))

        canonical_id = hashlib.sha256(
            f"{representative.get('claim_type')}::{representative.get('subject')}::{representative.get('predicate')}".encode()
        ).hexdigest()[:12]

        canonical = {
            "claim_id": canonical_id,
            "claim_type": representative.get("claim_type", "Generic"),
            "subject": representative.get("subject"),
            "object": representative.get("object"),
            "predicate": representative.get("predicate"),
            "value": representative.get("value"),
            "confidence": round(max_confidence, 3),
            "evidence": all_evidence,
            "source_ids": sorted(all_sources),
            "support_count": len(cluster_indices),
            "status": "active",
            "extraction_meta": representative.get("extraction_meta"),
        }
        canonical_claims.append(canonical)

        # Log merges for clusters > 1
        if len(cluster_indices) > 1:
            merged_ids = [claims[i].get("source_id", str(i)) for i in cluster_indices[1:]]
            merges.append(_merge_record(
                "claim",
                merged_ids,
                canonical_id,
                f"Semantically equivalent claims (cluster size={len(cluster_indices)})",
                score=float(np.mean([sim[cluster_indices[0]][j] for j in cluster_indices[1:]])),
                snapshot=[claims[i] for i in cluster_indices],
            ))

    log.info("Claim dedup: %d → %d canonical claims (%d merges)",
             len(claims), len(canonical_claims), len(merges))
    return canonical_claims, merges


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_dedup() -> None:
    """Run the full 3-layer dedup pipeline."""
    os.makedirs("data/processed", exist_ok=True)

    # Load model once
    log.info("Loading embedding model …")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_merges: List[Dict] = []

    # Load events
    events = []
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

    # Load claims
    claims = []
    if os.path.exists(CLAIMS_FILE):
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            claims = json.load(f)

    # Layer 1: Artifact dedup
    events, artifact_merges = dedup_artifacts(events, model)
    all_merges.extend(artifact_merges)

    # Layer 2: Entity canonicalization
    entities, alias_map, entity_merges = canonicalize_entities(events, claims, model)
    all_merges.extend(entity_merges)

    # Layer 3: Claim dedup
    canonical_claims, claim_merges = dedup_claims(claims, model)
    all_merges.extend(claim_merges)

    # Persist outputs
    with open(ENTITIES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)

    with open(CANONICAL_CLAIMS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(canonical_claims, f, indent=2)

    with open(MERGE_LOG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "total_merges": len(all_merges),
            "artifact_merges": len(artifact_merges),
            "entity_merges": len(entity_merges),
            "claim_merges": len(claim_merges),
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "records": all_merges,
        }, f, indent=2)

    # Also save alias map for retrieval
    alias_file = "data/processed/alias_map.json"
    with open(alias_file, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=2)

    log.info(
        "Dedup complete: %d entities, %d canonical claims, %d merges logged",
        len(entities), len(canonical_claims), len(all_merges),
    )


if __name__ == "__main__":
    run_dedup()
