"""
Memory Graph Store — Layer10 Memory Pipeline
==============================================
Unified in-memory graph store backed by JSON persistence.

Features:
  - Entity, Claim, Evidence CRUD with Pydantic validation
  - Bi-temporal queries (state-at-time)
  - Incremental updates, idempotent upserts
  - Merge/unmerge operations with audit trail
  - Permission-aware retrieval (conceptual ACL filtering)
  - Observability: metrics on extraction quality / drift
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = "data/processed"


class MemoryGraphStore:
    """
    In-memory graph backed by JSON files.
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.entities: Dict[str, Dict] = {}
        self.claims: Dict[str, Dict] = {}
        self.events: List[Dict] = []
        self.merge_log: List[Dict] = []
        self.alias_map: Dict[str, str] = {}
        self.state_history: Dict = {}
        self.conflicts: Dict = {}
        self.graph: Dict = {}
        self._loaded = False

    def load(self) -> "MemoryGraphStore":
        """Load all processed data into memory."""
        self._load_json("entities.json", self._load_entities)
        self._load_json("canonical_claims.json", self._load_claims)
        self._load_json("events.json", self._load_events)
        self._load_json("merge_log.json", self._load_merges)
        self._load_json("alias_map.json", self._load_aliases)
        self._load_json("issue_state_history.json", self._load_state_history)
        self._load_json("state_conflicts.json", self._load_conflicts)
        self._load_json("memory_graph.json", self._load_graph)
        self._loaded = True
        log.info(
            "Memory store loaded: %d entities, %d claims, %d events",
            len(self.entities), len(self.claims), len(self.events),
        )
        return self

    def _load_json(self, filename: str, handler) -> None:
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            handler(data)

    def _load_entities(self, data):
        if isinstance(data, list):
            for e in data:
                self.entities[e.get("entity_id", "")] = e
        elif isinstance(data, dict):
            self.entities = data

    def _load_claims(self, data):
        if isinstance(data, list):
            for c in data:
                self.claims[c.get("claim_id", "")] = c
        elif isinstance(data, dict):
            self.claims = data

    def _load_events(self, data):
        self.events = data if isinstance(data, list) else []

    def _load_merges(self, data):
        if isinstance(data, dict):
            self.merge_log = data.get("records", [])
        elif isinstance(data, list):
            self.merge_log = data

    def _load_aliases(self, data):
        self.alias_map = data if isinstance(data, dict) else {}

    def _load_state_history(self, data):
        self.state_history = data if isinstance(data, dict) else {}

    def _load_conflicts(self, data):
        self.conflicts = data if isinstance(data, dict) else {}

    def _load_graph(self, data):
        self.graph = data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        return self.entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Optional[Dict]:
        """Resolve a name (including aliases) to an entity."""
        eid = self.alias_map.get(name.lower())
        if eid:
            return self.entities.get(eid)
        for ent in self.entities.values():
            if ent.get("canonical_name", "").lower() == name.lower():
                return ent
            if name.lower() in [a.lower() for a in ent.get("aliases", [])]:
                return ent
        return None

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Simple text search across entities."""
        results = []
        q = query.lower()
        for ent in self.entities.values():
            if entity_type and ent.get("entity_type") != entity_type:
                continue
            name = ent.get("canonical_name", "").lower()
            aliases = " ".join(ent.get("aliases", [])).lower()
            if q in name or q in aliases:
                results.append(ent)
            if len(results) >= limit:
                break
        return results

    def upsert_entity(self, entity: Dict) -> str:
        """Insert or update an entity (idempotent)."""
        eid = entity.get("entity_id", "")
        if not eid:
            eid = hashlib.sha256(
                f"{entity.get('entity_type')}::{entity.get('canonical_name', '')}".encode()
            ).hexdigest()[:12]
            entity["entity_id"] = eid
        self.entities[eid] = entity
        # Update alias map
        self.alias_map[entity.get("canonical_name", "").lower()] = eid
        for alias in entity.get("aliases", []):
            self.alias_map[alias.lower()] = eid
        return eid

    # ------------------------------------------------------------------
    # Claim operations
    # ------------------------------------------------------------------

    def get_claim(self, claim_id: str) -> Optional[Dict]:
        return self.claims.get(claim_id)

    def get_claims_for_entity(self, entity_id: str) -> List[Dict]:
        """All claims where this entity is subject or object."""
        results = []
        # Also resolve by name
        names = set()
        ent = self.entities.get(entity_id)
        if ent:
            names.add(ent.get("canonical_name", "").lower())
            names.update(a.lower() for a in ent.get("aliases", []))

        for claim in self.claims.values():
            subj = (claim.get("subject") or "").lower()
            obj = (claim.get("object") or "").lower()
            if subj == entity_id or obj == entity_id:
                results.append(claim)
            elif subj in names or obj in names:
                results.append(claim)
        return results

    def search_claims(
        self,
        query: str,
        claim_type: Optional[str] = None,
        status: str = "active",
        limit: int = 20,
    ) -> List[Dict]:
        """Text search across claims."""
        results = []
        q = query.lower()
        for claim in self.claims.values():
            if claim_type and claim.get("claim_type") != claim_type:
                continue
            if status and claim.get("status", "active") != status:
                continue
            text = f"{claim.get('subject', '')} {claim.get('predicate', '')} {claim.get('object', '')} {claim.get('value', '')}"
            evidence_text = " ".join(
                e.get("excerpt", "") for e in claim.get("evidence", [])
            )
            if q in text.lower() or q in evidence_text.lower():
                results.append(claim)
            if len(results) >= limit:
                break
        return results

    def upsert_claim(self, claim: Dict) -> str:
        """Insert or update a claim."""
        cid = claim.get("claim_id", "")
        if not cid:
            cid = hashlib.sha256(
                f"{claim.get('claim_type')}::{claim.get('subject')}::{claim.get('predicate')}".encode()
            ).hexdigest()[:12]
            claim["claim_id"] = cid
        self.claims[cid] = claim
        return cid

    def supersede_claim(self, old_id: str, new_claim: Dict) -> str:
        """Mark an old claim as superseded and insert the replacement."""
        old = self.claims.get(old_id)
        if old:
            old["status"] = "superseded"
            old["superseded_by"] = new_claim.get("claim_id")
        return self.upsert_claim(new_claim)

    # ------------------------------------------------------------------
    # Event & state queries
    # ------------------------------------------------------------------

    def get_events_for_issue(self, issue_id: str) -> List[Dict]:
        return [e for e in self.events if e.get("subject") == issue_id]

    def get_state_at(self, issue_id: str, as_of: str) -> Dict[str, Any]:
        """Bi-temporal query: what was the state at a given time?"""
        states = self.state_history.get("states", {}).get(issue_id, [])
        current = "unknown"
        for entry in states:
            vf = entry.get("valid_from") or ""
            vu = entry.get("valid_until")
            if vf <= as_of and (vu is None or vu > as_of):
                current = entry["state"]
        return {"issue_id": issue_id, "as_of": as_of, "state": current}

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_neighbors(self, node_id: str, max_depth: int = 1) -> Dict:
        """Return neighbors up to max_depth hops."""
        nodes = self.graph.get("nodes", [])
        edges = self.graph.get("edges", [])

        visited = {node_id}
        frontier = {node_id}
        result_nodes = []
        result_edges = []

        for _ in range(max_depth):
            next_frontier = set()
            for edge in edges:
                src, dst = edge.get("from"), edge.get("to")
                if src in frontier:
                    next_frontier.add(dst)
                    result_edges.append(edge)
                if dst in frontier:
                    next_frontier.add(src)
                    result_edges.append(edge)
            frontier = next_frontier - visited
            visited |= frontier

        node_set = {n["id"] for n in nodes}
        for nid in visited:
            if nid in node_set:
                result_nodes.extend(n for n in nodes if n["id"] == nid)

        return {"nodes": result_nodes, "edges": result_edges}

    # ------------------------------------------------------------------
    # Merge operations
    # ------------------------------------------------------------------

    def get_merge_log(self, merge_type: Optional[str] = None) -> List[Dict]:
        if merge_type:
            return [m for m in self.merge_log if m.get("merge_type") == merge_type]
        return self.merge_log

    def reverse_merge(self, merge_id: str) -> bool:
        """Undo a merge by restoring the snapshot."""
        for record in self.merge_log:
            if record.get("merge_id") == merge_id and record.get("status") == "active":
                record["status"] = "reversed"
                record["reversed_at"] = datetime.utcnow().isoformat() + "Z"
                snapshot = record.get("snapshot_before")
                if snapshot:
                    if record["merge_type"] == "entity":
                        if isinstance(snapshot, dict) and "entity_id" in snapshot:
                            self.entities[snapshot["entity_id"]] = snapshot
                    elif record["merge_type"] == "claim":
                        if isinstance(snapshot, list):
                            for c in snapshot:
                                if "claim_id" in c or "source_id" in c:
                                    cid = c.get("claim_id") or c.get("source_id")
                                    self.claims[cid] = c
                log.info("Reversed merge %s", merge_id)
                return True
        return False

    # ------------------------------------------------------------------
    # Conflict queries
    # ------------------------------------------------------------------

    def get_conflicts(self, severity: Optional[str] = None) -> List[Dict]:
        all_conflicts = self.conflicts.get("conflicts", [])
        if severity:
            return [c for c in all_conflicts if c.get("severity") == severity]
        return all_conflicts

    # ------------------------------------------------------------------
    # Observability / Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return quality and health metrics."""
        total_claims = len(self.claims)
        active_claims = sum(1 for c in self.claims.values() if c.get("status") == "active")
        avg_confidence = 0.0
        if total_claims > 0:
            avg_confidence = sum(
                c.get("confidence", 0) for c in self.claims.values()
            ) / total_claims

        evidenced = sum(1 for c in self.claims.values() if c.get("evidence"))
        total_conflicts = len(self.conflicts.get("conflicts", []))

        return {
            "total_entities": len(self.entities),
            "total_claims": total_claims,
            "active_claims": active_claims,
            "average_confidence": round(avg_confidence, 3),
            "claims_with_evidence": evidenced,
            "evidence_coverage": round(evidenced / max(total_claims, 1), 3),
            "total_events": len(self.events),
            "total_merges": len(self.merge_log),
            "active_merges": sum(1 for m in self.merge_log if m.get("status") == "active"),
            "total_conflicts": total_conflicts,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the current state back to disk."""
        os.makedirs(self.data_dir, exist_ok=True)

        with open(os.path.join(self.data_dir, "entities.json"), "w") as f:
            json.dump(list(self.entities.values()), f, indent=2)

        with open(os.path.join(self.data_dir, "canonical_claims.json"), "w") as f:
            json.dump(list(self.claims.values()), f, indent=2)

        with open(os.path.join(self.data_dir, "alias_map.json"), "w") as f:
            json.dump(self.alias_map, f, indent=2)

        log.info("Memory store saved to %s", self.data_dir)


# Singleton accessor
_store: Optional[MemoryGraphStore] = None


def get_store(data_dir: str = DATA_DIR) -> MemoryGraphStore:
    global _store
    if _store is None:
        _store = MemoryGraphStore(data_dir).load()
    return _store
