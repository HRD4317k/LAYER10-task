"""
Graph Construction — Layer10 Memory Pipeline
==============================================
Builds a rich interactive graph from entities, claims, and evidence.

Two outputs:
  1. data/processed/memory_graph.json — machine-readable graph (nodes + edges)
  2. data/processed/issue_graph.html  — interactive PyVis visualization

Node types: Issue, Person, Label, Claim (with color coding)
Edge types: authored, assigned_to, labeled, references, depends_on, etc.
Every edge links back to evidence.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from pyvis.network import Network

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ENTITIES_FILE = "data/processed/entities.json"
CLAIMS_FILE = "data/processed/canonical_claims.json"
EVENTS_FILE = "data/processed/events.json"
STATE_HISTORY_FILE = "data/processed/issue_state_history.json"
CONFLICTS_FILE = "data/processed/state_conflicts.json"
GRAPH_JSON = "data/processed/memory_graph.json"
GRAPH_HTML = "data/processed/issue_graph.html"

# Color scheme
COLORS = {
    "Issue": "#4A90D9",
    "Person": "#50C878",
    "Label": "#F5A623",
    "PullRequest": "#9B59B6",
    "Claim": "#E74C3C",
    "Milestone": "#1ABC9C",
    "default": "#95A5A6",
}


def _node_color(entity_type: str) -> str:
    return COLORS.get(entity_type, COLORS["default"])


def run_graph_build() -> str:
    """Build the memory graph from all processed data."""

    # Load data
    entities = []
    if os.path.exists(ENTITIES_FILE):
        with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
            entities = json.load(f)

    claims = []
    if os.path.exists(CLAIMS_FILE):
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            claims = json.load(f)

    events = []
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

    state_history = {}
    if os.path.exists(STATE_HISTORY_FILE):
        with open(STATE_HISTORY_FILE, "r", encoding="utf-8") as f:
            state_history = json.load(f)

    conflicts = {}
    if os.path.exists(CONFLICTS_FILE):
        with open(CONFLICTS_FILE, "r", encoding="utf-8") as f:
            conflicts = json.load(f)

    # Build graph data structure
    nodes: List[Dict] = []
    edges: List[Dict] = []
    added_nodes: Set[str] = set()

    # Add entity nodes
    for ent in entities:
        nid = ent["entity_id"]
        if nid in added_nodes:
            continue
        nodes.append({
            "id": nid,
            "label": ent["canonical_name"],
            "type": ent["entity_type"],
            "color": _node_color(ent["entity_type"]),
            "properties": ent.get("properties", {}),
            "aliases": ent.get("aliases", []),
            "first_seen": ent.get("first_seen"),
            "last_seen": ent.get("last_seen"),
        })
        added_nodes.add(nid)

    # Build entity lookup
    entity_by_name: Dict[str, str] = {}
    for ent in entities:
        entity_by_name[ent["canonical_name"].lower()] = ent["entity_id"]
        for alias in ent.get("aliases", []):
            entity_by_name[alias.lower()] = ent["entity_id"]

    def _resolve(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return entity_by_name.get(name.lower()) or entity_by_name.get(
            name.replace("issue_", "").lower()
        )

    # Add edges from events (structural relationships)
    for evt in events:
        etype = evt["event_type"]
        subject = evt["subject"]
        actor = evt.get("actor", "")
        source_id = evt.get("source_id", "")
        timestamp = evt.get("timestamp", "")

        subject_id = _resolve(subject)
        actor_id = _resolve(actor)

        if etype == "IssueCreated" and subject_id and actor_id:
            edges.append({
                "from": actor_id,
                "to": subject_id,
                "label": "created",
                "type": "created",
                "source_id": source_id,
                "timestamp": timestamp,
                "evidence": evt.get("evidence", ""),
            })

        elif etype == "AssignedTo" and subject_id:
            extra = evt.get("extra", {})
            assignee_id = _resolve(extra.get("assignee"))
            if assignee_id:
                edges.append({
                    "from": subject_id,
                    "to": assignee_id,
                    "label": "assigned_to",
                    "type": "assigned_to",
                    "source_id": source_id,
                    "timestamp": timestamp,
                    "evidence": evt.get("evidence", ""),
                })

        elif etype == "LabelAdded" and subject_id:
            extra = evt.get("extra", {})
            label_id = _resolve(extra.get("label"))
            if label_id:
                edges.append({
                    "from": subject_id,
                    "to": label_id,
                    "label": "labeled",
                    "type": "labeled",
                    "source_id": source_id,
                    "timestamp": timestamp,
                    "evidence": evt.get("evidence", ""),
                })

        elif etype == "CommentAdded" and subject_id and actor_id:
            edges.append({
                "from": actor_id,
                "to": subject_id,
                "label": "commented",
                "type": "commented",
                "source_id": source_id,
                "timestamp": timestamp,
                "evidence": evt.get("evidence", "")[:200],
            })

    # Add edges from claims
    for claim in claims:
        subj = claim.get("subject")
        obj = claim.get("object")
        subject_id = _resolve(subj)
        object_id = _resolve(obj) if obj else None

        if not subject_id:
            continue

        # If no object entity, create a claim node
        if not object_id and obj:
            claim_nid = f"claim_{claim.get('claim_id', '')}"
            if claim_nid not in added_nodes:
                nodes.append({
                    "id": claim_nid,
                    "label": f"{claim.get('predicate', '')} → {obj}",
                    "type": "Claim",
                    "color": COLORS["Claim"],
                    "properties": {
                        "claim_type": claim.get("claim_type"),
                        "confidence": claim.get("confidence"),
                        "value": claim.get("value"),
                    },
                })
                added_nodes.add(claim_nid)
            object_id = claim_nid

        if object_id:
            edges.append({
                "from": subject_id,
                "to": object_id,
                "label": claim.get("predicate", "related_to"),
                "type": claim.get("claim_type", "Generic"),
                "confidence": claim.get("confidence", 0.5),
                "evidence": [e.get("excerpt", "") for e in claim.get("evidence", [])],
                "source_ids": claim.get("source_ids", []),
                "status": claim.get("status", "active"),
            })

    # Persist graph JSON
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_count": len(entities),
            "claim_count": len(claims),
        },
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    # Build PyVis visualization
    net = Network(height="800px", width="100%", directed=True, bgcolor="#1a1a2e")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node in nodes:
        net.add_node(
            node["id"],
            label=node["label"],
            color=node["color"],
            title=json.dumps({
                "type": node["type"],
                "aliases": node.get("aliases", []),
                "properties": node.get("properties", {}),
            }, indent=2),
            shape="dot" if node["type"] in ("Person", "Label") else "box",
            size=20 if node["type"] == "Issue" else 15,
        )

    for edge in edges:
        title = edge.get("evidence", "")
        if isinstance(title, list):
            title = "\n".join(title[:3])
        net.add_edge(
            edge["from"],
            edge["to"],
            label=edge.get("label", ""),
            title=str(title)[:300],
            color="#666" if edge.get("status") == "active" else "#999",
        )

    net.write_html(GRAPH_HTML, open_browser=False)

    log.info("Graph built: %d nodes, %d edges → %s / %s",
             len(nodes), len(edges), GRAPH_JSON, GRAPH_HTML)
    return GRAPH_JSON


if __name__ == "__main__":
    run_graph_build()
