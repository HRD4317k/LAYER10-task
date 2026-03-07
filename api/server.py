"""
FastAPI Server — Layer10 Memory Pipeline
==========================================
REST API for querying the memory graph, entities, claims, evidence,
conflicts, merges, and running the full pipeline.

Endpoints:
  GET  /api/health              — Health check + metrics
  GET  /api/entities            — List/search entities
  GET  /api/entities/{id}       — Entity detail + claims
  GET  /api/claims              — List/search claims
  GET  /api/claims/{id}         — Claim detail + evidence
  GET  /api/events              — List events (paginated)
  GET  /api/conflicts           — Conflict report
  GET  /api/merges              — Merge audit log
  POST /api/merges/{id}/reverse — Reverse a merge
  POST /api/retrieve            — Question → ContextPack
  GET  /api/graph               — Full graph data (nodes + edges)
  GET  /api/graph/neighbors/{id} — Subgraph around a node
  GET  /api/state/{issue_id}    — Bi-temporal state query
  GET  /api/metrics             — Observability metrics
  POST /api/pipeline/run        — Trigger pipeline stages
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.graph_store import MemoryGraphStore, get_store
from retrieval.retrieval_engine import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Layer10 Memory Graph API",
    description="Retrieval-grounded long-term memory over GitHub issues",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the UI
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

# Lazy-loaded singletons
_store: Optional[MemoryGraphStore] = None
_retrieval: Optional[RetrievalEngine] = None


def _get_store() -> MemoryGraphStore:
    global _store
    if _store is None:
        _store = get_store()
    return _store


def _get_retrieval() -> RetrievalEngine:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalEngine().load()
    return _retrieval


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    question: str
    max_items: int = 10


class PipelineRequest(BaseModel):
    stages: List[str] = ["transform", "temporal", "conflicts", "dedup", "graph"]


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    store = _get_store()
    return {
        "status": "ok",
        "metrics": store.get_metrics(),
    }


@app.get("/api/metrics")
def metrics():
    store = _get_store()
    return store.get_metrics()


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@app.get("/api/entities")
def list_entities(
    q: str = Query("", description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=500),
):
    store = _get_store()
    if q:
        results = store.search_entities(q, entity_type=entity_type, limit=limit)
    else:
        results = list(store.entities.values())
        if entity_type:
            results = [e for e in results if e.get("entity_type") == entity_type]
        results = results[:limit]
    return {"entities": results, "total": len(results)}


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str):
    store = _get_store()
    entity = store.get_entity(entity_id)
    if not entity:
        raise HTTPException(404, "Entity not found")
    claims = store.get_claims_for_entity(entity_id)
    return {"entity": entity, "claims": claims}


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

@app.get("/api/claims")
def list_claims(
    q: str = Query("", description="Search query"),
    claim_type: Optional[str] = Query(None),
    status: str = Query("active"),
    limit: int = Query(50, ge=1, le=500),
):
    store = _get_store()
    if q:
        results = store.search_claims(q, claim_type=claim_type, status=status, limit=limit)
    else:
        results = list(store.claims.values())
        if claim_type:
            results = [c for c in results if c.get("claim_type") == claim_type]
        if status:
            results = [c for c in results if c.get("status", "active") == status]
        results = results[:limit]
    return {"claims": results, "total": len(results)}


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: str):
    store = _get_store()
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(404, "Claim not found")
    return {"claim": claim}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/api/events")
def list_events(
    issue_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    store = _get_store()
    events = store.events
    if issue_id:
        events = [e for e in events if e.get("subject") == issue_id]
    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]
    total = len(events)
    return {"events": events[offset:offset + limit], "total": total}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@app.get("/api/graph")
def get_graph():
    store = _get_store()
    return store.graph


@app.get("/api/graph/neighbors/{node_id}")
def get_neighbors(node_id: str, depth: int = Query(1, ge=1, le=3)):
    store = _get_store()
    return store.get_neighbors(node_id, max_depth=depth)


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------

@app.get("/api/conflicts")
def get_conflicts(severity: Optional[str] = Query(None)):
    store = _get_store()
    conflicts = store.get_conflicts(severity)
    return {
        "conflicts": conflicts,
        "total": len(conflicts),
        "summary": store.conflicts.get("summary", {}),
    }


# ---------------------------------------------------------------------------
# Merges
# ---------------------------------------------------------------------------

@app.get("/api/merges")
def get_merges(merge_type: Optional[str] = Query(None)):
    store = _get_store()
    merges = store.get_merge_log(merge_type)
    return {"merges": merges, "total": len(merges)}


@app.post("/api/merges/{merge_id}/reverse")
def reverse_merge(merge_id: str):
    store = _get_store()
    success = store.reverse_merge(merge_id)
    if not success:
        raise HTTPException(404, "Merge not found or already reversed")
    store.save()
    return {"status": "reversed", "merge_id": merge_id}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    engine = _get_retrieval()
    return engine.retrieve(req.question, max_items=req.max_items)


# ---------------------------------------------------------------------------
# Bi-temporal state query
# ---------------------------------------------------------------------------

@app.get("/api/state/{issue_id}")
def get_state(issue_id: str, as_of: Optional[str] = Query(None)):
    store = _get_store()
    if as_of:
        return store.get_state_at(issue_id, as_of)
    # Return full history
    states = store.state_history.get("states", {}).get(issue_id, [])
    assignments = store.state_history.get("assignments", {}).get(issue_id, [])
    labels = store.state_history.get("labels", {}).get(issue_id, [])
    return {
        "issue_id": issue_id,
        "states": states,
        "assignments": assignments,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Pipeline trigger
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run")
def run_pipeline(req: PipelineRequest):
    """Trigger pipeline stages (non-blocking for demo)."""
    results = {}
    for stage in req.stages:
        try:
            if stage == "transform":
                from transformation.github_to_events import process_issues
                results[stage] = process_issues()
            elif stage == "temporal":
                from graph.temporal_model import build_state_history
                results[stage] = build_state_history()
            elif stage == "conflicts":
                from graph.conflict_detection import run_conflict_detection
                results[stage] = run_conflict_detection()
            elif stage == "dedup":
                from dedup.identity_resolution import run_dedup
                run_dedup()
                results[stage] = "completed"
            elif stage == "graph":
                from graph.build_graph import run_graph_build
                results[stage] = run_graph_build()
            else:
                results[stage] = "unknown stage"
        except Exception as exc:
            results[stage] = f"error: {exc}"

    # Reload store
    global _store, _retrieval
    _store = None
    _retrieval = None

    return {"results": results}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Layer10 Memory Graph</h1><p>UI not found. Place index.html in ui/</p>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
