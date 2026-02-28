GitHub Issue Execution Graph
Overview

This repository implements a deterministic, multi-stage execution pipeline that transforms GitHub issue discussions into a structured relationship graph.

The system ingests issues, reconstructs state transitions, extracts relational claims using a locally hosted LLM, performs semantic canonicalization, and builds a directed issue knowledge graph.

The design emphasizes determinism, observability, and stage isolation over agentic orchestration.

System Architecture

The pipeline consists of explicit execution stages. Each stage consumes structured input artifacts and produces deterministic output artifacts.

1. Ingestion

Pull issues, comments, and timeline events from a GitHub repository.

Output:

data/raw/fastapi_issues.json
2. Event Normalization

Transform GitHub responses into a canonical event schema:

IssueCreated

CommentAdded

IssueStateChanged

Output:

data/processed/events.json
3. Temporal Modeling

Reconstruct issue state history from ordered events.

Output:

data/processed/issue_state_history.json
4. Conflict Detection

Detect state irregularities such as reopen conflicts or invalid transitions.

Output:

data/processed/state_conflicts.json
5. Relational Claim Extraction (LLM)

Extract structured relational claims from issue comments using a locally hosted LLaMA3 model via Ollama.

Key design constraints:

JSON-only output

Signal filtering (comments containing "#")

Post-generation validation

No blind trust in model output

Output:

data/processed/claims.json
6. Semantic Canonicalization

Cluster semantically equivalent claims using sentence-transformer embeddings and cosine similarity.

Output:

data/processed/entities.json
7. Graph Construction

Build a directed issue relationship graph using canonical entities.

Output:

data/processed/issue_graph.html
Execution Flow

Run stages sequentially:

python ingestion/github_ingest.py
python transformation/github_to_events.py
python modeling/temporal_model.py
python modeling/conflict_detection.py
python extraction/claim_extractor.py
python canonicalization/identity_resolution.py
python graph/build_graph.py

Each stage is independently re-runnable and produces persisted artifacts.

Design Philosophy

This system is intentionally non-agentic.

It follows deterministic execution principles:

Explicit state transitions

Stage-level artifact persistence

Controlled LLM usage

Post-LLM structural validation

No hidden memory

Reproducible graph construction

The LLM is treated as a bounded extraction component inside a larger structured workflow.

Technology Stack

Python

Ollama (local LLaMA3 inference)

Sentence-Transformers

scikit-learn

PyVis

Requests

Installation

Create virtual environment:

python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Install Ollama:

https://ollama.com

Pull model:

ollama pull llama3

Verify:

ollama --version
Observability

Each stage writes structured output to disk.
Artifacts can be inspected independently for debugging and auditability.

This enables deterministic replay and transparent execution tracking.

Example Outcome

From ~300 issues:

6 relational claims extracted

3 canonical entities formed

Directed issue graph constructed

Future Extensions

Export graph to Neo4j

REST API wrapper via FastAPI

Execution logging framework

Metrics dashboard

Deterministic workflow runner abstraction