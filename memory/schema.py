"""
Layer10 Memory System — Ontology & Schema Definitions
======================================================
Pydantic models for the full extraction → memory pipeline.

Entity types:   Person, Issue, Project, Component, Label, PullRequest
Claim types:    StatusChange, Assignment, Reference, Decision, LabelAction,
                DependsOn, Duplicates, Performance, Generic
Evidence:       Source pointers with excerpt, offsets, timestamps
Temporal:       Bi-temporal model (event_time + validity interval)
Versioning:     Extraction metadata (model, prompt hash, schema version)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON = "Person"
    ISSUE = "Issue"
    PULL_REQUEST = "PullRequest"
    PROJECT = "Project"
    COMPONENT = "Component"
    LABEL = "Label"
    MILESTONE = "Milestone"
    TEAM = "Team"


class ClaimType(str, Enum):
    STATUS_CHANGE = "StatusChange"
    ASSIGNMENT = "Assignment"
    REFERENCE = "Reference"
    DECISION = "Decision"
    LABEL_ACTION = "LabelAction"
    DEPENDS_ON = "DependsOn"
    DUPLICATES = "Duplicates"
    BLOCKS = "Blocks"
    FIXES = "Fixes"
    PERFORMANCE = "Performance"
    OWNERSHIP_CHANGE = "OwnershipChange"
    GENERIC = "Generic"


class ClaimStatus(str, Enum):
    ACTIVE = "active"          # Currently believed true
    SUPERSEDED = "superseded"  # Replaced by a newer claim
    RETRACTED = "retracted"    # Explicitly reversed
    HISTORICAL = "historical"  # Was true in a past interval


class MergeStatus(str, Enum):
    ACTIVE = "active"
    REVERSED = "reversed"


# ---------------------------------------------------------------------------
# Evidence — the grounding anchor
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """A pointer to the exact source material supporting a claim."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_id: str                    # e.g. "comment_12345", "issue_789"
    source_type: str = "github_comment"  # github_issue, github_comment, github_event
    source_url: Optional[str] = None
    excerpt: str                      # The exact text span
    full_text: Optional[str] = None   # The complete source text (for context)
    char_offset_start: Optional[int] = None
    char_offset_end: Optional[int] = None
    timestamp: Optional[str] = None   # ISO-8601
    author: Optional[str] = None
    repo: Optional[str] = None

    def content_hash(self) -> str:
        return hashlib.sha256(
            f"{self.source_id}:{self.excerpt}".encode()
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extraction Metadata — versioning for reproducibility
# ---------------------------------------------------------------------------

class ExtractionMeta(BaseModel):
    """Tracks how a claim was produced — model, prompt, schema version."""
    extraction_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    model_name: str = "llama3"
    model_version: Optional[str] = None
    prompt_hash: Optional[str] = None      # SHA-256 of the prompt template
    schema_version: str = "1.0.0"
    extracted_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_flags: List[str] = Field(default_factory=list)
    # e.g. ["low_confidence", "partial_parse", "retry_success"]


# ---------------------------------------------------------------------------
# Temporal interval — bi-temporal support
# ---------------------------------------------------------------------------

class TemporalInterval(BaseModel):
    """Bi-temporal: event_time is when it happened; valid_from/valid_until
    bracket when the system considers the claim current."""
    event_time: Optional[str] = None        # When the real-world event occurred
    valid_from: Optional[str] = None        # When the claim became current
    valid_until: Optional[str] = None       # When superseded / retracted (None = still current)
    recorded_at: Optional[str] = Field(     # When we ingested it
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A canonical entity in the memory graph."""
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    entity_type: EntityType
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    source_ids: List[str] = Field(default_factory=list)

    def merge_key(self) -> str:
        """Deterministic key for dedup."""
        return f"{self.entity_type.value}::{self.canonical_name.lower().strip()}"


# ---------------------------------------------------------------------------
# Claim — the core knowledge unit
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    """A typed, grounded, temporal assertion linking entities."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    claim_type: ClaimType
    subject_entity_id: str             # Source entity
    object_entity_id: Optional[str] = None  # Target entity (if binary relation)
    predicate: str                     # Human-readable, e.g. "assigned_to", "depends_on"
    value: Optional[str] = None        # Optional value (e.g. label name, status)
    status: ClaimStatus = ClaimStatus.ACTIVE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    temporal: TemporalInterval = Field(default_factory=TemporalInterval)
    extraction_meta: Optional[ExtractionMeta] = None
    superseded_by: Optional[str] = None  # claim_id of the replacement
    tags: List[str] = Field(default_factory=list)

    def content_hash(self) -> str:
        """Hash for dedup — ignores evidence and meta."""
        raw = (
            f"{self.claim_type.value}::{self.subject_entity_id}::"
            f"{self.object_entity_id}::{self.predicate}::{self.value}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Merge Record — for auditable, reversible merges
# ---------------------------------------------------------------------------

class MergeRecord(BaseModel):
    """Audit trail for entity or claim merges."""
    merge_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    merge_type: str  # "entity" or "claim"
    source_ids: List[str]          # IDs that were merged
    target_id: str                 # The surviving canonical ID
    reason: str                    # Why the merge happened
    similarity_score: Optional[float] = None
    merged_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    merged_by: str = "system"      # "system" or user identifier
    status: MergeStatus = MergeStatus.ACTIVE
    snapshot_before: Optional[Dict[str, Any]] = None  # Pre-merge state for rollback


# ---------------------------------------------------------------------------
# Context Pack — retrieval output
# ---------------------------------------------------------------------------

class ContextItem(BaseModel):
    """A single ranked item in a retrieval response."""
    rank: int
    claim: Optional[Claim] = None
    entity: Optional[Entity] = None
    evidence_snippets: List[Evidence] = Field(default_factory=list)
    relevance_score: float = 0.0
    grounding_status: str = "grounded"  # "grounded", "partial", "ungrounded"


class ContextPack(BaseModel):
    """The full retrieval response for a question."""
    question: str
    retrieved_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    items: List[ContextItem] = Field(default_factory=list)
    entities_mentioned: List[Entity] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    total_evidence_count: int = 0


# ---------------------------------------------------------------------------
# Quality Gate — configurable thresholds
# ---------------------------------------------------------------------------

class QualityGateConfig(BaseModel):
    """Thresholds for accepting extracted claims into durable memory."""
    min_confidence: float = 0.3
    require_evidence: bool = True
    min_evidence_count: int = 1
    max_retries: int = 2
    similarity_threshold: float = 0.85    # For dedup
    entity_merge_threshold: float = 0.90  # For entity canonicalization
    decay_rate: float = 0.01              # Confidence decay per day for unconfirmed claims
    human_review_threshold: float = 0.4   # Below this, flag for review

    def passes(self, claim: Claim) -> bool:
        """Check if a claim passes quality gates."""
        if claim.confidence < self.min_confidence:
            return False
        if self.require_evidence and len(claim.evidence) < self.min_evidence_count:
            return False
        return True

    def needs_review(self, claim: Claim) -> bool:
        return claim.confidence < self.human_review_threshold
