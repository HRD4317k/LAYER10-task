"""
Rule-Based Claim Generator — Layer10 Memory Pipeline
======================================================
Generates structured claims from the event stream WITHOUT an LLM.
This complements the LLM-based claim_extractor.py for environments
where Ollama / llama3 is not available.

Claim types generated:
  - StatusChange:  issue opened / closed / reopened
  - Assignment:    issue assigned to person
  - Reference:     cross-reference between entities
  - Decision:      rapid close or label change as implicit decision
  - OwnershipChange: reassignment patterns
  - Duplicates:    duplicate label application (data quality signal)

Every claim is grounded back to the source event with:
  - evidence:  excerpt + source_id + source_url + char offsets
  - extraction_meta:  model=rule-engine, schema_version, confidence
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INPUT_EVENTS = "data/processed/events.json"
INPUT_STATE_HISTORY = "data/processed/issue_state_history.json"
OUTPUT_FILE = "data/processed/claims.json"

SCHEMA_VERSION = "2.0.0"
NOW = datetime.utcnow().isoformat() + "Z"


def _claim_id(*parts: str) -> str:
    return hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _make_claim(
    claim_type: str,
    subject: str,
    predicate: str,
    obj: Optional[str],
    value: Optional[str],
    confidence: float,
    evidence_excerpt: str,
    source_id: str = "",
    source_url: str = "",
    timestamp: str = "",
) -> Dict[str, Any]:
    cid = _claim_id(claim_type, subject, predicate, obj or "", value or "", source_id)
    return {
        "claim_id": cid,
        "claim_type": claim_type,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "value": value,
        "confidence": confidence,
        "status": "active",
        "evidence": [
            {
                "excerpt": evidence_excerpt[:500],
                "source_id": source_id,
                "source_url": source_url,
                "timestamp": timestamp,
                "char_offset_start": 0,
                "char_offset_end": min(len(evidence_excerpt), 500),
            }
        ],
        "source_ids": [source_id],
        "support_count": 1,
        "content_hash": hashlib.sha256(
            f"{claim_type}::{subject}::{predicate}::{obj}::{value}".encode()
        ).hexdigest()[:16],
        "extraction_meta": {
            "model": "rule-engine",
            "schema_version": SCHEMA_VERSION,
            "prompt_hash": "rule-based-v1",
            "confidence": confidence,
            "extracted_at": NOW,
        },
        "created_at": timestamp or NOW,
    }


def generate_claims_from_events() -> str:
    """
    Generate structured claims from the event stream.
    Returns output file path.
    """
    with open(INPUT_EVENTS, "r", encoding="utf-8") as f:
        events = json.load(f)

    state_history = {}
    if os.path.exists(INPUT_STATE_HISTORY):
        with open(INPUT_STATE_HISTORY, "r", encoding="utf-8") as f:
            state_history = json.load(f)

    claims: List[Dict] = []
    seen: set = set()

    for evt in events:
        etype = evt["event_type"]
        subject = evt["subject"]
        actor = evt.get("actor", "system")
        ts = evt.get("timestamp", "")
        sid = evt.get("source_id", "")
        url = evt.get("source_url", "")
        evidence = evt.get("evidence", "") or ""
        extra = evt.get("extra", {})

        # ---- StatusChange claims ----
        if etype == "IssueCreated":
            c = _make_claim(
                "StatusChange", subject, "created_by", actor,
                value="open",
                confidence=0.95,
                evidence_excerpt=f"Issue {subject} created by {actor}: {evidence}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

            # If it's a PR, add that signal
            if extra.get("is_pull_request"):
                c2 = _make_claim(
                    "Reference", subject, "is_pull_request", "true",
                    value="PR",
                    confidence=0.99,
                    evidence_excerpt=f"{subject} is a pull request",
                    source_id=sid, source_url=url, timestamp=ts,
                )
                if c2["claim_id"] not in seen:
                    claims.append(c2)
                    seen.add(c2["claim_id"])

            # Labels from creation
            for label in extra.get("labels", []):
                c3 = _make_claim(
                    "Reference", subject, "has_label", label,
                    value=None,
                    confidence=0.95,
                    evidence_excerpt=f"{subject} labeled '{label}' at creation",
                    source_id=sid, source_url=url, timestamp=ts,
                )
                if c3["claim_id"] not in seen:
                    claims.append(c3)
                    seen.add(c3["claim_id"])

        elif etype == "IssueClosed":
            synthetic = extra.get("synthetic", False)
            c = _make_claim(
                "StatusChange", subject, "closed_by", actor,
                value="closed",
                confidence=0.85 if synthetic else 0.95,
                evidence_excerpt=f"Issue {subject} closed by {actor}" + (" (synthesized from snapshot)" if synthetic else ""),
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        elif etype == "IssueReopened":
            c = _make_claim(
                "StatusChange", subject, "reopened_by", actor,
                value="open",
                confidence=0.95,
                evidence_excerpt=f"Issue {subject} reopened by {actor}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        # ---- Assignment claims ----
        elif etype == "AssignedTo":
            assignee = extra.get("assignee", actor)
            c = _make_claim(
                "Assignment", subject, "assigned_to", assignee,
                value=None,
                confidence=0.90,
                evidence_excerpt=f"{subject} assigned to {assignee}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        elif etype == "Unassigned":
            assignee = extra.get("assignee", "")
            c = _make_claim(
                "OwnershipChange", subject, "unassigned", assignee,
                value=None,
                confidence=0.90,
                evidence_excerpt=f"{subject} unassigned from {assignee}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        # ---- Label claims ----
        elif etype == "LabelAdded":
            label = extra.get("label", "")
            c = _make_claim(
                "Reference", subject, "labeled", label,
                value=None,
                confidence=0.95,
                evidence_excerpt=f"Label '{label}' added to {subject}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        elif etype == "LabelRemoved":
            label = extra.get("label", "")
            c = _make_claim(
                "Reference", subject, "label_removed", label,
                value=None,
                confidence=0.95,
                evidence_excerpt=f"Label '{label}' removed from {subject}",
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

        # ---- Comment claims (evidence-rich) ----
        elif etype == "CommentAdded":
            # The comment itself is evidence for a "discussed" claim
            c = _make_claim(
                "Generic", subject, "discussed_by", actor,
                value=None,
                confidence=0.80,
                evidence_excerpt=evidence[:500],
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

            # Try to detect cross-references in comment text
            full_text = evt.get("evidence_full", evidence) or ""
            import re
            refs = re.findall(r'#(\d+)', full_text)
            for ref in refs:
                ref_id = f"issue_{ref}"
                c2 = _make_claim(
                    "Reference", subject, "references", ref_id,
                    value=None,
                    confidence=0.75,
                    evidence_excerpt=f"Comment by {actor} references #{ref}: {full_text[:200]}",
                    source_id=sid, source_url=url, timestamp=ts,
                )
                if c2["claim_id"] not in seen:
                    claims.append(c2)
                    seen.add(c2["claim_id"])

            # Detect decision keywords
            decision_keywords = ["approved", "merged", "rejected", "wontfix",
                                 "resolved", "duplicate", "confirmed", "accepted"]
            lower_text = full_text.lower()
            for kw in decision_keywords:
                if kw in lower_text:
                    c3 = _make_claim(
                        "Decision", subject, f"decision_{kw}", actor,
                        value=kw,
                        confidence=0.60,
                        evidence_excerpt=f"Comment by {actor} contains '{kw}': {full_text[:300]}",
                        source_id=sid, source_url=url, timestamp=ts,
                    )
                    if c3["claim_id"] not in seen:
                        claims.append(c3)
                        seen.add(c3["claim_id"])

        # ---- Referenced claims ----
        elif etype == "Referenced":
            c = _make_claim(
                "Reference", subject, "externally_referenced", actor,
                value=None,
                confidence=0.70,
                evidence_excerpt=evidence,
                source_id=sid, source_url=url, timestamp=ts,
            )
            if c["claim_id"] not in seen:
                claims.append(c)
                seen.add(c["claim_id"])

    # --- Derived claims from state history ---

    # Rapid closures → Decision claims
    states = state_history.get("states", {})
    for issue_id, entries in states.items():
        if len(entries) >= 2:
            first = entries[0]
            last = entries[-1]
            if first.get("state") == "open" and last.get("state") == "closed":
                try:
                    from datetime import datetime as dt
                    t1 = dt.fromisoformat(first["event_time"].replace("Z", "+00:00"))
                    t2 = dt.fromisoformat(last["event_time"].replace("Z", "+00:00"))
                    delta_min = (t2 - t1).total_seconds() / 60.0
                    if 0 < delta_min <= 10:
                        c = _make_claim(
                            "Decision", issue_id, "rapid_closure", last.get("actor", "system"),
                            value=f"closed_in_{delta_min:.0f}_min",
                            confidence=0.85,
                            evidence_excerpt=f"{issue_id} closed {delta_min:.1f} min after creation",
                            source_id=last.get("source_id", ""),
                            timestamp=last.get("event_time", ""),
                        )
                        if c["claim_id"] not in seen:
                            claims.append(c)
                            seen.add(c["claim_id"])
                except (ValueError, KeyError):
                    pass

    # Sort by timestamp
    claims.sort(key=lambda c: c.get("created_at") or "")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)

    # Summary
    from collections import Counter
    type_counts = Counter(c["claim_type"] for c in claims)
    log.info(
        "Generated %d rule-based claims → %s (types: %s)",
        len(claims), OUTPUT_FILE,
        ", ".join(f"{k}:{v}" for k, v in type_counts.most_common()),
    )
    return OUTPUT_FILE


if __name__ == "__main__":
    generate_claims_from_events()
