"""
Conflict Detection — Layer10 Memory Pipeline
==============================================
Detects six categories of conflicts from the event stream,
state history, claims, and raw metadata:

  1. State consistency — event stream says "open" but snapshot says "closed"
     (i.e. missing close events for issues that are actually closed)
  2. State irregularities — reopens, redundant transitions, rapid flapping
  3. Rapid closures  — issues closed within 10 minutes of creation
  4. Duplicate labels — same label added to an issue more than once
  5. Claim conflicts — contradictory claims about the same subject
  6. Ownership changes — reassignment patterns / multiple assignees

Each conflict carries:
  - conflict_type (enum)
  - severity  (high / medium / low)
  - issue_id or subject
  - description (human-readable)
  - evidence  (supporting data)

Outputs a structured conflict report at data/processed/state_conflicts.json.
"""

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

INPUT_STATE_FILE = "data/processed/issue_state_history.json"
INPUT_CLAIMS_FILE = "data/processed/canonical_claims.json"
INPUT_EVENTS_FILE = "data/processed/events.json"
OUTPUT_FILE = "data/processed/state_conflicts.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _ts(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating trailing Z."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


# -----------------------------------------------------------------------
# 1. State consistency:  snapshot says closed but event stream lacks close
# -----------------------------------------------------------------------

def detect_state_consistency(events: List[Dict]) -> List[Dict]:
    """
    Compare the IssueCreated extra.state against the presence of
    an IssueClosed event.  If an issue's snapshot state is "closed"
    but there is no IssueClosed event at all, that means the close
    happened outside the observed event window — a data gap.
    """
    conflicts: List[Dict] = []

    created_meta: Dict[str, Dict] = {}
    closed_ids: set = set()

    for evt in events:
        if evt["event_type"] == "IssueCreated":
            created_meta[evt["subject"]] = evt
        if evt["event_type"] == "IssueClosed":
            closed_ids.add(evt["subject"])

    for issue_id, create_evt in created_meta.items():
        extra = create_evt.get("extra", {})
        snapshot_state = extra.get("state", "open")
        is_pr = extra.get("is_pull_request", False)

        if snapshot_state == "closed" and issue_id not in closed_ids:
            conflicts.append({
                "conflict_type": "state_consistency_gap",
                "severity": "medium",
                "issue_id": issue_id,
                "description": (
                    f"{'PR' if is_pr else 'Issue'} {issue_id} snapshot state "
                    f"is 'closed' but NO IssueClosed event exists in the event "
                    f"stream — the close likely happened outside the ingested "
                    f"window."
                ),
                "expected_state": "closed",
                "event_stream_state": "open (no close event)",
                "created_at": create_evt.get("timestamp"),
                "source_url": create_evt.get("source_url", ""),
                "evidence": [
                    {
                        "excerpt": (
                            f"Created at {create_evt.get('timestamp')} with "
                            f"snapshot state=closed, but no IssueClosed event."
                        ),
                        "source_id": create_evt.get("source_id", ""),
                    }
                ],
            })

    return conflicts


# -----------------------------------------------------------------------
# 2. State irregularities:  reopen cycles, redundant, rapid flap
# -----------------------------------------------------------------------

def detect_state_irregularities(history_data: Dict) -> List[Dict]:
    """Detect reopen cycles, redundant transitions, rapid flapping."""
    conflicts: List[Dict] = []
    states = history_data.get("states", {})

    for issue_id, entries in states.items():
        if len(entries) < 2:
            continue

        reopen_count = 0
        redundant_transitions = 0
        rapid_flaps = 0
        prev = entries[0]

        for i in range(1, len(entries)):
            curr = entries[i]
            prev_state = prev.get("state", "")
            curr_state = curr.get("state", "")

            # Reopen
            if prev_state == "closed" and curr_state == "open":
                reopen_count += 1

            # Redundant (same state twice in a row)
            if prev_state == curr_state:
                redundant_transitions += 1

            # Rapid flapping (< 60 seconds)
            try:
                t1 = _ts(prev.get("event_time", ""))
                t2 = _ts(curr.get("event_time", ""))
                if abs((t2 - t1).total_seconds()) < 60 and prev_state != curr_state:
                    rapid_flaps += 1
            except (ValueError, TypeError):
                pass

            prev = curr

        if reopen_count > 0 or redundant_transitions > 0 or rapid_flaps > 0:
            parts = []
            if reopen_count:
                parts.append(f"{reopen_count} reopen(s)")
            if redundant_transitions:
                parts.append(f"{redundant_transitions} redundant transition(s)")
            if rapid_flaps:
                parts.append(f"{rapid_flaps} rapid flap(s)")

            conflicts.append({
                "conflict_type": "state_irregularity",
                "severity": "high" if reopen_count > 1 or rapid_flaps > 0 else "medium",
                "issue_id": issue_id,
                "reopen_count": reopen_count,
                "redundant_transitions": redundant_transitions,
                "rapid_flaps": rapid_flaps,
                "description": f"Issue {issue_id} has: {', '.join(parts)}",
                "evidence": [
                    {
                        "excerpt": " → ".join(
                            f"{e['state']}@{e.get('event_time','?')}"
                            for e in entries
                        ),
                        "source_id": entries[0].get("source_id", ""),
                    }
                ],
            })

    return conflicts


# -----------------------------------------------------------------------
# 3. Rapid closures — issues closed within 10 minutes of creation
# -----------------------------------------------------------------------

def detect_rapid_closures(events: List[Dict]) -> List[Dict]:
    """Issues closed within 10 minutes of creation."""
    conflicts: List[Dict] = []

    create_map: Dict[str, Dict] = {}
    close_map: Dict[str, Dict] = {}

    for evt in events:
        if evt["event_type"] == "IssueCreated":
            create_map[evt["subject"]] = evt
        elif evt["event_type"] == "IssueClosed":
            close_map[evt["subject"]] = evt

    for issue_id, close_evt in close_map.items():
        create_evt = create_map.get(issue_id)
        if not create_evt:
            continue
        try:
            t_create = _ts(create_evt["timestamp"])
            t_close = _ts(close_evt["timestamp"])
            delta_min = (t_close - t_create).total_seconds() / 60.0
            if 0 < delta_min <= 10:
                conflicts.append({
                    "conflict_type": "rapid_closure",
                    "severity": "low" if delta_min > 5 else "medium",
                    "issue_id": issue_id,
                    "description": (
                        f"Issue {issue_id} was closed only "
                        f"{delta_min:.1f} min after creation."
                    ),
                    "minutes_to_close": round(delta_min, 1),
                    "created_at": create_evt["timestamp"],
                    "closed_at": close_evt["timestamp"],
                    "source_url": create_evt.get("source_url", ""),
                    "evidence": [
                        {
                            "excerpt": (
                                f"Created {create_evt['timestamp']}, "
                                f"closed {close_evt['timestamp']} "
                                f"({delta_min:.1f} min)"
                            ),
                            "source_id": close_evt.get("source_id", ""),
                        }
                    ],
                })
        except (ValueError, TypeError, KeyError):
            pass

    return conflicts


# -----------------------------------------------------------------------
# 4. Duplicate label events — same label added to the same issue twice
# -----------------------------------------------------------------------

def detect_duplicate_labels(events: List[Dict]) -> List[Dict]:
    """Detect labels applied more than once to the same issue."""
    conflicts: List[Dict] = []

    by_issue_label: Dict[str, List[Dict]] = defaultdict(list)
    for evt in events:
        if evt["event_type"] == "LabelAdded":
            label = evt.get("extra", {}).get("label", "")
            key = f"{evt['subject']}::{label}"
            by_issue_label[key].append(evt)

    for key, evts in by_issue_label.items():
        if len(evts) < 2:
            continue
        issue_id, label = key.split("::", 1)
        conflicts.append({
            "conflict_type": "duplicate_label",
            "severity": "low",
            "issue_id": issue_id,
            "label": label,
            "count": len(evts),
            "description": (
                f"Label '{label}' was added to {issue_id} "
                f"{len(evts)} times."
            ),
            "timestamps": [e.get("timestamp") for e in evts],
            "evidence": [
                {
                    "excerpt": f"LabelAdded '{label}' at {e.get('timestamp')}",
                    "source_id": e.get("source_id", ""),
                }
                for e in evts
            ],
        })

    return conflicts


# -----------------------------------------------------------------------
# 5. Claim conflicts — contradictory or superseding claims
# -----------------------------------------------------------------------

def detect_claim_conflicts(claims: List[Dict]) -> List[Dict]:
    """Detect contradictory claims about the same subject+predicate."""
    conflicts: List[Dict] = []

    by_key: Dict[str, List[Dict]] = defaultdict(list)
    for claim in claims:
        key = f"{claim.get('subject', '')}::{claim.get('predicate', '')}"
        by_key[key].append(claim)

    for key, group in by_key.items():
        if len(group) < 2:
            continue

        values = set()
        for c in group:
            v = c.get("value") or c.get("object")
            if v:
                values.add(str(v))

        if len(values) > 1:
            subj = group[0].get("subject", "")
            pred = group[0].get("predicate", "")
            conflicts.append({
                "conflict_type": "contradictory_claims",
                "severity": "high",
                "subject": subj,
                "predicate": pred,
                "conflicting_values": sorted(values),
                "description": (
                    f"Claims about '{subj}' / '{pred}' disagree: "
                    f"{', '.join(sorted(values))}"
                ),
                "evidence": [
                    {
                        "excerpt": (
                            f"{c.get('claim_type')}: "
                            f"{c.get('subject')} {c.get('predicate')} "
                            f"{c.get('value') or c.get('object')}"
                        ),
                        "source_id": c.get("claim_id", ""),
                    }
                    for c in group
                ],
            })

    return conflicts


# -----------------------------------------------------------------------
# 6. Ownership changes — reassignment patterns
# -----------------------------------------------------------------------

def detect_ownership_changes(history_data: Dict) -> List[Dict]:
    """Detect issues with multiple distinct assignees over time."""
    conflicts: List[Dict] = []
    assignments = history_data.get("assignments", {})

    for issue_id, entries in assignments.items():
        if len(entries) < 2:
            continue

        assignees = [e["assignee"] for e in entries]
        unique_assignees = list(dict.fromkeys(assignees))
        if len(unique_assignees) > 1:
            conflicts.append({
                "conflict_type": "ownership_change",
                "severity": "medium" if len(unique_assignees) > 2 else "low",
                "issue_id": issue_id,
                "assignee_history": unique_assignees,
                "change_count": len(entries),
                "description": (
                    f"Issue {issue_id} was reassigned between "
                    f"{' → '.join(unique_assignees)} "
                    f"({len(entries)} events)"
                ),
                "evidence": [
                    {
                        "excerpt": f"Assigned to {e['assignee']} at {e.get('event_time')}",
                        "source_id": e.get("source_id", ""),
                    }
                    for e in entries
                ],
            })

    return conflicts


# -----------------------------------------------------------------------
# 7. Temporal anomalies — out-of-order events per subject
# -----------------------------------------------------------------------

def detect_temporal_anomalies(events: List[Dict]) -> List[Dict]:
    """Detect out-of-order timestamps within a single issue."""
    conflicts: List[Dict] = []

    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for evt in events:
        by_subject[evt.get("subject", "")].append(evt)

    for subject, evts in by_subject.items():
        sorted_evts = sorted(evts, key=lambda e: e.get("timestamp") or "")
        for i in range(1, len(sorted_evts)):
            prev_ts = sorted_evts[i - 1].get("timestamp") or ""
            curr_ts = sorted_evts[i].get("timestamp") or ""
            if prev_ts and curr_ts and curr_ts < prev_ts:
                conflicts.append({
                    "conflict_type": "temporal_anomaly",
                    "severity": "low",
                    "issue_id": subject,
                    "description": (
                        f"Event at {curr_ts} arrived before previous "
                        f"event at {prev_ts}"
                    ),
                    "evidence": [
                        {
                            "excerpt": (
                                f"{sorted_evts[i-1]['event_type']}@{prev_ts} "
                                f"> {sorted_evts[i]['event_type']}@{curr_ts}"
                            ),
                            "source_id": sorted_evts[i].get("source_id", ""),
                        }
                    ],
                })

    return conflicts


# ======================================================================
# Main
# ======================================================================

def run_conflict_detection() -> str:
    """Run all conflict detectors and write a combined report."""
    all_conflicts: List[Dict] = []

    # Load inputs
    history_data: Dict = {}
    if os.path.exists(INPUT_STATE_FILE):
        with open(INPUT_STATE_FILE, "r", encoding="utf-8") as f:
            history_data = json.load(f)

    claims: List[Dict] = []
    if os.path.exists(INPUT_CLAIMS_FILE):
        with open(INPUT_CLAIMS_FILE, "r", encoding="utf-8") as f:
            claims = json.load(f)

    events: List[Dict] = []
    if os.path.exists(INPUT_EVENTS_FILE):
        with open(INPUT_EVENTS_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

    # Run detectors
    d1 = detect_state_consistency(events)
    d2 = detect_state_irregularities(history_data)
    d3 = detect_rapid_closures(events)
    d4 = detect_duplicate_labels(events)
    d5 = detect_claim_conflicts(claims)
    d6 = detect_ownership_changes(history_data)
    d7 = detect_temporal_anomalies(events)

    all_conflicts = d1 + d2 + d3 + d4 + d5 + d6 + d7

    # Build report
    severity_counts = Counter(c.get("severity") for c in all_conflicts)
    type_counts = Counter(c.get("conflict_type") for c in all_conflicts)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_conflicts": len(all_conflicts),
            "high_severity": severity_counts.get("high", 0),
            "medium_severity": severity_counts.get("medium", 0),
            "low_severity": severity_counts.get("low", 0),
            "by_type": dict(type_counts),
        },
        "conflicts": all_conflicts,
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(
        "Conflict detection: %d total — consistency: %d, irregularity: %d, "
        "rapid: %d, dup-label: %d, claims: %d, ownership: %d, temporal: %d",
        len(all_conflicts), len(d1), len(d2), len(d3),
        len(d4), len(d5), len(d6), len(d7),
    )
    return OUTPUT_FILE


if __name__ == "__main__":
    run_conflict_detection()
