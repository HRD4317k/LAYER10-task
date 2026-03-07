"""
Temporal Model — Layer10 Memory Pipeline
==========================================
Bi-temporal state reconstruction for issues.

Two time axes:
  - event_time  : when the real-world event happened
  - valid_from / valid_until : when our system considers the state current

This enables queries like:
  "What was the state of issue X as of date Y?"
  "What do we currently believe the state is?"
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

INPUT_FILE = "data/processed/events.json"
OUTPUT_FILE = "data/processed/issue_state_history.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Event types that affect issue state
STATE_CHANGE_EVENTS = {
    "IssueCreated": "open",
    "IssueClosed": "closed",
    "IssueReopened": "open",
}


def build_state_history() -> str:
    """
    Build bi-temporal state history from events.
    Returns output file path.
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    now = datetime.utcnow().isoformat() + "Z"
    issue_states: Dict[str, List[Dict]] = defaultdict(list)
    issue_assignments: Dict[str, List[Dict]] = defaultdict(list)
    issue_labels: Dict[str, List[Dict]] = defaultdict(list)

    for event in events:
        issue_id = event["subject"]
        timestamp = event.get("timestamp")
        etype = event["event_type"]
        actor = event.get("actor", "system")
        source_id = event.get("source_id", "")

        # --- State transitions ---
        if etype in STATE_CHANGE_EVENTS:
            new_state = STATE_CHANGE_EVENTS[etype]

            # Close previous interval
            history = issue_states[issue_id]
            if history and history[-1]["valid_until"] is None:
                history[-1]["valid_until"] = timestamp

            issue_states[issue_id].append({
                "state": new_state,
                "event_time": timestamp,
                "valid_from": timestamp,
                "valid_until": None,  # open interval = current
                "caused_by": etype,
                "actor": actor,
                "source_id": source_id,
                "recorded_at": now,
            })

        # --- Assignments ---
        elif etype == "AssignedTo":
            extra = event.get("extra", {})
            assignee = extra.get("assignee", event.get("actor", ""))
            issue_assignments[issue_id].append({
                "assignee": assignee,
                "event_time": timestamp,
                "valid_from": timestamp,
                "valid_until": None,
                "source_id": source_id,
                "recorded_at": now,
            })

        elif etype == "Unassigned":
            extra = event.get("extra", {})
            assignee = extra.get("assignee", "")
            # Close the matching assignment interval
            for asg in reversed(issue_assignments[issue_id]):
                if asg["assignee"] == assignee and asg["valid_until"] is None:
                    asg["valid_until"] = timestamp
                    break

        # --- Labels ---
        elif etype == "LabelAdded":
            extra = event.get("extra", {})
            label = extra.get("label", "")
            issue_labels[issue_id].append({
                "label": label,
                "event_time": timestamp,
                "valid_from": timestamp,
                "valid_until": None,
                "source_id": source_id,
                "recorded_at": now,
            })

        elif etype == "LabelRemoved":
            extra = event.get("extra", {})
            label = extra.get("label", "")
            for lb in reversed(issue_labels[issue_id]):
                if lb["label"] == label and lb["valid_until"] is None:
                    lb["valid_until"] = timestamp
                    break

    # Sort each history by event_time
    for issue_id in issue_states:
        issue_states[issue_id].sort(key=lambda x: x.get("event_time") or "")
    for issue_id in issue_assignments:
        issue_assignments[issue_id].sort(key=lambda x: x.get("event_time") or "")
    for issue_id in issue_labels:
        issue_labels[issue_id].sort(key=lambda x: x.get("event_time") or "")

    output = {
        "generated_at": now,
        "total_issues": len(issue_states),
        "states": dict(issue_states),
        "assignments": dict(issue_assignments),
        "labels": dict(issue_labels),
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info("Saved bi-temporal state history for %d issues → %s",
             len(issue_states), OUTPUT_FILE)
    return OUTPUT_FILE


def query_state_at(issue_id: str, as_of: str, history_data: Dict) -> Dict[str, Any]:
    """Query what we believe was the state of an issue at a given time."""
    states = history_data.get("states", {}).get(issue_id, [])
    current_state = "unknown"
    for entry in states:
        vf = entry.get("valid_from") or ""
        vu = entry.get("valid_until")
        if vf <= as_of and (vu is None or vu > as_of):
            current_state = entry["state"]

    assignments = history_data.get("assignments", {}).get(issue_id, [])
    current_assignees = []
    for asg in assignments:
        vf = asg.get("valid_from") or ""
        vu = asg.get("valid_until")
        if vf <= as_of and (vu is None or vu > as_of):
            current_assignees.append(asg["assignee"])

    labels = history_data.get("labels", {}).get(issue_id, [])
    current_labels = []
    for lb in labels:
        vf = lb.get("valid_from") or ""
        vu = lb.get("valid_until")
        if vf <= as_of and (vu is None or vu > as_of):
            current_labels.append(lb["label"])

    return {
        "issue_id": issue_id,
        "as_of": as_of,
        "state": current_state,
        "assignees": current_assignees,
        "labels": current_labels,
    }


if __name__ == "__main__":
    build_state_history()
