"""
GitHub → Canonical Events Transformation — Layer10 Memory Pipeline
===================================================================
Converts raw GitHub issue JSON into a normalized event stream with
proper evidence grounding (full text + offsets) and richer event types.

Event types produced:
  IssueCreated, IssueClosed, IssueReopened, CommentAdded,
  AssignedTo, Unassigned, LabelAdded, LabelRemoved,
  MilestoneSet, Referenced

Every event carries:
  - subject (issue id)
  - actor (who performed the action)
  - timestamp (ISO-8601)
  - source_id (deterministic hash — idempotent)
  - evidence (text excerpt, max 500 chars)
  - evidence_full (complete source text)
  - char_offset_start / char_offset_end
  - source_url (GitHub permalink)
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

INPUT_FILE = "data/raw/fastapi_issues.json"
OUTPUT_FILE = "data/processed/events.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _stable_id(*parts: str) -> str:
    """Deterministic content-addressed ID for idempotent re-runs."""
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def create_event(
    event_type: str,
    subject: str,
    actor: str,
    timestamp: Optional[str],
    source_id: str,
    evidence: str,
    evidence_full: Optional[str] = None,
    source_url: Optional[str] = None,
    char_offset_start: Optional[int] = None,
    char_offset_end: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evt: Dict[str, Any] = {
        "event_type": event_type,
        "subject": subject,
        "actor": actor,
        "timestamp": timestamp,
        "source_id": source_id,
        "evidence": (evidence or "")[:500],
        "evidence_full": evidence_full,
        "source_url": source_url,
        "char_offset_start": char_offset_start,
        "char_offset_end": char_offset_end,
    }
    if extra:
        evt["extra"] = extra
    return evt


def process_issues() -> str:
    """
    Transform raw issues → canonical events.
    Returns output file path.
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        issues = json.load(f)

    events: List[Dict] = []
    seen_ids: set = set()

    for issue in issues:
        number = issue["number"]
        issue_id = f"issue_{number}"
        creator = (issue.get("user") or {}).get("login", "unknown")
        html_url = issue.get("html_url", "")
        title = issue.get("title", "")
        body = issue.get("body") or ""

        # --- IssueCreated ---
        sid = _stable_id("IssueCreated", issue_id)
        if sid not in seen_ids:
            events.append(create_event(
                "IssueCreated", issue_id, creator,
                issue.get("created_at"), sid,
                evidence=title,
                evidence_full=body[:2000] if body else title,
                source_url=html_url,
                extra={
                    "labels": [l["name"] for l in issue.get("labels", [])],
                    "state": issue.get("state"),
                    "is_pull_request": "pull_request" in issue,
                },
            ))
            seen_ids.add(sid)

        # --- Comments ---
        for comment in issue.get("comments_data", []):
            cid = comment.get("id", "")
            sid = _stable_id("CommentAdded", issue_id, str(cid))
            if sid in seen_ids:
                continue
            comment_body = comment.get("body") or ""
            comment_author = (comment.get("user") or {}).get("login", "unknown")
            events.append(create_event(
                "CommentAdded", issue_id, comment_author,
                comment.get("created_at"), sid,
                evidence=comment_body[:500],
                evidence_full=comment_body,
                source_url=comment.get("html_url"),
                char_offset_start=0,
                char_offset_end=min(len(comment_body), 500),
            ))
            seen_ids.add(sid)

        # --- Timeline events ---
        for tev in issue.get("timeline_events", []):
            etype = tev.get("event", "")
            ts = tev.get("created_at")
            tev_id = tev.get("id", "")
            actor = "system"
            if isinstance(tev.get("actor"), dict):
                actor = tev["actor"].get("login", "system")
            elif isinstance(tev.get("user"), dict):
                actor = tev["user"].get("login", "system")

            sid = _stable_id(etype, issue_id, str(tev_id))
            if sid in seen_ids:
                continue

            if etype == "closed":
                events.append(create_event(
                    "IssueClosed", issue_id, actor, ts, sid,
                    evidence="Issue closed", source_url=html_url,
                ))
            elif etype == "reopened":
                events.append(create_event(
                    "IssueReopened", issue_id, actor, ts, sid,
                    evidence="Issue reopened", source_url=html_url,
                ))
            elif etype == "assigned":
                assignee = tev.get("assignee")
                if assignee:
                    aname = assignee.get("login", "unknown")
                    events.append(create_event(
                        "AssignedTo", issue_id, actor, ts, sid,
                        evidence=f"Assigned to {aname}",
                        source_url=html_url,
                        extra={"assignee": aname},
                    ))
            elif etype == "unassigned":
                assignee = tev.get("assignee")
                if assignee:
                    aname = assignee.get("login", "unknown")
                    events.append(create_event(
                        "Unassigned", issue_id, actor, ts, sid,
                        evidence=f"Unassigned {aname}",
                        source_url=html_url,
                        extra={"assignee": aname},
                    ))
            elif etype == "labeled":
                label = tev.get("label")
                if label:
                    lname = label.get("name", "")
                    events.append(create_event(
                        "LabelAdded", issue_id, actor, ts, sid,
                        evidence=f"Label added: {lname}",
                        source_url=html_url,
                        extra={"label": lname},
                    ))
            elif etype == "unlabeled":
                label = tev.get("label")
                if label:
                    lname = label.get("name", "")
                    events.append(create_event(
                        "LabelRemoved", issue_id, actor, ts, sid,
                        evidence=f"Label removed: {lname}",
                        source_url=html_url,
                        extra={"label": lname},
                    ))
            elif etype == "milestoned":
                ms = tev.get("milestone") or {}
                events.append(create_event(
                    "MilestoneSet", issue_id, actor, ts, sid,
                    evidence=f"Milestone set: {ms.get('title', '')}",
                    source_url=html_url,
                    extra={"milestone": ms.get("title")},
                ))
            elif etype in ("referenced", "cross-referenced"):
                events.append(create_event(
                    "Referenced", issue_id, actor, ts, sid,
                    evidence=f"Referenced in event {tev_id}",
                    source_url=html_url,
                ))

            seen_ids.add(sid)

    # ------------------------------------------------------------------
    # Synthesize missing IssueClosed events from the snapshot state.
    # Many issues are fetched with state="closed" but the timeline
    # doesn't contain a "closed" event (e.g. the close happened before
    # the ingested page window).  We generate a synthetic close event
    # so the temporal model correctly marks them closed.
    # ------------------------------------------------------------------
    closed_event_subjects = {
        e["subject"]
        for e in events
        if e["event_type"] == "IssueClosed"
    }

    for issue in issues:
        number = issue["number"]
        issue_id = f"issue_{number}"
        if issue.get("state") == "closed" and issue_id not in closed_event_subjects:
            closed_at = issue.get("closed_at") or issue.get("updated_at") or issue.get("created_at")
            sid = _stable_id("IssueClosed", issue_id, "synthetic")
            if sid not in seen_ids:
                closer = "system"
                if issue.get("closed_by") and isinstance(issue["closed_by"], dict):
                    closer = issue["closed_by"].get("login", "system")
                events.append(create_event(
                    "IssueClosed", issue_id, closer, closed_at, sid,
                    evidence="Issue state is 'closed' (synthesized from snapshot)",
                    source_url=issue.get("html_url", ""),
                    extra={"synthetic": True},
                ))
                seen_ids.add(sid)

    # ------------------------------------------------------------------
    # Synthesize AssignedTo events from snapshot assignees if no
    # timeline 'assigned' event exists for the issue.
    # ------------------------------------------------------------------
    assigned_subjects = {
        e["subject"]
        for e in events
        if e["event_type"] == "AssignedTo"
    }

    for issue in issues:
        number = issue["number"]
        issue_id = f"issue_{number}"
        if issue_id not in assigned_subjects:
            for assignee_data in issue.get("assignees", []):
                aname = assignee_data.get("login", "unknown")
                sid = _stable_id("AssignedTo", issue_id, aname, "synthetic")
                if sid not in seen_ids:
                    events.append(create_event(
                        "AssignedTo", issue_id, "system",
                        issue.get("created_at"), sid,
                        evidence=f"Assigned to {aname} (from snapshot)",
                        source_url=issue.get("html_url", ""),
                        extra={"assignee": aname, "synthetic": True},
                    ))
                    seen_ids.add(sid)

    # Sort chronologically
    events.sort(key=lambda e: e.get("timestamp") or "")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, default=str)

    log.info("Saved %d events → %s", len(events), OUTPUT_FILE)
    return OUTPUT_FILE


if __name__ == "__main__":
    process_issues()
