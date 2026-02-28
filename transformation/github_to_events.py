import json
import os

INPUT_FILE = "data/raw/fastapi_issues.json"
OUTPUT_FILE = "data/processed/events.json"


def create_event(event_type, subject, obj, timestamp, source_id, evidence):
    return {
        "event_type": event_type,
        "subject": subject,
        "object": obj,
        "timestamp": timestamp,
        "source_id": source_id,
        "evidence": evidence
    }


def process_issues():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        issues = json.load(f)

    events = []

    for issue in issues:
        issue_id = f"issue_{issue['number']}"

        # Issue created
        events.append(
            create_event(
                "IssueCreated",
                issue_id,
                issue["user"]["login"],
                issue["created_at"],
                issue_id,
                issue["title"]
            )
        )

        # Comments
        for comment in issue.get("comments_data", []):
            events.append(
                create_event(
                    "CommentAdded",
                    issue_id,
                    comment["user"]["login"],
                    comment["created_at"],
                    f"comment_{comment['id']}",
                    comment["body"][:200]
                )
            )

        # Timeline Events
        for timeline_event in issue.get("timeline_events", []):
            event_type = timeline_event.get("event")
            timestamp = timeline_event.get("created_at")
            source_id = f"timeline_{timeline_event.get('id')}"

            if event_type == "closed":
                events.append(
                    create_event(
                        "IssueClosed",
                        issue_id,
                        "closed",
                        timestamp,
                        source_id,
                        "Issue closed"
                    )
                )

            elif event_type == "reopened":
                events.append(
                    create_event(
                        "IssueReopened",
                        issue_id,
                        "open",
                        timestamp,
                        source_id,
                        "Issue reopened"
                    )
                )

            elif event_type == "assigned":
                assignee = timeline_event.get("assignee")
                if assignee:
                    events.append(
                        create_event(
                            "AssignedTo",
                            issue_id,
                            assignee.get("login"),
                            timestamp,
                            source_id,
                            f"Assigned to {assignee.get('login')}"
                        )
                    )

            elif event_type == "labeled":
                label = timeline_event.get("label")
                if label:
                    events.append(
                        create_event(
                            "LabelAdded",
                            issue_id,
                            label.get("name"),
                            timestamp,
                            source_id,
                            f"Label added: {label.get('name')}"
                        )
                    )

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Saved {len(events)} events to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_issues()