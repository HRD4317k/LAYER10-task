import json
import os
from collections import defaultdict

INPUT_FILE = "data/processed/events.json"
OUTPUT_FILE = "data/processed/issue_state_history.json"


def build_state_history():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    issue_states = defaultdict(list)

    for event in events:
        issue_id = event["subject"]
        timestamp = event["timestamp"]

        if event["event_type"] == "IssueCreated":
            issue_states[issue_id].append({
                "timestamp": timestamp,
                "state": "open"
            })

        elif event["event_type"] == "IssueClosed":
            issue_states[issue_id].append({
                "timestamp": timestamp,
                "state": "closed"
            })

        elif event["event_type"] == "IssueReopened":
            issue_states[issue_id].append({
                "timestamp": timestamp,
                "state": "open"
            })

    # Sort chronologically
    for issue_id in issue_states:
        issue_states[issue_id] = sorted(
            issue_states[issue_id],
            key=lambda x: x["timestamp"]
        )

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(issue_states, f, indent=2)

    print(f"Saved state history for {len(issue_states)} issues")


if __name__ == "__main__":
    build_state_history()