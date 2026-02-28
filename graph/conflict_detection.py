import json
import os

INPUT_FILE = "data/processed/issue_state_history.json"
OUTPUT_FILE = "data/processed/state_conflicts.json"


def detect_conflicts():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        issue_states = json.load(f)

    conflicts = {}

    for issue_id, history in issue_states.items():
        states = [entry["state"] for entry in history]

        reopen_count = 0
        redundant_transitions = 0

        for i in range(1, len(states)):
            if states[i-1] == "closed" and states[i] == "open":
                reopen_count += 1

            if states[i-1] == states[i]:
                redundant_transitions += 1

        if reopen_count > 0 or redundant_transitions > 0:
            conflicts[issue_id] = {
                "reopen_count": reopen_count,
                "redundant_transitions": redundant_transitions,
                "history": history
            }

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(conflicts, f, indent=2)

    print(f"Detected {len(conflicts)} issues with state irregularities")


if __name__ == "__main__":
    detect_conflicts()