import json
import os
import requests
import re

INPUT_FILE = "data/processed/events.json"
OUTPUT_FILE = "data/processed/claims.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def extract_claim(comment_text, issue_id, source_id):
    """
    Calls Ollama and extracts structured claims from a comment.
    Returns a list of claim dicts.
    """

    prompt = f"""
You are an information extraction system.

Extract factual relationship claims from the following GitHub comment.

Only extract claims that reference other issues or PRs (e.g., #1234).

Return JSON only.
If no claims exist, return an empty JSON array [].

Comment:
\"\"\"
{comment_text}
\"\"\"
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
    )

    raw_output = response.json()["response"]

    # Debug print (can comment later)
    print("----- RAW MODEL OUTPUT -----")
    print(raw_output)
    print("----------------------------")

    # Try extracting JSON array safely
    try:
        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not json_match:
            return []

        claims = json.loads(json_match.group())

        # Attach metadata
        for claim in claims:
            claim["issue_id"] = issue_id
            claim["source_id"] = source_id

        return claims

    except Exception as e:
        print("JSON parsing error:", e)
        return []


def run_extraction():
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    all_claims = []

    for event in events:

        if event["event_type"] != "CommentAdded":
            continue

        comment_text = event.get("evidence", "")

        # Signal filter: only process comments with references
        if "#" not in comment_text:
            continue

        issue_id = event["subject"]
        source_id = event["source_id"]

        print(f"Processing comment {source_id}")

        claims = extract_claim(comment_text, issue_id, source_id)

        all_claims.extend(claims)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, indent=2)

    print(f"\nSaved {len(all_claims)} claims to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_extraction()