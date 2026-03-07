"""
GitHub Issue Ingestion — Layer10 Memory Pipeline
=================================================
Fetches issues, comments, and timeline events from a GitHub repository.

Features:
  - GitHub token support (avoids rate limits)
  - Incremental ingestion via checkpoint file
  - Idempotent re-runs (deduplicates by issue number)
  - Configurable repo / page limits
  - Retry with exponential backoff
  - Structured logging
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OWNER = os.environ.get("GITHUB_OWNER", "fastapi")
REPO = os.environ.get("GITHUB_REPO", "fastapi")
BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

HEADERS: Dict[str, str] = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

MAX_PAGES = int(os.environ.get("MAX_PAGES", "3"))  # 3 pages × 100 = 300 issues
OUTPUT_FILE = "data/raw/fastapi_issues.json"
CHECKPOINT_FILE = "data/raw/.ingest_checkpoint.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3, backoff: float = 2.0) -> Optional[Any]:
    """GET with retry + exponential backoff + rate-limit handling."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset:
                    wait = max(int(reset) - int(time.time()), 1)
                    log.warning("Rate-limited. Sleeping %d s …", wait)
                    time.sleep(wait)
                    continue
            if resp.status_code >= 500:
                log.warning("Server error %d (attempt %d/%d)", resp.status_code, attempt, retries)
                time.sleep(backoff * attempt)
                continue
            log.error("HTTP %d for %s", resp.status_code, url)
            return None
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(backoff * attempt)
    return None


def _load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_checkpoint(data: Dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_issues() -> List[Dict]:
    """Paginate through issues (state=all, sorted by updated desc)."""
    issues: List[Dict] = []
    page = 1
    while page <= MAX_PAGES:
        log.info("Fetching issues page %d / %d …", page, MAX_PAGES)
        url = (
            f"{BASE_URL}/issues?state=all&per_page=100&page={page}"
            f"&sort=updated&direction=desc"
        )
        data = _get(url)
        if not data:
            break
        issues.extend(data)
        page += 1
        time.sleep(0.5)
    return issues


def fetch_comments(issue_number: int) -> List[Dict]:
    """Fetch all comments for a single issue."""
    comments: List[Dict] = []
    page = 1
    while True:
        url = f"{BASE_URL}/issues/{issue_number}/comments?per_page=100&page={page}"
        data = _get(url)
        if not data:
            break
        comments.extend(data)
        page += 1
        time.sleep(0.3)
    return comments


def fetch_issue_events(issue_number: int) -> List[Dict]:
    """Fetch timeline events for a single issue."""
    events: List[Dict] = []
    page = 1
    while True:
        url = f"{BASE_URL}/issues/{issue_number}/events?per_page=100&page={page}"
        data = _get(url)
        if not data:
            break
        events.extend(data)
        page += 1
        time.sleep(0.3)
    return events


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion() -> str:
    """
    Run the full ingestion pipeline.
    Returns the path to the output file.
    """
    os.makedirs("data/raw", exist_ok=True)

    checkpoint = _load_checkpoint()
    already_fetched: set = set(checkpoint.get("fetched_issues", []))

    # Load existing data for incremental merge
    existing: List[Dict] = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing_map = {iss["number"]: iss for iss in existing}

    issues = fetch_issues()
    log.info("Downloaded %d issue stubs", len(issues))

    enriched = dict(existing_map)  # start from existing
    new_count = 0

    for issue in issues:
        num = issue["number"]
        if num in already_fetched and num in enriched:
            continue  # idempotent skip

        log.info("Enriching issue #%d …", num)
        issue["comments_data"] = fetch_comments(num)
        issue["timeline_events"] = fetch_issue_events(num)
        issue["_ingested_at"] = datetime.utcnow().isoformat() + "Z"
        enriched[num] = issue
        already_fetched.add(num)
        new_count += 1

        # Periodic checkpoint every 20 issues
        if new_count % 20 == 0:
            _save_checkpoint({"fetched_issues": sorted(already_fetched)})

    result = list(enriched.values())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    _save_checkpoint({
        "fetched_issues": sorted(already_fetched),
        "last_run": datetime.utcnow().isoformat() + "Z",
        "total_issues": len(result),
    })

    log.info("Saved %d issues (%d new) → %s", len(result), new_count, OUTPUT_FILE)
    return OUTPUT_FILE


if __name__ == "__main__":
    run_ingestion()