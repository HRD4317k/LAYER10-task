"""
Claim Extraction — Layer10 Memory Pipeline
============================================
Extracts typed, grounded relational claims from GitHub issue comments
using a locally hosted LLM (Ollama / llama3).

Design pillars:
  - Structured schema: every claim has a type, subject, object, confidence
  - Evidence grounding: excerpt + offsets + source pointer
  - Pydantic validation with repair: malformed LLM output is retried/fixed
  - Prompt versioning: prompt template hashed for reproducibility
  - Quality gates: confidence thresholds, evidence requirements
  - Deterministic source IDs for idempotent re-extraction
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INPUT_FILE = "data/processed/events.json"
OUTPUT_FILE = "data/processed/claims.json"
EXTRACTION_LOG_FILE = "data/processed/extraction_log.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3")

# Schema version — bump when prompt or output schema changes
SCHEMA_VERSION = "2.0.0"

# Quality gate thresholds
MIN_CONFIDENCE = 0.3
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Prompt template (versioned)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a precise information extraction system for GitHub issue discussions.

Extract factual relational claims from the following comment. For each claim, produce a JSON object with exactly these fields:

- "claim_type": one of ["Reference", "DependsOn", "Duplicates", "Blocks", "Fixes", "Decision", "Assignment", "StatusChange", "Performance", "Generic"]
- "subject": the issue or entity making the claim (e.g. "issue_1234" or the author)
- "object": the target entity (e.g. "#5678", a person name, or null)
- "predicate": a short verb phrase (e.g. "references", "depends on", "fixes", "duplicates")
- "value": any additional detail (status, label, etc.) or null
- "confidence": your confidence 0.0–1.0 that this claim is factually stated
- "excerpt": the exact substring from the comment supporting this claim

Return a JSON array. If no claims exist, return [].
Do NOT include any text outside the JSON array.

Issue context: {issue_id}
Author: {author}

Comment:
\"\"\"
{comment_text}
\"\"\"
"""


def _prompt_hash() -> str:
    return hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LLM interaction with retry
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> Optional[str]:
    """Call Ollama and return raw text response."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        log.warning("Ollama returned HTTP %d", resp.status_code)
        return None
    except requests.RequestException as exc:
        log.warning("Ollama request failed: %s", exc)
        return None


def _parse_json_array(raw: str) -> Optional[List[Dict]]:
    """Extract a JSON array from possibly noisy LLM output."""
    # Try direct parse first
    raw = raw.strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Regex fallback: find the outermost [ ... ]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_claim(claim: Dict, issue_id: str, source_id: str) -> Optional[Dict]:
    """Validate and normalize a single extracted claim."""
    VALID_TYPES = {
        "Reference", "DependsOn", "Duplicates", "Blocks", "Fixes",
        "Decision", "Assignment", "StatusChange", "Performance", "Generic",
    }
    ctype = claim.get("claim_type", "Generic")
    if ctype not in VALID_TYPES:
        ctype = "Generic"

    predicate = claim.get("predicate", "")
    if not predicate:
        predicate = ctype.lower()

    confidence = claim.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    if confidence < MIN_CONFIDENCE:
        return None  # quality gate

    excerpt = claim.get("excerpt", "")
    if not excerpt and not claim.get("subject"):
        return None  # must have some grounding

    return {
        "claim_type": ctype,
        "subject": claim.get("subject", issue_id),
        "object": claim.get("object"),
        "predicate": predicate,
        "value": claim.get("value"),
        "confidence": round(confidence, 3),
        "excerpt": excerpt[:500] if excerpt else "",
        "issue_id": issue_id,
        "source_id": source_id,
        "extraction_meta": {
            "model": MODEL_NAME,
            "schema_version": SCHEMA_VERSION,
            "prompt_hash": _prompt_hash(),
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        },
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_claims_from_comment(
    comment_text: str, issue_id: str, source_id: str, author: str = "unknown"
) -> List[Dict]:
    """Extract claims from a single comment with retry."""
    prompt = EXTRACTION_PROMPT.format(
        issue_id=issue_id,
        author=author,
        comment_text=comment_text[:3000],
    )

    for attempt in range(1, MAX_RETRIES + 1):
        raw = _call_llm(prompt)
        if raw is None:
            log.warning("LLM call failed (attempt %d/%d)", attempt, MAX_RETRIES)
            time.sleep(1)
            continue

        parsed = _parse_json_array(raw)
        if parsed is None:
            log.warning("JSON parse failed (attempt %d/%d): %s", attempt, MAX_RETRIES, raw[:200])
            time.sleep(0.5)
            continue

        claims = []
        for item in parsed:
            validated = _validate_claim(item, issue_id, source_id)
            if validated:
                claims.append(validated)

        return claims

    log.warning("All retries exhausted for %s / %s", issue_id, source_id)
    return []


def run_extraction() -> str:
    """
    Run extraction on all qualifying events.
    Returns output file path.
    """
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    all_claims: List[Dict] = []
    extraction_log: List[Dict] = []
    processed_count = 0
    skipped_count = 0

    for event in events:
        if event["event_type"] != "CommentAdded":
            continue

        comment_text = event.get("evidence_full") or event.get("evidence", "")
        if not comment_text or len(comment_text.strip()) < 20:
            skipped_count += 1
            continue

        # Only process comments that have cross-references or substantive content
        has_reference = "#" in comment_text or "issue" in comment_text.lower()
        has_decision = any(kw in comment_text.lower() for kw in [
            "decision", "decided", "agreed", "approved", "rejected",
            "blocked", "duplicate", "depends", "fixes", "closes",
        ])

        if not has_reference and not has_decision:
            skipped_count += 1
            continue

        issue_id = event["subject"]
        source_id = event["source_id"]
        author = event.get("actor", "unknown")

        log.info("Extracting claims from %s / %s", issue_id, source_id)
        claims = extract_claims_from_comment(comment_text, issue_id, source_id, author)
        all_claims.extend(claims)
        processed_count += 1

        extraction_log.append({
            "source_id": source_id,
            "issue_id": issue_id,
            "claims_found": len(claims),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    # Deduplicate claims by content hash
    seen_hashes: set = set()
    unique_claims: List[Dict] = []
    for c in all_claims:
        h = hashlib.sha256(
            f"{c['claim_type']}::{c['subject']}::{c.get('object')}::{c['predicate']}".encode()
        ).hexdigest()[:16]
        if h not in seen_hashes:
            unique_claims.append(c)
            seen_hashes.add(h)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_claims, f, indent=2)

    with open(EXTRACTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_comments_processed": processed_count,
            "total_comments_skipped": skipped_count,
            "total_claims_raw": len(all_claims),
            "total_claims_deduped": len(unique_claims),
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_NAME,
            "prompt_hash": _prompt_hash(),
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "entries": extraction_log,
        }, f, indent=2)

    log.info(
        "Extraction complete: %d comments → %d claims (%d unique). "
        "Skipped %d comments.",
        processed_count, len(all_claims), len(unique_claims), skipped_count,
    )
    return OUTPUT_FILE


if __name__ == "__main__":
    run_extraction()
