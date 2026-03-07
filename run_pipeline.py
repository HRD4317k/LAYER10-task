"""
Pipeline Runner — Layer10 Memory Pipeline
===========================================
Orchestrates the full pipeline from raw data to visualization.

Usage:
    python run_pipeline.py                    # Run all stages
    python run_pipeline.py transform dedup    # Run specific stages
    python run_pipeline.py --skip-extraction  # Skip LLM extraction
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


STAGES = {
    "ingest": {
        "label": "1. GitHub Ingestion",
        "run": lambda: __import__("ingestion.github_ingest", fromlist=["run_ingestion"]).run_ingestion(),
    },
    "transform": {
        "label": "2. Event Transformation",
        "run": lambda: __import__("transformation.github_to_events", fromlist=["process_issues"]).process_issues(),
    },
    "temporal": {
        "label": "3. Temporal Modeling",
        "run": lambda: __import__("graph.temporal_model", fromlist=["build_state_history"]).build_state_history(),
    },
    "extract": {
        "label": "4a. Claim Extraction (LLM)",
        "run": lambda: __import__("extraction.claim_extractor", fromlist=["run_extraction"]).run_extraction(),
    },
    "rule_extract": {
        "label": "4b. Rule-Based Claim Generation",
        "run": lambda: __import__("extraction.rule_claims", fromlist=["generate_claims_from_events"]).generate_claims_from_events(),
    },
    "dedup": {
        "label": "5. Deduplication & Canonicalization",
        "run": lambda: __import__("dedup.identity_resolution", fromlist=["run_dedup"]).run_dedup(),
    },
    "conflicts": {
        "label": "6. Conflict Detection",
        "run": lambda: __import__("graph.conflict_detection", fromlist=["run_conflict_detection"]).run_conflict_detection(),
    },
    "graph": {
        "label": "7. Graph Construction",
        "run": lambda: __import__("graph.build_graph", fromlist=["run_graph_build"]).run_graph_build(),
    },
    "retrieve": {
        "label": "8. Example Context Packs",
        "run": lambda: __import__("retrieval.retrieval_engine", fromlist=["generate_example_packs"]).generate_example_packs(),
    },
}

DEFAULT_ORDER = ["transform", "temporal", "extract", "dedup", "conflicts", "graph", "retrieve"]
NO_LLM_ORDER = ["transform", "temporal", "rule_extract", "dedup", "conflicts", "graph", "retrieve"]


def run_stage(name: str) -> bool:
    stage = STAGES.get(name)
    if not stage:
        log.error("Unknown stage: %s", name)
        return False

    log.info("=" * 60)
    log.info("Starting: %s", stage["label"])
    log.info("=" * 60)

    t0 = time.time()
    try:
        result = stage["run"]()
        elapsed = time.time() - t0
        log.info("✓ %s completed in %.1f s → %s", stage["label"], elapsed, result or "done")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        log.error("✗ %s failed after %.1f s: %s", stage["label"], elapsed, exc, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Layer10 Pipeline Runner")
    parser.add_argument(
        "stages", nargs="*",
        help=f"Stages to run (default: all). Options: {', '.join(STAGES.keys())}",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip the LLM extraction stage (useful if Ollama is not running)",
    )
    parser.add_argument(
        "--include-ingest", action="store_true",
        help="Include the GitHub ingestion stage (fetches from API)",
    )
    args = parser.parse_args()

    if args.stages:
        stage_list = args.stages
    elif args.skip_extraction:
        stage_list = NO_LLM_ORDER
    else:
        stage_list = DEFAULT_ORDER

    if args.include_ingest:
        stage_list = ["ingest"] + stage_list

    log.info("Pipeline stages: %s", " → ".join(stage_list))

    results = {}
    for stage_name in stage_list:
        success = run_stage(stage_name)
        results[stage_name] = "✓" if success else "✗"
        if not success:
            log.warning("Stage %s failed — continuing with remaining stages", stage_name)

    log.info("")
    log.info("=" * 60)
    log.info("Pipeline Summary")
    log.info("=" * 60)
    for name, status in results.items():
        log.info("  %s %s", status, STAGES[name]["label"])


if __name__ == "__main__":
    main()
