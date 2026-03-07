"""
Retrieval & Grounding Engine — Layer10 Memory Pipeline
=======================================================
Given a natural-language question, produce a ContextPack of ranked,
grounded evidence snippets linked to entities and claims.

Search strategy:
  1. Keyword extraction from question
  2. Entity resolution (names → canonical entities)
  3. Claim search (type, subject, predicate)
  4. Evidence expansion (follow evidence pointers)
  5. Ranking by relevance + confidence + recency
  6. Conflict surfacing (show both sides)
  7. Citation formatting

Handles ambiguity by returning multiple interpretations when
entity resolution is uncertain.
"""

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = "data/processed"

# Try to import sentence-transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    log.warning("sentence-transformers not available; falling back to keyword-only retrieval")


class RetrievalEngine:
    """
    Retrieval engine that maps questions to grounded context packs.
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.entities: List[Dict] = []
        self.claims: List[Dict] = []
        self.events: List[Dict] = []
        self.state_history: Dict = {}
        self.conflicts: Dict = {}
        self.alias_map: Dict[str, str] = {}
        self.model = None
        self._entity_embeddings = None
        self._claim_embeddings = None
        self._loaded = False

    def load(self) -> "RetrievalEngine":
        """Load all data and build indices."""
        path = lambda f: os.path.join(self.data_dir, f)

        if os.path.exists(path("entities.json")):
            with open(path("entities.json"), "r", encoding="utf-8") as f:
                self.entities = json.load(f)

        if os.path.exists(path("canonical_claims.json")):
            with open(path("canonical_claims.json"), "r", encoding="utf-8") as f:
                self.claims = json.load(f)

        # Fallback to claims.json if canonical doesn't exist
        if not self.claims and os.path.exists(path("claims.json")):
            with open(path("claims.json"), "r", encoding="utf-8") as f:
                self.claims = json.load(f)

        if os.path.exists(path("events.json")):
            with open(path("events.json"), "r", encoding="utf-8") as f:
                self.events = json.load(f)

        if os.path.exists(path("issue_state_history.json")):
            with open(path("issue_state_history.json"), "r", encoding="utf-8") as f:
                self.state_history = json.load(f)

        if os.path.exists(path("state_conflicts.json")):
            with open(path("state_conflicts.json"), "r", encoding="utf-8") as f:
                self.conflicts = json.load(f)

        if os.path.exists(path("alias_map.json")):
            with open(path("alias_map.json"), "r", encoding="utf-8") as f:
                self.alias_map = json.load(f)

        # Build embeddings if available
        if HAS_EMBEDDINGS and (self.entities or self.claims):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._build_embeddings()

        self._loaded = True
        log.info("Retrieval engine loaded: %d entities, %d claims, %d events",
                 len(self.entities), len(self.claims), len(self.events))
        return self

    def _build_embeddings(self):
        """Pre-compute embeddings for entities and claims."""
        if not self.model:
            return

        if self.entities:
            entity_texts = [
                f"{e.get('entity_type', '')} {e.get('canonical_name', '')} {' '.join(e.get('aliases', []))}"
                for e in self.entities
            ]
            self._entity_embeddings = self.model.encode(entity_texts)

        if self.claims:
            claim_texts = [
                f"{c.get('claim_type', '')} {c.get('subject', '')} {c.get('predicate', '')} {c.get('object', '')} {c.get('value', '')}"
                for c in self.claims
            ]
            self._claim_embeddings = self.model.encode(claim_texts)

    # ------------------------------------------------------------------
    # Question parsing
    # ------------------------------------------------------------------

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract meaningful keywords from a question."""
        # Remove stop words and question words
        stop_words = {
            "what", "who", "where", "when", "how", "why", "which", "is", "are",
            "was", "were", "the", "a", "an", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "and", "or", "but", "not", "do", "does",
            "did", "has", "have", "had", "be", "been", "being", "this", "that",
            "these", "those", "it", "its", "they", "them", "their", "can", "could",
            "would", "should", "will", "shall", "may", "might", "about", "all",
        }
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        return keywords

    def _extract_issue_refs(self, question: str) -> List[str]:
        """Extract issue references like #1234 or issue_1234."""
        refs = re.findall(r'#(\d+)', question)
        refs += re.findall(r'issue[_\s]?(\d+)', question.lower())
        return [f"issue_{r}" for r in refs]

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_entities(self, question: str, keywords: List[str]) -> List[Dict]:
        """Find entities matching the question."""
        matched: List[Tuple[Dict, float]] = []

        # Direct name matching
        for ent in self.entities:
            name = ent.get("canonical_name", "").lower()
            aliases = [a.lower() for a in ent.get("aliases", [])]
            all_names = [name] + aliases

            for kw in keywords:
                if any(kw in n for n in all_names):
                    matched.append((ent, 0.8))
                    break

        # Issue reference matching
        issue_refs = self._extract_issue_refs(question)
        for ref in issue_refs:
            for ent in self.entities:
                if ent.get("canonical_name", "").lower() == ref.lower():
                    matched.append((ent, 1.0))

        # Semantic matching
        if self.model and self._entity_embeddings is not None:
            q_emb = self.model.encode([question])
            sims = cosine_similarity(q_emb, self._entity_embeddings)[0]
            top_k = sims.argsort()[-5:][::-1]
            for idx in top_k:
                if sims[idx] > 0.3:
                    ent = self.entities[idx]
                    # Avoid duplicates
                    if not any(m[0].get("entity_id") == ent.get("entity_id") for m in matched):
                        matched.append((ent, float(sims[idx])))

        # Sort by score and deduplicate
        matched.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        results = []
        for ent, score in matched:
            eid = ent.get("entity_id", "")
            if eid not in seen:
                ent_copy = dict(ent)
                ent_copy["_relevance_score"] = round(score, 3)
                results.append(ent_copy)
                seen.add(eid)

        return results[:10]

    # ------------------------------------------------------------------
    # Claim search
    # ------------------------------------------------------------------

    def _search_claims(self, question: str, entity_ids: Set[str], keywords: List[str]) -> List[Dict]:
        """Find claims relevant to the question."""
        matched: List[Tuple[Dict, float]] = []

        # Claims involving matched entities
        for claim in self.claims:
            subj = (claim.get("subject") or "").lower()
            obj = (claim.get("object") or "").lower()

            score = 0.0
            # Entity match
            for eid in entity_ids:
                ent = next((e for e in self.entities if e.get("entity_id") == eid), None)
                if ent:
                    names = [ent.get("canonical_name", "").lower()] + [a.lower() for a in ent.get("aliases", [])]
                    if subj in names or obj in names or eid in (subj, obj):
                        score += 0.7

            # Keyword match in claim text
            claim_text = f"{claim.get('predicate', '')} {claim.get('value', '')}".lower()
            evidence_text = " ".join(e.get("excerpt", "") for e in claim.get("evidence", [])).lower()
            for kw in keywords:
                if kw in claim_text:
                    score += 0.3
                if kw in evidence_text:
                    score += 0.2

            # Confidence boost
            score += claim.get("confidence", 0) * 0.2

            if score > 0.2:
                matched.append((claim, score))

        # Semantic search
        if self.model and self._claim_embeddings is not None:
            q_emb = self.model.encode([question])
            sims = cosine_similarity(q_emb, self._claim_embeddings)[0]
            top_k = sims.argsort()[-10:][::-1]
            for idx in top_k:
                if sims[idx] > 0.3:
                    claim = self.claims[idx]
                    cid = claim.get("claim_id", "")
                    existing = next((m for m in matched if m[0].get("claim_id") == cid), None)
                    if existing:
                        # Boost existing score
                        matched = [
                            (c, s + float(sims[idx]) * 0.5) if c.get("claim_id") == cid else (c, s)
                            for c, s in matched
                        ]
                    else:
                        matched.append((claim, float(sims[idx]) * 0.8))

        matched.sort(key=lambda x: x[1], reverse=True)
        results = []
        seen = set()
        for claim, score in matched[:15]:
            cid = claim.get("claim_id", "")
            if cid not in seen:
                claim_copy = dict(claim)
                claim_copy["_relevance_score"] = round(score, 3)
                results.append(claim_copy)
                seen.add(cid)

        return results

    # ------------------------------------------------------------------
    # Evidence expansion
    # ------------------------------------------------------------------

    def _expand_evidence(self, claims: List[Dict], entity_ids: Set[str]) -> List[Dict]:
        """Gather supporting evidence from events."""
        evidence: List[Dict] = []
        seen = set()

        # From claims
        for claim in claims:
            for ev in claim.get("evidence", []):
                key = ev.get("source_id", "") + ev.get("excerpt", "")[:50]
                if key not in seen:
                    evidence.append({
                        "source_id": ev.get("source_id", ""),
                        "excerpt": ev.get("excerpt", ""),
                        "issue_id": ev.get("issue_id", claim.get("subject", "")),
                        "from_claim": claim.get("claim_id", ""),
                        "confidence": claim.get("confidence", 0.5),
                    })
                    seen.add(key)

        # Relevant events for matched entities
        entity_names = set()
        for eid in entity_ids:
            ent = next((e for e in self.entities if e.get("entity_id") == eid), None)
            if ent:
                entity_names.add(ent.get("canonical_name", "").lower())

        for evt in self.events:
            subj = evt.get("subject", "").lower()
            if subj in entity_names or any(subj == n for n in entity_names):
                key = evt.get("source_id", "")
                if key not in seen:
                    evidence.append({
                        "source_id": evt.get("source_id", ""),
                        "excerpt": evt.get("evidence", ""),
                        "evidence_full": evt.get("evidence_full"),
                        "issue_id": evt.get("subject", ""),
                        "event_type": evt.get("event_type", ""),
                        "timestamp": evt.get("timestamp", ""),
                        "source_url": evt.get("source_url"),
                        "actor": evt.get("actor", ""),
                    })
                    seen.add(key)
                    if len(evidence) > 50:
                        break  # Pruning limit

        return evidence[:30]

    # ------------------------------------------------------------------
    # Conflict detection for question
    # ------------------------------------------------------------------

    def _find_conflicts(self, entity_ids: Set[str]) -> List[Dict]:
        """Find conflicts related to matched entities."""
        result = []
        all_conflicts = self.conflicts.get("conflicts", [])
        entity_names = set()
        for eid in entity_ids:
            ent = next((e for e in self.entities if e.get("entity_id") == eid), None)
            if ent:
                entity_names.add(ent.get("canonical_name", "").lower())

        for conflict in all_conflicts:
            issue = conflict.get("issue_id", "").lower()
            subject = conflict.get("subject", "").lower()
            if issue in entity_names or subject in entity_names:
                result.append(conflict)

        return result[:5]

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, max_items: int = 10) -> Dict[str, Any]:
        """
        Main entry point: question → ContextPack.

        Returns a dict with:
          - question
          - items: ranked list of {rank, claim, entity, evidence, score, grounding}
          - entities_mentioned
          - conflicts
          - total_evidence_count
          - citations
        """
        if not self._loaded:
            self.load()

        keywords = self._extract_keywords(question)
        log.info("Query: %s | Keywords: %s", question, keywords)

        # Step 1: Entity resolution
        matched_entities = self._resolve_entities(question, keywords)
        entity_ids = {e.get("entity_id", "") for e in matched_entities}

        # Step 2: Claim search
        matched_claims = self._search_claims(question, entity_ids, keywords)

        # Step 3: Evidence expansion
        evidence = self._expand_evidence(matched_claims, entity_ids)

        # Step 4: Conflict surfacing
        conflicts = self._find_conflicts(entity_ids)

        # Step 5: Build ranked items
        items = []
        for i, claim in enumerate(matched_claims[:max_items]):
            claim_evidence = [
                e for e in evidence if e.get("from_claim") == claim.get("claim_id")
            ]
            # Also include event evidence for the claim's subject
            related_events = [
                e for e in evidence
                if e.get("issue_id", "").lower() == (claim.get("subject") or "").lower()
                and "event_type" in e
            ][:3]

            all_ev = claim_evidence + related_events
            grounding = "grounded" if all_ev else "ungrounded"

            items.append({
                "rank": i + 1,
                "claim": claim,
                "evidence_snippets": all_ev,
                "relevance_score": claim.get("_relevance_score", 0),
                "grounding_status": grounding,
            })

        # Step 6: Format citations
        citations = []
        for item in items:
            for ev in item.get("evidence_snippets", []):
                citations.append({
                    "source_id": ev.get("source_id", ""),
                    "excerpt": (ev.get("excerpt") or "")[:200],
                    "source_url": ev.get("source_url", ""),
                    "issue_id": ev.get("issue_id", ""),
                })

        context_pack = {
            "question": question,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
            "items": items,
            "entities_mentioned": matched_entities[:5],
            "conflicts": conflicts,
            "total_evidence_count": len(evidence),
            "citations": citations[:20],
        }

        return context_pack


# ---------------------------------------------------------------------------
# CLI for generating example context packs
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What issues has tiangolo been involved with?",
    "Which issues are currently open?",
    "What labels are most commonly used?",
    "Are there any issues with state conflicts or reopens?",
    "What decisions were made about performance?",
]


def generate_example_packs():
    """Generate context packs for example questions and save them."""
    engine = RetrievalEngine().load()

    packs: List[Dict] = []

    for q in EXAMPLE_QUESTIONS:
        log.info("Retrieving: %s", q)
        pack = engine.retrieve(q)
        packs.append(pack)

    output_file = os.path.join(DATA_DIR, "example_context_packs.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(packs, f, indent=2)

    log.info("Saved %d example context packs → %s", len(packs), output_file)
    return output_file


if __name__ == "__main__":
    generate_example_packs()
