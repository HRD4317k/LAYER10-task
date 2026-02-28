import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "data/processed/claims.json"
OUTPUT_FILE = "data/processed/entities.json"

SIMILARITY_THRESHOLD = 0.85


def run_dedup():
    if not os.path.exists(INPUT_FILE):
        print("No claims file found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        claims = json.load(f)

    if not claims:
        print("No claims to process.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert claims to text for embedding
    claim_texts = [
        f"{c.get('claim_type', '')} {c.get('reference', '')}"
        for c in claims
    ]

    embeddings = model.encode(claim_texts)

    similarity_matrix = cosine_similarity(embeddings)

    visited = set()
    clusters = []

    for i in range(len(claims)):
        if i in visited:
            continue

        cluster = [claims[i]]
        visited.add(i)

        for j in range(i + 1, len(claims)):
            if j in visited:
                continue

            if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                cluster.append(claims[j])
                visited.add(j)

        clusters.append(cluster)

    # Flatten clusters to representative entities
    entities = []

    for cluster in clusters:
        representative = cluster[0].copy()
        representative["supporting_claims"] = cluster
        entities.append(representative)

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)

    print(f"Created {len(entities)} canonical entities from {len(claims)} claims")


if __name__ == "__main__":
    run_dedup()