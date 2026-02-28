import json
import os
from pyvis.network import Network

INPUT_FILE = "data/processed/entities.json"
OUTPUT_HTML = "data/processed/issue_graph.html"


def run_graph_build():
    if not os.path.exists(INPUT_FILE):
        print("No entities file found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entities = json.load(f)

    net = Network(height="750px", width="100%", directed=True)

    added_nodes = set()

    for entity in entities:
        issue = entity.get("issue_id")
        reference = entity.get("reference")

        if not issue or not reference:
            continue

        ref_clean = reference.replace("#", "issue_")

        if issue not in added_nodes:
            net.add_node(issue, label=issue)
            added_nodes.add(issue)

        if ref_clean not in added_nodes:
            net.add_node(ref_clean, label=ref_clean)
            added_nodes.add(ref_clean)

        net.add_edge(issue, ref_clean, label=entity.get("claim_type", "rel"))

    os.makedirs("data/processed", exist_ok=True)
    net.write_html(OUTPUT_HTML, open_browser=False)

    print(f"Graph saved to {OUTPUT_HTML}")


if __name__ == "__main__":
    run_graph_build()