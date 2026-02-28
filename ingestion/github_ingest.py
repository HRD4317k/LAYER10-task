import requests
import os
import json
import time

OWNER = "fastapi"
REPO = "fastapi"
BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"

HEADERS = {
    "Accept": "application/vnd.github+json"
}

MAX_PAGES = 3  # 3 pages × 100 issues = 300 issues


def fetch_issues():
    issues = []
    page = 1

    while page <= MAX_PAGES:
        print(f"Fetching issues page {page}...")
        url = f"{BASE_URL}/issues?state=all&per_page=100&page={page}"

        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print("Error:", response.status_code)
            break

        data = response.json()

        if not data:
            break

        issues.extend(data)
        page += 1
        time.sleep(1)

    return issues


def fetch_comments(issue_number):
    comments = []
    page = 1

    while True:
        url = f"{BASE_URL}/issues/{issue_number}/comments?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            break

        data = response.json()

        if not data:
            break

        comments.extend(data)
        page += 1
        time.sleep(0.5)

    return comments


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    issues = fetch_issues()

    print(f"Downloaded {len(issues)} issues")

    enriched_issues = []

    for issue in issues:
        issue_number = issue["number"]
        print(f"Fetching comments for issue #{issue_number}")
        comments = fetch_comments(issue_number)

        issue["comments_data"] = comments
        enriched_issues.append(issue)

    with open("data/raw/fastapi_issues.json", "w", encoding="utf-8") as f:
        json.dump(enriched_issues, f, indent=2)

    print("Saved to data/raw/fastapi_issues.json")