"""Script to batch process text2sql tasks with user authentication."""

import json
from pathlib import Path
from typing import Any, Dict, List

import requests

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "final_dataset.json"
COMMON_KNOWLEDGE_PATH = BASE_DIR / "data" / "common_knowledge.md"
OUTPUT_PATH = BASE_DIR / "upload" / "dataset_exe_result.json"
API_URL = "http://localhost:6080/workflows/run"
TOKEN_URL = "http://localhost:6080/auth/token"

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "client"
WORKFLOW_NAME = "reflection"
NAMESPACE = "game"

_ACCESS_TOKEN: str | None = None


def authenticate() -> str:
    """Retrieve OAuth2 access token via client credentials flow."""
    global _ACCESS_TOKEN
    if _ACCESS_TOKEN:
        return _ACCESS_TOKEN

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    response.raise_for_status()
    token_payload = response.json()
    _ACCESS_TOKEN = token_payload.get("access_token")
    return _ACCESS_TOKEN


def build_prompt(entry: Dict[str, Any]) -> str:
    """Build prompt engineering input from dataset entry."""
    question = entry.get("question", "")
    tables = entry.get("table_list", [])
    knowledge = entry.get("knowledge", "")
    common_knowledge = COMMON_KNOWLEDGE_PATH.read_text(encoding="utf-8")

    db_info = "SQL数据库: StarRocks 3.5.7 (MySQL方言)"

    parts = [
        question.strip(),
        f"表名: {tables}",
        f"业务知识: {knowledge}",
        f"数据库信息: {db_info}",
        f"通用知识库: {common_knowledge}",
    ]
    return "\n\n".join(parts)


def run_text2sql_task(prompt: str) -> Dict[str, Any]:
    """Send text2sql task to workflow service."""
    headers = {"Authorization": f"Bearer {authenticate()}"}
    payload = {
        "workflow": WORKFLOW_NAME,
        "namespace": NAMESPACE,
        "task": prompt,
        "mode": "sync",
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def batch_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process dataset entries in batch."""
    results = []
    for entry in dataset:
        sql_id = entry.get("sql_id", "<unknown>")
        print(f"Processing {sql_id}...")

        prompt = build_prompt(entry)
        result = run_text2sql_task(prompt)

        sql_text = result.get("sql")
        query_result = result.get("result")

        print(f"  SQL: {sql_text}")
        print(f"  Result: {query_result}\n")

        results.append({
            "sql_id": sql_id,
            "sql": sql_text,
            "result": query_result,
        })

    return results


def export_results(results: List[Dict[str, Any]]) -> None:
    """Export task results to JSON file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=4)
    print(f"Results exported to {OUTPUT_PATH}")


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    results = batch_process(dataset)
    export_results(results)


if __name__ == "__main__":
    main()
