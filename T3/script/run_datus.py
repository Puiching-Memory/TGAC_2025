"""Utility script to run datus tasks generated from the final dataset.
The script is intentionally simple: adjust the constants below to match
local paths or namespaces before execution.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import requests

# Path configuration (update as needed)
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "final_dataset.json"
API_URL = "http://localhost:6080/workflows/run"
TOKEN_URL = "http://localhost:6080/auth/token"
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "client"
WORKFLOW_NAME = "reflection"
NAMESPACE = "game"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "upload" / "dataset_exe_result.json"

_ACCESS_TOKEN: str | None = None


def load_dataset(path: Path) -> Iterable[Dict[str, Any]]:
    """Load dataset entries from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_task_payload(entry: Dict[str, Any]) -> str:
    """Create a structured payload string for the workflow service."""

    sql_id = entry.get("sql_id")
    question = entry.get("question","")
    tables = entry.get("table_list", [])
    knowledge = entry.get("knowledge", "")

    return f"{question} table_names: {tables} knowledge: {knowledge}"


def get_access_token() -> str:
    """Retrieve an OAuth2 access token via the client credentials flow."""
    global _ACCESS_TOKEN
    if _ACCESS_TOKEN:
        return _ACCESS_TOKEN

    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "DATUS_CLIENT_ID or DATUS_CLIENT_SECRET is not set. Provide OAuth2 client "
            "credentials via environment variables and rerun the script."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Failed to obtain access token. Check CLIENT_ID/CLIENT_SECRET and server auth_clients.yml."
            f" Response: {exc.response.text}"
        ) from exc
    token_payload = response.json()
    access_token = token_payload.get("access_token")
    if not access_token:
        raise RuntimeError("Token endpoint response missing 'access_token'.")
    _ACCESS_TOKEN = access_token
    return access_token


def run_task(task_payload: str) -> Dict[str, Any]:
    """Send the task to the workflow service."""
    headers = {"Authorization": f"Bearer {get_access_token()}"}
    payload = {
        "workflow": WORKFLOW_NAME,
        "namespace": NAMESPACE,
        "task": task_payload,
        "mode": "sync",
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    execution_results: list[Dict[str, Any]] = []
    for entry in dataset:
        sql_id = entry.get("sql_id", "<unknown>")
        task_payload = build_task_payload(entry=entry)
        print(f"Running task for {sql_id}...")
        try:
            result = run_task(task_payload)
        except requests.HTTPError as exc:
            print(f"request failed for {sql_id}: {exc.response.text}")
            raise
        except requests.RequestException as exc:
            print(f"request error for {sql_id}: {exc}")
            raise
        sql_text = result.get("sql")
        query_result = result.get("result")
        print(sql_text)
        print(query_result)
        execution_results.append(
            {
                "sql_id": sql_id,
                "sql": sql_text,
                "result": query_result,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(execution_results, handle, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
