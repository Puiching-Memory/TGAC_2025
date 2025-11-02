"""Utility script to run datus tasks generated from the final dataset.
The script is intentionally simple: adjust the constants below to match
local paths or namespaces before execution.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

# Path configuration (update as needed)
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "final_dataset.json"
DATUs_EXECUTABLE = "datus"  # Update if the binary lives elsewhere
NAMESPACE = "your_db"  # Update to the target namespace


def load_dataset(path: Path) -> Iterable[Dict[str, Any]]:
    """Load dataset entries from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_task_payload(entry: Dict[str, Any]) -> str:
    """Create a structured payload string for the datus CLI."""

    sql_id = entry.get("sql_id")
    question = entry.get("question","")
    tables = entry.get("table_list", [])
    knowledge = entry.get("knowledge", "")

    return f"{question} table_names: {tables} knowledge: {knowledge}"


def run_task(task_payload: str) -> None:
    """Invoke the datus CLI with the provided payload."""
    command = [DATUs_EXECUTABLE, "run", "--namespace", NAMESPACE, "--task", task_payload]
    print(f"Running command: {command}")
    # subprocess will stream stdout/stderr to the console, which is helpful for monitoring.
    subprocess.run(command, check=True, text=True)


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    for entry in dataset:
        sql_id = entry.get("sql_id", "<unknown>")
        task_payload = build_task_payload(entry=entry)
        print(f"Running task for {sql_id}...")
        try:
            run_task(task_payload)
        except subprocess.CalledProcessError as exc:
            print(f"datus run failed for {sql_id} with exit code {exc.returncode}")
            raise


if __name__ == "__main__":
    main()
