import csv
import json
import os
import requests
import re
from pathlib import Path
from pprint import pprint
from typing import Dict, List
from colorama import Fore, Style

SERVER_URL = "http://localhost:8000/api/vanna/v2/chat_poll"
AUTHORIZATION_HEADER = {"Authorization": "admin@example.com"}
CKPT_DIR = Path("T3/ckpt/V6.1_33.72_1109")
SCORE_PATH = CKPT_DIR / "score.csv"
CKPT_RESULT_PATH = CKPT_DIR / "dataset_exe_result.json"
UPLOAD_RESULT_PATH = Path("T3/upload/dataset_exe_result.json")
PROMPT_DIR = Path("T3/prompt/input/V1")


def natural_sort_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def load_score_map(path: Path) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    if not path.exists():
        print(f"{Fore.YELLOW}Score file missing at {path}. All tasks will hit the server.{Style.RESET_ALL}")
        return scores

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sql_id = (row.get("SQL ID") or "").strip()
            if not sql_id:
                continue
            try:
                scores[sql_id] = int(row.get("得分", 0))
            except (TypeError, ValueError):
                scores[sql_id] = 0
    return scores


def load_ckpt_results(path: Path) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    if not path.exists():
        print(f"{Fore.YELLOW}CKPT result file missing at {path}.{Style.RESET_ALL}")
        return results

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"{Fore.RED}Failed to parse CKPT results: {exc}{Style.RESET_ALL}")
        return results

    if not isinstance(payload, list):
        print(f"{Fore.RED}Unexpected CKPT payload format; expected list.{Style.RESET_ALL}")
        return results

    for item in payload:
        if not isinstance(item, dict):
            continue
        sql_id = item.get("sql_id")
        if isinstance(sql_id, str) and sql_id:
            results[sql_id] = item
    return results


def load_upload_dataset(path: Path) -> List[dict]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return []

    return payload if isinstance(payload, list) else []


def write_upload_dataset(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def persist_cached_entry(entry: dict) -> None:
    entries = load_upload_dataset(UPLOAD_RESULT_PATH)
    updated = False
    for idx, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("sql_id") == entry.get("sql_id"):
            entries[idx] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)
    write_upload_dataset(UPLOAD_RESULT_PATH, entries)


def main():
    score_map = load_score_map(SCORE_PATH)
    ckpt_results = load_ckpt_results(CKPT_RESULT_PATH)

    task_names = os.listdir(path=str(PROMPT_DIR))
    task_names.sort(key=natural_sort_key)

    for task_name in task_names:
        print(f"{Fore.GREEN}Processing task: {task_name}{Style.RESET_ALL}")

        sql_id = Path(task_name).stem
        prompt_path = PROMPT_DIR / task_name
        with prompt_path.open("r", encoding="utf-8") as handle:
            task_prompt = handle.read()

        score = score_map.get(sql_id, 0)
        cached_entry = ckpt_results.get(sql_id)

        if score == 1:
            if isinstance(cached_entry, dict):
                print(f"{Fore.CYAN}Using cached answer for {sql_id}.{Style.RESET_ALL}")
                persist_cached_entry(cached_entry)
                cached_output = dict(cached_entry)
                cached_output["cached"] = True
                pprint(cached_output)
                print("\n" + "=" * 50 + "\n")
                continue
            print(f"{Fore.YELLOW}No cached payload for {sql_id}; falling back to server.{Style.RESET_ALL}")

        response = requests.post(
            SERVER_URL,
            json={
                "message": task_prompt,
                "conversation_id": task_name,
            },
            headers=AUTHORIZATION_HEADER,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            response_payload = {
                "error": str(exc),
                "status_code": response.status_code,
                "text": response.text,
            }
        else:
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = {
                    "error": "Invalid JSON response",
                    "status_code": response.status_code,
                    "text": response.text,
                }

        pprint(response_payload)
        print("\n" + "=" * 50 + "\n")

        # break


if __name__ == "__main__":
    main()