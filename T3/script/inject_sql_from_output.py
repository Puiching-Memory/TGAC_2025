from __future__ import annotations

import json
import os
from typing import Dict, List


def load_output_sqls(output_path: str) -> Dict[str, str]:
    with open(output_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    answer = raw.get("Answer", [])
    if isinstance(answer, str):
        try:
            parsed_answer = json.loads(answer)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unable to parse Answer string as JSON: {exc}") from exc
    elif isinstance(answer, list):
        parsed_answer = answer
    else:
        raise ValueError("Unsupported Answer payload type")

    sql_map: Dict[str, str] = {}
    for item in parsed_answer:
        results_blob = item.get("results")
        if not results_blob:
            continue
        try:
            result_payload = json.loads(results_blob)
        except json.JSONDecodeError:
            continue
        sql_id = result_payload.get("sql_id")
        sql_text = result_payload.get("sql")
        if sql_id and sql_text:
            sql_map[sql_id] = sql_text
    return sql_map


def inject_sql(dataset: List[dict], sql_map: Dict[str, str]) -> int:
    updated = 0
    for entry in dataset:
        sql_id = entry.get("sql_id")
        if not sql_id:
            continue
        sql_text = sql_map.get(sql_id)
        if not sql_text:
            continue
        entry["sql"] = sql_text
        updated += 1
    return updated


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, os.pardir, "data"))
    dataset_path = os.path.join(data_dir, "final_dataset.json")
    output_path = os.path.join(data_dir, "output.json")

    with open(dataset_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    sql_map = load_output_sqls(output_path)
    applied = inject_sql(dataset, sql_map)

    rendered = json.dumps(dataset, ensure_ascii=False, indent=4) + "\n"

    with open(dataset_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(f"Updated {applied} records with SQL text")


if __name__ == "__main__":
    main()
