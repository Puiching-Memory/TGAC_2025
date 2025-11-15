from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ==== 开发者可根据实际路径调整以下常量 ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "ckpt"
FINAL_DATASET_PATH = REPO_ROOT / "data" / "final_dataset.json"
OUTPUT_DIR = REPO_ROOT / "data" / "correct_examples"
OUTPUT_ENCODING = "utf-8"
DB_CONNECTION_URL = "mysql+pymysql://root:@localhost:9030/database_main"
SQL_RESULT_LIMIT = 100
# ==========================================


@dataclass
class Record:
    sql_id: str
    sources: List[str]
    question: Optional[str]
    sql: str
    result: List[Mapping[str, Any]]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_question_lookup(final_dataset: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for item in final_dataset:
        sql_id = item.get("sql_id")
        question = item.get("question")
        if isinstance(sql_id, str) and isinstance(question, str):
            lookup[sql_id] = question
    return lookup


def collect_correct_from_ckpt(
    question_lookup: Mapping[str, str],
) -> List[Record]:
    records: List[Record] = []

    for score_path in sorted(CKPT_DIR.glob("*/score.csv")):
        version = score_path.parent.name
        correct_ids = load_correct_ids(score_path)
        dataset_path = score_path.parent / "dataset_exe_result.json"
        if not dataset_path.exists():
            continue

        dataset_entries = load_json(dataset_path)
        if not isinstance(dataset_entries, list):
            continue

        for entry in dataset_entries:
            if not isinstance(entry, Mapping):
                continue
            sql_id = entry.get("sql_id")
            if sql_id not in correct_ids:
                continue
            sql = entry.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                continue
            result = entry.get("result")
            if not isinstance(result, list):
                result = []
            cleaned_result: List[Mapping[str, Any]] = [
                row for row in result if isinstance(row, Mapping)
            ]

            record = Record(
                sql_id=sql_id,
                sources=[f"ckpt/{version}"],
                question=question_lookup.get(sql_id),
                sql=sql,
                result=cleaned_result,
            )
            records.append(record)

    return records


def load_correct_ids(score_path: Path) -> set[str]:
    correct_ids: set[str] = set()
    with score_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            sql_id = row.get("SQL ID") or row.get("sql_id")
            score = row.get("得分") or row.get("score")
            if not isinstance(sql_id, str):
                continue
            if isinstance(score, str) and score.strip() == "1":
                correct_ids.add(sql_id.strip())
    return correct_ids


def collect_golden_from_final_dataset(
    final_dataset: Iterable[Mapping[str, Any]],
    available_results: Mapping[str, List[Mapping[str, Any]]],
) -> List[Record]:
    records: List[Record] = []
    execution_cache: Dict[str, List[Mapping[str, Any]]] = {}
    for item in final_dataset:
        if not isinstance(item, Mapping):
            continue
        if not item.get("golden_sql"):
            continue
        sql_id = item.get("sql_id")
        sql = item.get("sql")
        question = item.get("question")
        if not isinstance(sql_id, str) or not isinstance(sql, str):
            continue
        result = available_results.get(sql_id, [])
        if not result:
            cache_key = sql.strip()
            if cache_key not in execution_cache:
                execution_cache[cache_key] = execute_sql(cache_key)
            result = execution_cache[cache_key]
        record = Record(
            sql_id=sql_id,
            sources=["final_dataset"],
            question=question if isinstance(question, str) else None,
            sql=sql,
            result=result,
        )
        records.append(record)
    return records


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for txt_file in path.glob("*.txt"):
        txt_file.unlink()


def sanitize_filename(text: str) -> str:
    safe_text = re.sub(r"[^A-Za-z0-9_.+-]", "_", text)
    safe_text = re.sub(r"_+", "_", safe_text).strip("_")
    return safe_text or "record"


def render_record(record: Record) -> str:
    lines: List[str] = [
        f"SQL ID: {record.sql_id}",
        "",
        "用户问题:",
        record.question.strip() if record.question else "（暂无）",
        "",
        "SQL:",
        record.sql.strip(),
        "",
        "运行结果:",
    ]

    if record.result:
        for row in record.result:
            lines.append(json.dumps(row, ensure_ascii=False))
    else:
        lines.append("（暂无数据）")

    return "\n".join(lines) + "\n"


def group_best_results(records: Iterable[Record]) -> Dict[str, List[Mapping[str, Any]]]:
    best: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if record.result:
            best.setdefault(record.sql_id, record.result)
    return best


_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DB_CONNECTION_URL)
    return _engine


def normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def execute_sql(sql: str) -> List[Mapping[str, Any]]:
    trimmed_sql = sql.strip().rstrip(";")
    if not trimmed_sql:
        return []

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(trimmed_sql))
            rows = result.fetchmany(SQL_RESULT_LIMIT)
            column_names = result.keys()
            result.close()
            data: List[Dict[str, Any]] = []
            for row in rows:
                row_dict = {
                    col: normalize_value(val)
                    for col, val in zip(column_names, row)
                }
                data.append(row_dict)
            return data
    except Exception as exc:  # noqa: BLE001
        print(f"执行 SQL 失败: {exc}")
        return [{"__error__": str(exc)}]


def merge_duplicates(records: Iterable[Record]) -> List[Record]:
    unique: Dict[str, Record] = {}
    for record in records:
        key_components = (
            record.sql_id,
            (record.question or "").strip(),
            record.sql.strip(),
            json.dumps(record.result, ensure_ascii=False, sort_keys=True),
        )
        key = "\u0000".join(key_components)
        if key in unique:
            existing = unique[key]
            for src in record.sources:
                if src not in existing.sources:
                    existing.sources.append(src)
        else:
            unique[key] = Record(
                sql_id=record.sql_id,
                sources=list(record.sources),
                question=record.question,
                sql=record.sql,
                result=record.result,
            )
    return list(unique.values())


def main() -> None:
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"未找到 final_dataset.json: {FINAL_DATASET_PATH}")

    final_dataset = load_json(FINAL_DATASET_PATH)
    if not isinstance(final_dataset, list):
        raise ValueError("final_dataset.json 格式错误，应为数组。")

    question_lookup = build_question_lookup(final_dataset)
    ckpt_records = collect_correct_from_ckpt(question_lookup)
    results_lookup = group_best_results(ckpt_records)
    golden_records = collect_golden_from_final_dataset(final_dataset, results_lookup)

    ensure_output_dir(OUTPUT_DIR)

    all_records = merge_duplicates(ckpt_records + golden_records)
    for record in all_records:
        source_tag = "+".join(record.sources)
        file_name = f"{sanitize_filename(record.sql_id)}__{sanitize_filename(source_tag)}.txt"
        output_path = OUTPUT_DIR / file_name
        output_path.write_text(render_record(record), encoding=OUTPUT_ENCODING)

    print(f"已输出 {len(all_records)} 个正确示例到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

