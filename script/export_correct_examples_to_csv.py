from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

# TOON 格式支持
REPO_ROOT = Path(__file__).resolve().parents[1]
TOON_LIB_SRC = REPO_ROOT / "lib" / "toon-python" / "src"

def ensure_toon_format() -> None:
    """确保 toon_format 模块可用"""
    try:
        import toon_format  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(TOON_LIB_SRC))

ensure_toon_format()

# ==== 开发者可根据实际路径调整以下常量 ====
CKPT_DIR = REPO_ROOT / "ckpt"
FINAL_DATASET_PATH = REPO_ROOT / "data" / "final_dataset.json"
OUTPUT_CSV_PATH = REPO_ROOT / "data" / "SQL 示例库.csv"
OUTPUT_ENCODING = "utf-8-sig"  # 使用 utf-8-sig 以支持 Excel 正确显示中文
# ==========================================


@dataclass
class Record:
    sql_id: str
    sources: List[str]
    question: Optional[str]
    sql: str
    result: List[Mapping[str, Any]]
    table_list: Optional[List[str]] = None


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


def build_table_list_lookup(final_dataset: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for item in final_dataset:
        sql_id = item.get("sql_id")
        table_list = item.get("table_list")
        if isinstance(sql_id, str) and isinstance(table_list, list):
            lookup[sql_id] = [str(t) for t in table_list if isinstance(t, (str, int, float))]
    return lookup


def load_correct_ids(score_path: Path) -> set[str]:
    """加载正确示例的SQL ID（得分为1的记录）"""
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


def collect_correct_from_ckpt(
    question_lookup: Mapping[str, str],
    table_list_lookup: Mapping[str, List[str]],
) -> List[Record]:
    """从ckpt目录收集正确示例"""
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
                table_list=table_list_lookup.get(sql_id),
            )
            records.append(record)

    return records


def collect_golden_from_final_dataset(
    final_dataset: Iterable[Mapping[str, Any]],
    available_results: Mapping[str, List[Mapping[str, Any]]],
) -> List[Record]:
    """从final_dataset收集golden SQL示例"""
    records: List[Record] = []
    for item in final_dataset:
        if not isinstance(item, Mapping):
            continue
        if not item.get("golden_sql"):
            continue
        sql_id = item.get("sql_id")
        sql = item.get("sql")
        question = item.get("question")
        table_list = item.get("table_list")
        if not isinstance(sql_id, str) or not isinstance(sql, str):
            continue
        result = available_results.get(sql_id, [])
        cleaned_table_list: Optional[List[str]] = None
        if isinstance(table_list, list):
            cleaned_table_list = [str(t) for t in table_list if isinstance(t, (str, int, float))]
        record = Record(
            sql_id=sql_id,
            sources=["final_dataset"],
            question=question if isinstance(question, str) else None,
            sql=sql,
            result=result,
            table_list=cleaned_table_list,
        )
        records.append(record)
    return records


def group_best_results(records: Iterable[Record]) -> Dict[str, List[Mapping[str, Any]]]:
    """按sql_id分组，获取最佳结果"""
    best: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if record.result:
            best.setdefault(record.sql_id, record.result)
    return best


def merge_duplicates(records: Iterable[Record]) -> List[Record]:
    """合并重复记录"""
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
            # 如果现有记录没有表信息，但新记录有，则更新
            if not existing.table_list and record.table_list:
                existing.table_list = record.table_list
        else:
            unique[key] = Record(
                sql_id=record.sql_id,
                sources=list(record.sources),
                question=record.question,
                sql=record.sql,
                result=record.result,
                table_list=record.table_list,
            )
    return list(unique.values())


def format_sources(sources: List[str]) -> str:
    """格式化数据源列表为字符串"""
    return ", ".join(sources)


def export_to_csv(records: List[Record], output_path: Path) -> None:
    """导出记录到CSV文件"""
    # CSV列：问题描述（必填）,示例 SQL（必填）,生效数据源,高级应用
    with output_path.open("w", encoding=OUTPUT_ENCODING, newline="") as fp:
        writer = csv.writer(fp)
        # 写入表头
        writer.writerow(["问题描述（必填）", "示例 SQL（必填）", "生效数据源", "高级应用"])
        
        # 写入数据
        for record in records:
            question = record.question or ""  # 问题描述
            sql = record.sql.strip()  # 示例 SQL
            sources = "StarRocks"  # 生效数据源（固定为StarRocks）
            advanced_app = ""  # 高级应用（当前为空，可根据需要填充）
            
            writer.writerow([question, sql, sources, advanced_app])


def main() -> None:
    """主函数：收集正确示例并导出为CSV"""
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"未找到 final_dataset.json: {FINAL_DATASET_PATH}")

    final_dataset = load_json(FINAL_DATASET_PATH)
    if not isinstance(final_dataset, list):
        raise ValueError("final_dataset.json 格式错误，应为数组。")

    question_lookup = build_question_lookup(final_dataset)
    table_list_lookup = build_table_list_lookup(final_dataset)
    
    # 收集正确示例
    ckpt_records = collect_correct_from_ckpt(question_lookup, table_list_lookup)
    results_lookup = group_best_results(ckpt_records)
    golden_records = collect_golden_from_final_dataset(final_dataset, results_lookup)

    # 合并去重
    all_records = merge_duplicates(ckpt_records + golden_records)
    
    # 导出到CSV
    export_to_csv(all_records, OUTPUT_CSV_PATH)
    
    print(f"已导出 {len(all_records)} 个正确示例到 {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
