"""Merge correct SQL answers across checkpoint versions."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class SqlRecord:
    sql_id: str
    sql: str | None
    result: object
    versions: List[str] = field(default_factory=list)

    def add_version(self, version: str) -> None:
        if version not in self.versions:
            self.versions.append(version)


def load_dataset(dataset_path: Path) -> Dict[str, Dict[str, object]]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {entry["sql_id"]: entry for entry in data}


def collect_correct_ids(score_path: Path) -> List[str]:
    correct_ids: List[str] = []
    with score_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row:
                continue
            score_raw = row.get("得分") or row.get("score") or "0"
            try:
                score_val = float(score_raw)
            except ValueError:
                score_val = 0.0
            if abs(score_val - 1.0) < 1e-9:
                sql_id = row.get("SQL ID") or row.get("\ufeffSQL ID") or row.get("sql_id")
                if sql_id:
                    correct_ids.append(sql_id.strip())
    return correct_ids


def build_key(entry: Dict[str, object]) -> Tuple[str, str, str]:
    sql_id = entry.get("sql_id", "")
    sql_text = entry.get("sql")
    result_obj = entry.get("result")
    sql_norm = sql_text.strip() if isinstance(sql_text, str) else ""
    result_norm = json.dumps(result_obj, ensure_ascii=False, sort_keys=True)
    return sql_id, sql_norm, result_norm


def merge_versions(base_dir: Path) -> Tuple[List[SqlRecord], Dict[str, Dict[str, int]], Dict[str, List[str]]]:
    combined: Dict[Tuple[str, str, str], SqlRecord] = {}
    version_stats: Dict[str, Dict[str, int]] = {}
    missing_lookup: Dict[str, List[str]] = defaultdict(list)

    for version_dir in sorted(base_dir.glob("V*")):
        if not version_dir.is_dir():
            continue
        version_name = version_dir.name
        dataset_path = version_dir / "dataset_exe_result.json"
        score_path = version_dir / "score.csv"
        if not dataset_path.exists() or not score_path.exists():
            continue

        dataset = load_dataset(dataset_path)
        correct_ids = collect_correct_ids(score_path)
        version_stats[version_name] = {
            "correct_total": len(correct_ids),
            "records_in_combined": 0,
            "exclusive_records": 0,
            "shared_records": 0,
        }

        for sql_id in correct_ids:
            entry = dataset.get(sql_id)
            if not entry:
                missing_lookup[version_name].append(sql_id)
                continue
            key = build_key(entry)
            record = combined.get(key)
            if record is None:
                record = SqlRecord(sql_id=entry["sql_id"], sql=entry.get("sql"), result=entry.get("result"))
                combined[key] = record
            record.add_version(version_name)

    for record in combined.values():
        record.versions.sort()
        if len(record.versions) == 1:
            version = record.versions[0]
            version_stats[version]["exclusive_records"] += 1
            version_stats[version]["records_in_combined"] += 1
        else:
            for version in record.versions:
                version_stats[version]["shared_records"] += 1
                version_stats[version]["records_in_combined"] += 1

    return list(combined.values()), version_stats, missing_lookup


def write_outputs(
    records: Iterable[SqlRecord],
    version_stats: Dict[str, Dict[str, int]],
    missing_lookup: Dict[str, List[str]],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records_sorted = sorted(records, key=lambda rec: (int(rec.sql_id.split("_")[-1]), rec.sql or ""))
    dataset_serializable = [
        {
            "sql_id": rec.sql_id,
            "sql": rec.sql,
            "result": rec.result,
        }
        for rec in records_sorted
    ]

    dataset_path = output_dir / "dataset_exe_result.json"
    with dataset_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset_serializable, fh, ensure_ascii=False, indent=2)

    duplicated_ids = defaultdict(int)
    for rec in records_sorted:
        duplicated_ids[rec.sql_id] += 1
    duplicates = {sql_id: count for sql_id, count in duplicated_ids.items() if count > 1}

    grouped_by_sql_id: Dict[str, List[SqlRecord]] = defaultdict(list)
    for rec in records_sorted:
        grouped_by_sql_id[rec.sql_id].append(rec)

    same_answer_entries = []
    for sql_id, items in grouped_by_sql_id.items():
        version_set = {ver for item in items for ver in item.versions}
        seen_results: set[str] = set()
        unique_results: List[object] = []
        variants = []
        for item in items:
            result_key = json.dumps(item.result, ensure_ascii=False, sort_keys=True)
            if result_key not in seen_results:
                seen_results.add(result_key)
                unique_results.append(item.result)
            variants.append(
                {
                    "sql": item.sql,
                    "result": item.result,
                    "versions": item.versions,
                }
            )
        same_answer_entries.append(
            {
                "sql_id": sql_id,
                "variant_count": len(items),
                "result_variant_count": len(unique_results),
                "all_versions": sorted(version_set),
                "results": unique_results,
                "variants": variants,
            }
        )

    same_answer_entries.sort(key=lambda entry: (int(entry["sql_id"].split("_")[-1]), -entry["variant_count"]))

    same_answer_path = output_dir / "same_answer_variants.json"
    with same_answer_path.open("w", encoding="utf-8") as fh:
        json.dump(same_answer_entries, fh, ensure_ascii=False, indent=2)

    stats = {
        "版本统计": {
            version: {
                "正确SQL数量": info["correct_total"],
                "合并后条目数": info["records_in_combined"],
                "仅该版本条数": info["exclusive_records"],
                "跨版本共享条数": info["shared_records"],
            }
            for version, info in sorted(version_stats.items())
        },
        "合并SQL条目数": len(records_sorted),
        "去重后SQL编号数": len({rec.sql_id for rec in records_sorted}),
        "存在多SQL实现的SQL编号": {sql_id: count for sql_id, count in sorted(duplicates.items())},
        "缺失条目": {version: sorted(ids) for version, ids in sorted(missing_lookup.items()) if ids},
    }

    stats_path = output_dir / "combined_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "ckpt"
    output_dir = base_dir / "combined"
    records, version_stats, missing_lookup = merge_versions(base_dir)
    stats = write_outputs(records, version_stats, missing_lookup, output_dir)

    print("处理版本:", "、".join(sorted(version_stats)))
    print("合并SQL条目数:", stats["合并SQL条目数"])
    for version, info in sorted(version_stats.items()):
        print(
            f"{version}: 正确SQL数量={info['correct_total']}, "
            f"合并后条目数={info['records_in_combined']}, "
            f"仅该版本条数={info['exclusive_records']}, "
            f"跨版本共享条数={info['shared_records']}"
        )
    if stats["存在多SQL实现的SQL编号"]:
        print("存在多SQL实现且结果一致的SQL编号:")
        for sql_id, count in stats["存在多SQL实现的SQL编号"].items():
            print(f"  {sql_id}: {count} 条SQL")
    if stats["缺失条目"]:
        print("存在缺失条目:")
        for version, missing in stats["缺失条目"].items():
            print(f"  {version}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
