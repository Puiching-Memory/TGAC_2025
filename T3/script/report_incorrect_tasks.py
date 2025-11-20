"""
汇总各个复杂度下仍未正确的题目，便于分析。
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_final_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data:
        sql_id = item.get("sql_id")
        if not sql_id:
            continue
        result[sql_id] = item
    return result


def load_dataset_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    result = {}
    if isinstance(data, list):
        for item in data:
            sql_id = item.get("sql_id")
            if not sql_id:
                continue
            result[sql_id] = item
    return result


def load_scores(path: Path) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = { (k or "").strip().lstrip("\ufeff"): (v.strip() if isinstance(v, str) else v)
                    for k, v in raw_row.items() }
            sql_id = row.get("SQL ID") or row.get("sql_id")
            score_raw = row.get("得分") or row.get("score")
            if not sql_id or score_raw is None:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            scores.append((sql_id.strip(), score))
    return scores


def main() -> None:
    t3_root = Path(__file__).resolve().parents[1]
    ckpt_root = t3_root / "ckpt" / "V9_40.70_1116"

    dataset_info_path = t3_root / "data" / "final_dataset.json"
    score_path = ckpt_root / "score.csv"
    exe_result_path = ckpt_root / "dataset_exe_result.json"

    if not dataset_info_path.exists():
        raise FileNotFoundError(f"找不到数据集文件：{dataset_info_path}")

    dataset_map = load_final_dataset(dataset_info_path)
    exe_result_map = load_dataset_results(exe_result_path)
    scores = load_scores(score_path)

    failed_by_complexity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    table_usage_count: Dict[str, int] = defaultdict(int)
    
    # 统计每个难度级别的题目总数
    total_by_complexity: Dict[str, int] = defaultdict(int)
    for sql_id, info in dataset_map.items():
        complexity = info.get("复杂度", "未知")
        total_by_complexity[complexity] += 1

    for sql_id, score in scores:
        if score >= 1:
            continue
        info = dataset_map.get(sql_id, {})
        complexity = info.get("复杂度", "未知")
        question = (info.get("question") or "").strip().splitlines()[0] if info else ""
        knowledge = (info.get("knowledge") or "").strip()
        has_result = "是" if sql_id in exe_result_map else "否"
        tables = info.get("table_list", [])
        # 统计表使用次数
        for table in tables:
            table_usage_count[table] += 1
        failed_by_complexity[complexity].append(
            {
                "sql_id": sql_id,
                "question": question,
                "has_result": has_result,
                "knowledge": knowledge,
                "tables": tables,
            }
        )

    if not failed_by_complexity:
        print("所有题目均已正确。")
        return

    # 输出表使用统计
    if table_usage_count:
        print("=" * 80)
        print("未通过题目中的表使用统计（按使用次数降序）")
        print("-" * 80)
        sorted_tables = sorted(table_usage_count.items(), key=lambda x: x[1], reverse=True)
        for idx, (table, count) in enumerate(sorted_tables, 1):
            print(f"{idx}. {table}: {count} 次")
        print()

    for complexity in sorted(failed_by_complexity.keys()):
        items = failed_by_complexity[complexity]
        total = total_by_complexity.get(complexity, 0)
        print("=" * 80)
        print(f"复杂度：{complexity}（未通过 {len(items)} 题 / 总共 {total} 题）")
        print("-" * 80)
        for idx, item in enumerate(sorted(items, key=lambda x: x["sql_id"]), 1):
            tables = ", ".join(item["tables"]) if item["tables"] else "（未提供）"
            print(f"{idx}. {item['sql_id']}")
            print(f"   题干首行：{item['question']}")
            print(f"   涉及表：{tables}")
            print(f"   已生成执行结果：{item['has_result']}")
            if item["knowledge"]:
                snippet = item["knowledge"].splitlines()[0]
                print(f"   补充知识：{snippet}")
            print()


if __name__ == "__main__":
    main()

