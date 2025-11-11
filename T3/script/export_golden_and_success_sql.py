import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f'JSON 文件 {path} 的根类型为 {type(data)}, 无法处理')


def normalize_question(text: str) -> str:
    return (text or '').strip()


def extract_sql(entry: dict) -> str:
    golden_sql = entry.get('golden_sql')
    if isinstance(golden_sql, str) and golden_sql.strip():
        return golden_sql.strip()
    sql_text = entry.get('sql')
    return sql_text.strip() if isinstance(sql_text, str) else ''


def collect_golden_sql(final_dataset_path: Path) -> Tuple[List[dict], Dict[str, str]]:
    golden_rows: List[dict] = []
    sql_id_to_question: Dict[str, str] = {}

    for entry in load_json(final_dataset_path):
        sql_id = entry.get('sql_id')
        question = normalize_question(entry.get('question'))
        sql_text = extract_sql(entry)

        if sql_id:
            sql_id_to_question[sql_id] = question

        golden_flag = entry.get('golden_sql')
        if (golden_flag is True or isinstance(golden_flag, str)) and sql_text:
            golden_rows.append({'question': question, 'sql': sql_text})

    return golden_rows, sql_id_to_question


def is_success_entry(entry: dict) -> bool:
    if entry.get('success') is True:
        return True

    status = entry.get('status')
    if isinstance(status, str):
        lowered = status.strip().lower()
        if lowered in {'success', 'succeed', 'ok'}:
            return True
        if lowered in {'error', 'failed', 'fail'}:
            return False

    if entry.get('error') or entry.get('error_message') or entry.get('exception'):
        return False

    sql_text = entry.get('sql')
    if isinstance(sql_text, str) and sql_text.strip():
        # 若返回结果为 null 也视为执行成功（可能无结果）
        if 'result' not in entry:
            return True
        return entry.get('result') is not None

    return False


def load_score_map(score_path: Path) -> Dict[str, float]:
    if not score_path.exists():
        return {}

    score_map: Dict[str, float] = {}

    with score_path.open('r', encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped:
                header_skipped = True
                continue
            if len(row) < 2:
                continue

            sql_id = row[0].strip()
            try:
                score = float(row[1])
            except ValueError:
                continue

            if sql_id:
                score_map[sql_id] = score

    return score_map


def collect_ckpt_success(ckpt_root: Path, question_map: Dict[str, str]) -> List[dict]:
    success_rows: List[dict] = []

    for dataset_path in sorted(ckpt_root.glob('*/dataset_exe_result.json')):
        try:
            entries = load_json(dataset_path)
        except (json.JSONDecodeError, ValueError):
            continue

        score_map = load_score_map(dataset_path.with_name('score.csv'))

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not is_success_entry(entry):
                continue

            sql_text = entry.get('sql')
            if not isinstance(sql_text, str) or not sql_text.strip():
                continue

            sql_text = sql_text.strip()
            sql_id = entry.get('sql_id')
            question = ''

            # 只保留在得分文件中标记为正确的 SQL
            if isinstance(sql_id, str):
                scored_value = score_map.get(sql_id)
                if scored_value is None or scored_value <= 0:
                    continue
            else:
                # 无法确定对应得分时跳过
                continue

            if isinstance(sql_id, str) and sql_id in question_map:
                question = question_map[sql_id]
            elif isinstance(entry.get('question'), str):
                question = normalize_question(entry['question'])
            elif isinstance(sql_id, str):
                question = sql_id

            success_rows.append({'question': question, 'sql': sql_text})

    return success_rows


def remove_duplicates(rows: Iterable[dict]) -> List[dict]:
    deduped: List[dict] = []
    seen = set()

    for row in rows:
        key = (row.get('question', ''), row.get('sql', ''))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({'question': key[0], 'sql': key[1]})

    return deduped


def export_to_csv(rows: List[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['question', 'sql'])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    final_dataset_path = root / 'data' / 'final_dataset.json'
    ckpt_root = root / 'ckpt'
    output_file = root / 'export' / 'golden_and_success_sql.csv'

    golden_rows, question_map = collect_golden_sql(final_dataset_path)
    success_rows = collect_ckpt_success(ckpt_root, question_map)

    all_rows = remove_duplicates(golden_rows + success_rows)
    export_to_csv(all_rows, output_file)

    print(f'导出 {len(all_rows)} 条记录至 {output_file}')


if __name__ == '__main__':
    main()


