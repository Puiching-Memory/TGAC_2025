from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import re

# ==== 开发者可根据实际路径调整以下常量 ====
REPO_ROOT = Path(__file__).resolve().parents[1]
MSCHEMA_PATH = REPO_ROOT / "data" / "mschema_database_main.json"
SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"
OUTPUT_DIR = REPO_ROOT / "data" / "merged_schema_toon"
TOON_INDENT = 2
TOON_DELIMITER = ","
TOON_LIB_SRC = REPO_ROOT / "lib" / "toon-python" / "src"
# ==========================================


def ensure_toon_format() -> None:
    try:
        import toon_format  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        sys.path.append(str(TOON_LIB_SRC))


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_schema_lookup(schema_tables: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for table in schema_tables:
        name = table.get("table_name")
        if isinstance(name, str):
            lookup[name] = table
    return lookup


def build_column_lookup(schema_table: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for col in schema_table.get("columns", []) or []:
        name = col.get("col")
        if isinstance(name, str):
            result[name] = col
    return result


def merge_table(
    table_name: str,
    mschema_table: Optional[Mapping[str, Any]],
    schema_table: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    schema_columns_lookup: Dict[str, Mapping[str, Any]] = (
        build_column_lookup(schema_table) if schema_table else {}
    )
    schema_column_order: List[str] = []
    if schema_table:
        schema_column_order = [
            col.get("col")
            for col in schema_table.get("columns", []) or []
            if isinstance(col.get("col"), str)
        ]

    mschema_fields: Mapping[str, Any] = mschema_table.get("fields", {}) if mschema_table else {}

    columns: List[Dict[str, Any]] = []
    used_schema_columns: set[str] = set()

    for field_name, field_info in mschema_fields.items():
        schema_col = schema_columns_lookup.get(field_name)
        merged_col = merge_column(field_name, field_info, schema_col)
        columns.append(merged_col)
        if schema_col:
            used_schema_columns.add(field_name)

    for column_name in schema_column_order:
        if column_name and column_name not in used_schema_columns:
            schema_col = schema_columns_lookup[column_name]
            merged_col = merge_column(column_name, None, schema_col)
            columns.append(merged_col)

    table_comment_candidates: List[str] = []
    if schema_table:
        desc = schema_table.get("table_description")
        if isinstance(desc, str) and desc.strip():
            table_comment_candidates.append(desc.strip())
    if mschema_table:
        comment = mschema_table.get("comment")
        if isinstance(comment, str) and comment.strip():
            table_comment_candidates.append(comment.strip())

    merged_table: Dict[str, Any] = {
        "table_name": table_name,
        "description": " | ".join(dict.fromkeys(table_comment_candidates)) if table_comment_candidates else None,
        "columns": columns,
    }

    if mschema_table:
        examples = mschema_table.get("examples")
        if isinstance(examples, list) and examples:
            merged_table["examples"] = examples

    return prune_empty(merged_table)


def merge_column(
    column_name: str,
    mschema_field: Optional[Mapping[str, Any]],
    schema_column: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    column: Dict[str, Any] = {"name": column_name}

    schema_type = None
    schema_description = None
    if schema_column:
        schema_type = schema_column.get("type")
        schema_description = schema_column.get("description")

    mschema_type = None
    if mschema_field:
        mschema_type = mschema_field.get("type")

    preferred_type = next(
        (
            t
            for t in (schema_type, mschema_type)
            if isinstance(t, str) and t.strip()
        ),
        None,
    )

    if preferred_type:
        column["type"] = preferred_type

    if schema_type and mschema_type and schema_type != mschema_type:
        column["type_mschema"] = mschema_type

    if schema_description and isinstance(schema_description, str):
        column["description"] = schema_description.strip() or None

    if mschema_field:
        for attr in ("primary_key", "nullable", "autoincrement"):
            if attr in mschema_field:
                column[attr] = mschema_field[attr]
        if "default" in mschema_field:
            column["default"] = mschema_field["default"]
        comment = mschema_field.get("comment")
        if isinstance(comment, str) and comment.strip():
            column["comment"] = comment.strip()
        examples = mschema_field.get("examples")
        if isinstance(examples, list) and examples:
            column["examples"] = examples

    return prune_empty(column)


def prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned_dict: Dict[str, Any] = {}
        for key, val in value.items():
            pruned_value = prune_empty(val)
            if pruned_value is None:
                continue
            if isinstance(pruned_value, (list, dict)) and not pruned_value:
                continue
            pruned_dict[key] = pruned_value
        return pruned_dict

    if isinstance(value, list):
        pruned_list: List[Any] = []
        for item in value:
            pruned_item = prune_empty(item)
            if pruned_item is None:
                continue
            if isinstance(pruned_item, (list, dict)) and not pruned_item:
                continue
            pruned_list.append(pruned_item)
        return pruned_list

    if isinstance(value, str):
        return value if value.strip() else None

    return value


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def write_field_to_toon(field_data: Mapping[str, Any], output_path: Path, indent: int, delimiter: str) -> None:
    from toon_format import encode  # type: ignore

    options = {
        "indent": indent,
        "delimiter": delimiter,
    }
    encoded = encode(field_data, options)
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(encoded)


def run_merge(mschema_path: Path, schema_path: Path, output_dir: Path, indent: int, delimiter: str) -> int:
    ensure_toon_format()

    mschema_data = load_json(mschema_path)
    schema_data = load_json(schema_path)

    if not isinstance(mschema_data, Mapping):
        raise ValueError("mschema_database_main.json 格式不正确，应为对象。")
    if not isinstance(schema_data, list):
        raise ValueError("schema.json 格式不正确，应为数组。")

    schema_lookup = build_schema_lookup(schema_data)
    mschema_tables = mschema_data.get("tables", {})
    if not isinstance(mschema_tables, Mapping):
        raise ValueError("mschema_database_main.json 中缺少 tables 对象。")

    merged_tables: Dict[str, Dict[str, Any]] = {}

    for table_name, mschema_table in mschema_tables.items():
        if not isinstance(table_name, str):
            continue
        if not isinstance(mschema_table, Mapping):
            continue
        schema_table = schema_lookup.get(table_name)
        merged_tables[table_name] = merge_table(table_name, mschema_table, schema_table)

    for table_name, schema_table in schema_lookup.items():
        if table_name in merged_tables:
            continue
        merged_tables[table_name] = merge_table(table_name, None, schema_table)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧的表级输出
    if output_dir.exists():
        for old_file in output_dir.glob("*.txt"):
            old_file.unlink()

    total_fields = 0

    for table_name, table_data in merged_tables.items():
        columns: List[Mapping[str, Any]] = table_data.get("columns", []) or []
        for column in columns:
            column_name = column.get("name")
            if not isinstance(column_name, str):
                continue

            field_entry: Dict[str, Any] = {"table_name": table_name}
            field_entry.update(column)

            safe_table = sanitize_filename(table_name)
            safe_column = sanitize_filename(column_name)
            output_file = output_dir / f"{safe_table}__{safe_column}.txt"

            write_field_to_toon(field_entry, output_file, indent=indent, delimiter=delimiter)
            total_fields += 1

    return total_fields


def main() -> None:
    total_fields = run_merge(
        mschema_path=MSCHEMA_PATH,
        schema_path=SCHEMA_PATH,
        output_dir=OUTPUT_DIR,
        indent=TOON_INDENT,
        delimiter=TOON_DELIMITER,
    )
    print(f"已生成 {total_fields} 个字段的 TOON 文本，输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

