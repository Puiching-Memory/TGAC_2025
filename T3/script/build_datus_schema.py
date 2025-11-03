"""Convert T3 schema.json into Datus schema ingestion artifacts.

This helper reads the Text2SQL benchmark schema description, synthesises
CREATE TABLE statements for each table, and emits JSONL files that match the
`SchemaStorage` bootstrap format used by the Datus agent.  The generated files
can be referenced by Datus' `init_local_schema` pipeline to populate LanceDB
metadata for StarRocks (or other) databases.

Usage example (PowerShell):

    python T3\\script\\build_datus_schema.py \\
        --input T3\\data\\schema.json \\
        --catalog default_catalog \\
        --database database_main \\
        --schema ods \\
        --table-output T3\\upload\\datus_schema_tables.jsonl

By default the script writes `datus_schema_tables.jsonl` under the `upload`
folder.  The output is a JSON lines file where each record contains the table
identifier, catalog/database/schema coordinates, the table name, and an
auto-generated CREATE TABLE definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Minimal mapping from loose column type names in schema.json to canonical
# StarRocks types.  Extend this dictionary when new logical types appear.
_TYPE_MAP: Dict[str, str] = {
    "string": "STRING",
    "bigint": "BIGINT",
    "int": "INT",
    "integer": "INT",
    "double": "DOUBLE",
    "float": "FLOAT",
    "decimal": "DECIMAL",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
    "date": "DATE",
    "datetime": "DATETIME",
    "timestamp": "DATETIME",
}

# Characters that need escaping inside single-quoted StarRocks comments.
_SINGLE_QUOTE = "'"
_ESCAPE_QUOTE = "''"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render schema.json to Datus JSONL artifacts")
    parser.add_argument("--input", type=Path, default=Path("T3/data/schema.json"), help="Path to schema.json")
    parser.add_argument("--catalog", default="", help="Catalog name for identifier synthesis")
    parser.add_argument("--database", default="", help="Database name for identifier synthesis")
    parser.add_argument("--schema", dest="schema_name", default="", help="Schema name for identifier synthesis")
    parser.add_argument(
        "--table-type",
        default="table",
        choices=["table", "view", "mv", "full"],
        help="Logical table type recorded in Datus metadata",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("T3/upload/datus_schema_tables.jsonl"),
        help="Output JSONL path for table definitions",
    )
    return parser.parse_args()


def _load_schema(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"schema source not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("schema.json must contain a list of table descriptions")
    return data


def _escape_comment(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.replace(_SINGLE_QUOTE, _ESCAPE_QUOTE)


def _normalise_type(type_name: Optional[str]) -> str:
    if not type_name:
        return "STRING"
    cleaned = type_name.strip().lower()
    return _TYPE_MAP.get(cleaned, cleaned.upper())


def _compose_identifier(catalog: str, database: str, schema_name: str, table: str) -> str:
    parts: List[str] = []
    if catalog:
        parts.append(catalog)
    if database:
        parts.append(database)
    if schema_name:
        parts.append(schema_name)
    parts.append(table)
    return ".".join(parts)


def _quote_identifier(name: str) -> str:
    """Wrap identifiers with backticks to protect reserved words."""
    return f"`{name}`"


def _build_column_line(column: Dict[str, object]) -> str:
    raw_name = str(column.get("col"))
    dtype = _normalise_type(column.get("type"))
    comment = _escape_comment(column.get("description", ""))
    comment_clause = f" COMMENT '{comment}'" if comment else ""
    return f"    {_quote_identifier(raw_name)} {dtype}{comment_clause}"


def _build_create_statement(table_name: str, columns: Iterable[Dict[str, object]]) -> str:
    column_lines = [_build_column_line(col) for col in columns if col.get("col")]
    if not column_lines:
        column_lines.append("    `id` STRING COMMENT 'placeholder column generated from empty schema'")
    column_section = ",\n".join(column_lines)
    return f"CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} (\n{column_section}\n);"


def _build_table_record(
    table: Dict[str, object],
    catalog: str,
    database: str,
    schema_name: str,
    table_type: str,
) -> Dict[str, object]:
    table_name = str(table.get("table_name"))
    columns = table.get("columns") or []
    ddl = _build_create_statement(table_name, columns)
    identifier = _compose_identifier(catalog, database, schema_name, table_name)
    return {
        "identifier": identifier,
        "catalog_name": catalog,
        "database_name": database,
        "schema_name": schema_name,
        "table_name": table_name,
        "table_type": table_type,
        "definition": ddl,
        "table_description": table.get("table_description", ""),
    }


def _write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = _parse_args()
    tables = _load_schema(args.input)
    records = [
        _build_table_record(table, args.catalog, args.database, args.schema_name, args.table_type)
        for table in tables
        if table.get("table_name")
    ]
    if not records:
        raise ValueError("no valid table entries were produced")
    _write_jsonl(records, args.table_output)
    print(f"Generated {len(records)} table definition records -> {args.table_output}")


if __name__ == "__main__":
    main()
