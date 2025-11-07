"""Re-execute successful SQL entries that lack cached results."""

import argparse
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, List

import pymysql
from pymysql.cursors import DictCursor


def _clean_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _clean_rows(rows: Iterable[dict]) -> List[dict]:
    return [
        {key: _clean_value(val) for key, val in row.items()}
        for row in rows
    ]


def _load_entries(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_entries(path: Path, entries: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def refresh_results(path: Path, *, host: str, port: int, user: str, password: str, database: str) -> None:
    entries = _load_entries(path)
    pending = [entry for entry in entries if entry.get("success") and not entry.get("result")]
    if not pending:
        print("No successful entries with missing results found.")
        return

    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        cursorclass=DictCursor,
        autocommit=True,
    )

    try:
        for entry in pending:
            sql = entry.get("sql")
            if not sql:
                continue

            try:
                connection.ping(reconnect=True)
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                entry["result"] = _clean_rows(rows) if rows else None
                entry["error"] = None
                entry["success"] = True
                print(f"Updated {entry.get('sql_id', '<unknown>')} with {len(rows)} row(s).")
            except Exception as exc:  # noqa: PERF203 - keep broad for logging
                entry["success"] = False
                entry["error"] = str(exc)
                entry["result"] = None
                print(f"Failed to refresh {entry.get('sql_id', '<unknown>')}: {exc}")

    finally:
        connection.close()

    _write_entries(path, entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="T3/upload/dataset_exe_result.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9030)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="database_main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    refresh_results(
        path,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
    )


if __name__ == "__main__":
    main()
