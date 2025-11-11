import csv
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator, List, Sequence

import pymysql


# === 基础配置 ===
OUTPUT_DIR = Path("T3/export/database_csv")  # CSV 输出目录
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 9030,
    "user": "root",
    "password": "",
    "db": "database_main",
    "charset": "utf8mb4",
}
# 可选：指定需要忽略的表名（如审计表、临时表等）
EXCLUDE_TABLES: Sequence[str] = ()
# 每批次写入的行数，避免一次性加载过多数据造成内存压力
FETCH_BATCH_SIZE = 2_000


@contextmanager
def _db_connection() -> Iterator[pymysql.connections.Connection]:
    """管理数据库连接的上下文管理器。"""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.SSCursor)
        yield conn
    finally:
        if conn:
            conn.close()


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _list_tables(conn: pymysql.connections.Connection) -> List[str]:
    """获取当前数据库内所有基础表（排除视图）。"""
    sql = """
        SELECT TABLE_NAME
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s
          AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (DB_CONFIG["db"],))
        return [row[0] for row in cursor.fetchall()]


def _normalize_value(value):
    """将数据库中的值转换为适合写入 CSV 的格式。"""
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ")
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def export_table(conn: pymysql.connections.Connection, table_name: str) -> Path:
    """导出单张表为 CSV 文件。"""
    safe_name = table_name.replace("/", "_")
    output_path = OUTPUT_DIR / f"{safe_name}.csv"

    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM `{table_name}`")
        column_names = [desc[0] for desc in cursor.description]

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(column_names)

            while True:
                rows = cursor.fetchmany(FETCH_BATCH_SIZE)
                if not rows:
                    break

                normalized_rows = [
                    [_normalize_value(value) for value in row] for row in rows
                ]
                writer.writerows(normalized_rows)

    return output_path


def main() -> None:
    _ensure_output_dir()

    try:
        with _db_connection() as conn:
            tables = _list_tables(conn)
            if EXCLUDE_TABLES:
                tables = [t for t in tables if t not in set(EXCLUDE_TABLES)]

            total = len(tables)
            if total == 0:
                print("未找到可导出的表。")
                return

            print(f"共检测到 {total} 张表，开始导出...")
            for index, table in enumerate(tables, start=1):
                print(f"[{index}/{total}] 导出 `{table}`...", end=" ", flush=True)
                try:
                    export_table(conn, table)
                    print("完成")
                except Exception as exc:  # pylint: disable=broad-except
                    print("失败")
                    print(f"    错误：{exc}")
            print("全部导出任务完成。")
            print(f"CSV 文件输出目录：{OUTPUT_DIR.resolve()}")
    except pymysql.MySQLError as exc:
        print(f"数据库连接失败：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

