from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ============================================================================
# 配置变量 - 可根据实际需求修改以下配置
# ============================================================================

# 路径配置
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"
OUTPUT_DIR = REPO_ROOT / "data" / "generated_schema_toon"
TOON_LIB_SRC = REPO_ROOT / "lib" / "toon-python" / "src"

# TOON 格式配置
TOON_INDENT = 2
TOON_DELIMITER = ","

# 数据库配置（可选，如果不需要检查字段是否为null，设置为 None）
# 格式示例：mysql+pymysql://user:password@host:port/database
# 或：starrocks://user:password@host:port/database
DATABASE_URL: Optional[str] = "mysql+pymysql://root:@127.0.0.1:9030/database_main"

# 是否跳过全部为null的字段（需要提供 DATABASE_URL 才能生效）
SKIP_ALL_NULL = True

# ============================================================================


def ensure_toon_format() -> None:
    """确保 toon_format 模块可用"""
    try:
        import toon_format  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        sys.path.append(str(TOON_LIB_SRC))


def load_json(path: Path) -> Any:
    """加载 JSON 文件"""
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_type(type_str: str) -> str:
    """规范化类型字符串"""
    if not isinstance(type_str, str):
        return "string"
    type_str = type_str.strip().upper()
    # 处理常见类型映射
    type_mapping = {
        "BIGINT": "bigint",
        "INT": "int",
        "INTEGER": "int",
        "VARCHAR": "string",
        "STRING": "string",
        "TEXT": "string",
        "CHAR": "string",
        "DATE": "date",
        "DATETIME": "datetime",
        "TIMESTAMP": "timestamp",
        "TIME": "time",
        "FLOAT": "float",
        "DOUBLE": "double",
        "DECIMAL": "decimal",
        "BOOLEAN": "boolean",
        "BOOL": "boolean",
    }
    # 提取基础类型（去掉括号和长度）
    base_type = type_str.split("(")[0].strip()
    return type_mapping.get(base_type, type_str.lower())


def build_table_from_schema(schema_table: Mapping[str, Any]) -> Dict[str, Any]:
    """从 schema.json 表信息构建表数据结构"""
    table_name = schema_table.get("table_name")
    if not isinstance(table_name, str):
        raise ValueError("表名必须是字符串")
    
    table_description = schema_table.get("table_description", "")
    columns_data = schema_table.get("columns", [])
    
    columns: List[Dict[str, Any]] = []
    
    for col_info in columns_data:
        col_name = col_info.get("col")
        if not isinstance(col_name, str):
            continue
        
        col_type = col_info.get("type", "string")
        col_description = col_info.get("description", "")
        
        # 构建字段信息
        column: Dict[str, Any] = {
            "name": col_name,
            "type": normalize_type(col_type),  # 规范化类型
        }
        
        # 保留原始类型（如果与规范化类型不同）
        if col_type and normalize_type(col_type) != col_type.lower():
            column["type_original"] = col_type
        
        # 添加描述
        if col_description and isinstance(col_description, str):
            desc = col_description.strip()
            if desc:
                column["description"] = desc
        
        # 默认字段属性（schema.json 中没有这些信息，使用默认值）
        column["primary_key"] = False
        column["nullable"] = True
        column["autoincrement"] = False
        
        columns.append(prune_empty(column))
    
    # 构建表信息
    table: Dict[str, Any] = {
        "table_name": table_name,
        "columns": columns,
    }
    
    if table_description and isinstance(table_description, str):
        desc = table_description.strip()
        if desc:
            table["description"] = desc
    
    return prune_empty(table)


def prune_empty(value: Any) -> Any:
    """递归清理空值"""
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
    """清理文件名，移除非法字符"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def check_column_all_null(
    engine: Optional[Engine],
    table_name: str,
    column_name: str,
    dialect: Optional[str] = None
) -> bool:
    """
    检查字段是否全部为null
    
    Args:
        engine: 数据库引擎（可选）
        table_name: 表名
        column_name: 列名
        dialect: 数据库方言（可选，如果为None则从engine推断）
    
    Returns:
        True 如果字段全部为null，False 否则
    """
    if engine is None:
        return False
    
    try:
        if dialect is None:
            dialect = engine.dialect.name
        
        # 根据不同的数据库方言构建查询
        # 注意：使用参数化查询避免SQL注入，但这里表名和列名需要特殊处理
        # 对于表名和列名，我们需要确保它们是合法的标识符
        # 使用反引号或双引号包裹（根据数据库方言）
        if dialect in ('mysql', 'starrocks'):
            # MySQL/StarRocks 语法，使用反引号
            query = text(f"""
                SELECT COUNT(*) as total_count,
                       COUNT(`{column_name}`) as non_null_count
                FROM `{table_name}`
            """)
        elif dialect == 'postgresql':
            # PostgreSQL 语法，使用双引号
            query = text(f"""
                SELECT COUNT(*) as total_count,
                       COUNT("{column_name}") as non_null_count
                FROM "{table_name}"
            """)
        else:
            # 默认SQL语法
            query = text(f"""
                SELECT COUNT(*) as total_count,
                       COUNT({column_name}) as non_null_count
                FROM {table_name}
            """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                total_count = row[0] if hasattr(row, '__getitem__') else row.total_count
                non_null_count = row[1] if hasattr(row, '__getitem__') else row.non_null_count
                # 如果总行数为0，或者非null数量为0，则认为字段全部为null
                return total_count == 0 or non_null_count == 0
        return False
    except Exception as e:
        # 如果查询失败，记录警告但不中断流程
        print(f"警告: 无法检查字段 {table_name}.{column_name} 是否为null: {e}")
        return False


def should_skip_field(
    column: Mapping[str, Any],
    engine: Optional[Engine] = None,
    table_name: Optional[str] = None
) -> bool:
    """
    判断是否应该跳过某个字段
    
    Args:
        column: 字段信息
        engine: 数据库引擎（可选）
        table_name: 表名（可选，用于数据库查询）
    
    Returns:
        True 如果应该跳过，False 否则
    """
    # 如果提供了数据库连接，查询字段是否全部为null
    if engine is not None and table_name is not None:
        column_name = column.get("name")
        if isinstance(column_name, str):
            if check_column_all_null(engine, table_name, column_name):
                return True
    
    return False


def write_field_to_toon(field_data: Mapping[str, Any], output_path: Path, indent: int, delimiter: str) -> None:
    """将字段数据写入 TOON 格式文件"""
    from toon_format import encode  # type: ignore

    options = {
        "indent": indent,
        "delimiter": delimiter,
    }
    encoded = encode(field_data, options)
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(encoded)


def generate_schema_and_export(
    schema_path: Path,
    output_dir: Path,
    indent: int,
    delimiter: str,
    database_url: Optional[str] = None,
    skip_all_null: bool = True
) -> int:
    """
    从 schema.json 生成数据结构并导出为字段切片的 TOON 文件
    
    Args:
        schema_path: schema.json 文件路径
        output_dir: 输出目录
        indent: TOON 格式缩进
        delimiter: TOON 格式分隔符
        database_url: 数据库连接URL（可选，格式如：mysql+pymysql://user:password@host:port/database）
        skip_all_null: 是否跳过全部为null的字段
    """
    ensure_toon_format()

    # 创建数据库引擎（如果提供了数据库URL）
    engine: Optional[Engine] = None
    if database_url:
        try:
            engine = create_engine(database_url)
            print(f"已连接到数据库: {database_url.split('@')[-1] if '@' in database_url else database_url}")
        except Exception as e:
            print(f"警告: 无法连接到数据库: {e}，将跳过数据库检查")
            engine = None
    elif skip_all_null:
        print("提示: 未提供数据库连接（DATABASE_URL），无法检查字段是否全部为null")
        print("     如需启用此功能，请在脚本顶部设置 DATABASE_URL 配置变量")
        print("     格式示例: mysql+pymysql://user:password@host:port/database")

    # 加载 schema.json
    schema_data = load_json(schema_path)
    if not isinstance(schema_data, list):
        raise ValueError("schema.json 格式不正确，应为数组。")

    # 直接从 schema.json 构建表信息
    tables: Dict[str, Dict[str, Any]] = {}
    for schema_table in schema_data:
        if not isinstance(schema_table, Mapping):
            continue
        table_name = schema_table.get("table_name")
        if not isinstance(table_name, str):
            continue
        tables[table_name] = build_table_from_schema(schema_table)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧的输出文件
    if output_dir.exists():
        for old_file in output_dir.glob("*.txt"):
            old_file.unlink()

    total_fields = 0
    skipped_fields = 0

    # 导出每个字段为单独的 TOON 文件
    for table_name, table_data in tables.items():
        columns: List[Mapping[str, Any]] = table_data.get("columns", []) or []
        for column in columns:
            column_name = column.get("name")
            if not isinstance(column_name, str):
                continue

            # 检查是否应该跳过该字段（如果启用了跳过全null字段的功能）
            if skip_all_null and should_skip_field(column, engine, table_name):
                skipped_fields += 1
                print(f"跳过字段（全部为null）: {table_name}.{column_name}")
                continue

            # 构建字段条目
            field_entry: Dict[str, Any] = {"table_name": table_name}
            field_entry.update(column)

            # 生成安全的文件名
            safe_table = sanitize_filename(table_name)
            safe_column = sanitize_filename(column_name)
            output_file = output_dir / f"{safe_table}__{safe_column}.txt"

            # 写入 TOON 格式文件
            write_field_to_toon(field_entry, output_file, indent=indent, delimiter=delimiter)
            total_fields += 1

    if skipped_fields > 0:
        print(f"已跳过 {skipped_fields} 个全部为null的字段")
    
    return total_fields


def main() -> None:
    """主函数"""
    total_fields = generate_schema_and_export(
        schema_path=SCHEMA_PATH,
        output_dir=OUTPUT_DIR,
        indent=TOON_INDENT,
        delimiter=TOON_DELIMITER,
        database_url=DATABASE_URL,
        skip_all_null=SKIP_ALL_NULL
    )
    print(f"已生成 {total_fields} 个字段的 TOON 文本，输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

