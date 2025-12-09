from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from transformers import AutoTokenizer

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

# Token 配置
MAX_TOKENS_PER_FILE = 3000  # 每个文件的最大 token 数
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"  # Qwen3 分词器模型名称

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
        
        columns.append(column)
    
    # 统一所有字段的键集合，确保格式一致（使用表格格式）
    # 收集所有可能的键，并确定标准键集合
    all_keys = set()
    for col in columns:
        all_keys.update(col.keys())
    
    # 定义标准键集合（按顺序）
    standard_keys = ["name", "description", "type", "type_original", "primary_key", "nullable", "autoincrement"]
    # 只保留实际存在的键
    standard_keys = [k for k in standard_keys if k in all_keys]
    # 添加其他可能存在的键
    for key in sorted(all_keys):
        if key not in standard_keys:
            standard_keys.append(key)
    
    # 确保所有字段都有相同的键集合
    unified_columns: List[Dict[str, Any]] = []
    for col in columns:
        unified_col: Dict[str, Any] = {}
        for key in standard_keys:
            if key in col:
                val = col[key]
                # 保留值，即使是 None 或空字符串
                unified_col[key] = val
            else:
                # 对于缺失的键，根据键的类型设置默认值
                if key == "description":
                    unified_col[key] = ""  # 空字符串
                elif key == "type_original":
                    unified_col[key] = None  # type_original 可以为 None
                elif key in ["primary_key", "nullable", "autoincrement"]:
                    # 这些键应该已经有默认值，但为了安全起见
                    unified_col[key] = col.get(key, False if key == "primary_key" or key == "autoincrement" else True)
                else:
                    unified_col[key] = col.get(key, None)
        unified_columns.append(unified_col)
    
    # 构建表信息，确保顺序：table_name, description, columns
    table: Dict[str, Any] = {
        "table_name": table_name,
    }
    
    # 添加 description（如果存在），放在 table_name 之后
    if table_description and isinstance(table_description, str):
        desc = table_description.strip()
        if desc:
            table["description"] = desc
    
    # 最后添加 columns
    table["columns"] = unified_columns
    
    # 对于 columns 数组，我们需要保留所有键以保持表格格式
    # 但可以清理表级别的空值
    pruned_table: Dict[str, Any] = {}
    for key, val in table.items():
        if key == "columns":
            # 保留 columns 数组，不进行深度清理（以保持键的一致性）
            pruned_table[key] = val
        else:
            # 对于其他字段，可以清理空值
            pruned_val = prune_empty(val)
            if pruned_val is not None:
                if not (isinstance(pruned_val, (list, dict)) and not pruned_val):
                    pruned_table[key] = pruned_val
    
    return pruned_table


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


def prune_empty_but_keep_strings(value: Any) -> Any:
    """递归清理空值，但保留空字符串（用于保持键的一致性）"""
    if isinstance(value, dict):
        pruned_dict: Dict[str, Any] = {}
        for key, val in value.items():
            pruned_value = prune_empty_but_keep_strings(val)
            if pruned_value is None and not isinstance(val, str):
                continue
            if isinstance(pruned_value, (list, dict)) and not pruned_value:
                continue
            pruned_dict[key] = pruned_value
        return pruned_dict

    if isinstance(value, list):
        pruned_list: List[Any] = []
        for item in value:
            pruned_item = prune_empty_but_keep_strings(item)
            if pruned_item is None and not isinstance(item, str):
                continue
            if isinstance(pruned_item, (list, dict)) and not pruned_item:
                continue
            pruned_list.append(pruned_item)
        return pruned_list

    # 保留字符串，即使是空字符串
    if isinstance(value, str):
        return value

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


def write_table_to_toon(table_data: Mapping[str, Any], output_path: Path, indent: int, delimiter: str) -> None:
    """将表数据写入 TOON 格式文件"""
    from toon_format import encode  # type: ignore

    options = {
        "indent": indent,
        "delimiter": delimiter,
    }
    encoded = encode(table_data, options)
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(encoded)


def count_tokens_with_qwen(text: str, tokenizer: Any) -> int:
    """使用 Qwen 分词器计算 token 数量"""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"警告: 计算 token 时出错: {e}，使用字符数估算")
        # 如果分词失败，使用粗略估算（1 token ≈ 4 字符）
        return len(text) // 4


def split_table_by_tokens(
    table_data: Dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    indent: int,
    delimiter: str
) -> List[Dict[str, Any]]:
    """
    将表按 token 数量切分为多个部分
    
    Args:
        table_data: 表数据
        tokenizer: Qwen 分词器
        max_tokens: 每个部分的最大 token 数
        indent: TOON 格式缩进
        delimiter: TOON 格式分隔符
    
    Returns:
        切分后的表数据列表
    """
    from toon_format import encode  # type: ignore
    
    # 先编码整个表，计算总 token 数
    options = {
        "indent": indent,
        "delimiter": delimiter,
    }
    full_encoded = encode(table_data, options)
    total_tokens = count_tokens_with_qwen(full_encoded, tokenizer)
    
    # 如果总 token 数不超过限制，直接返回
    if total_tokens <= max_tokens:
        return [table_data]
    
    # 需要切分：按字段分组
    table_name = table_data.get("table_name", "")
    table_description = table_data.get("description", "")
    columns = table_data.get("columns", [])
    
    if not columns:
        return [table_data]
    
    # 按 token 数分组，尽量让每组不超过 max_tokens
    result_tables: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = []
    
    for col in columns:
        # 尝试将当前字段添加到当前组
        test_group = current_group + [col]
        test_table = {
            "table_name": table_name,
            "columns": test_group
        }
        if table_description:
            test_table["description"] = table_description
        
        # 计算添加字段后的实际 token 数
        test_encoded = encode(test_table, options)
        test_tokens = count_tokens_with_qwen(test_encoded, tokenizer)
        
        # 如果添加字段后超过限制，且当前组不为空，先保存当前组
        if test_tokens > max_tokens and current_group:
            result_table = {
                "table_name": table_name,
                "columns": current_group
            }
            if table_description:
                result_table["description"] = table_description
            result_tables.append(result_table)
            current_group = []
            # 重新计算单独字段的 token 数
            test_group = [col]
            test_table = {
                "table_name": table_name,
                "columns": test_group
            }
            if table_description:
                test_table["description"] = table_description
            test_encoded = encode(test_table, options)
            test_tokens = count_tokens_with_qwen(test_encoded, tokenizer)
        
        # 如果单个字段（含表头）就超过限制，仍然包含它（避免无限循环）
        if test_tokens > max_tokens:
            print(f"警告: 表 {table_name} 的字段 {col.get('name', 'unknown')} 单独编码后 token 数 ({test_tokens}) 超过限制 ({max_tokens})，仍将包含该字段")
        
        # 添加字段到当前组
        current_group.append(col)
    
    # 保存最后一组
    if current_group:
        result_table = {
            "table_name": table_name,
            "columns": current_group
        }
        if table_description:
            result_table["description"] = table_description
        result_tables.append(result_table)
    
    return result_tables if result_tables else [table_data]


def generate_schema_and_export(
    schema_path: Path,
    output_dir: Path,
    indent: int,
    delimiter: str,
    database_url: Optional[str] = None,
    skip_all_null: bool = True,
    max_tokens: int = 3000,
    qwen_model_name: str = "Qwen/Qwen3-0.6B"
) -> int:
    """
    从 schema.json 生成数据结构并导出为表切片的 TOON 文件
    
    Args:
        schema_path: schema.json 文件路径
        output_dir: 输出目录
        indent: TOON 格式缩进
        delimiter: TOON 格式分隔符
        database_url: 数据库连接URL（可选，格式如：mysql+pymysql://user:password@host:port/database）
        skip_all_null: 是否跳过全部为null的字段
        max_tokens: 每个文件的最大 token 数
        qwen_model_name: Qwen 分词器模型名称
    """
    ensure_toon_format()

    # 初始化 Qwen 分词器
    try:
        print(f"正在加载 Qwen 分词器: {qwen_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        print("Qwen 分词器加载成功")
    except Exception as e:
        raise RuntimeError(f"无法加载 Qwen 分词器: {e}") from e

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

    total_files = 0
    skipped_fields = 0

    # 导出每个表为 TOON 文件（如果超过 token 限制则切分）
    for table_name, table_data in tables.items():
        columns: List[Mapping[str, Any]] = table_data.get("columns", []) or []
        
        # 过滤掉全部为null的字段（如果启用了此功能）
        filtered_columns: List[Mapping[str, Any]] = []
        for column in columns:
            column_name = column.get("name")
            if not isinstance(column_name, str):
                continue
            
            # 检查是否应该跳过该字段
            if skip_all_null and should_skip_field(column, engine, table_name):
                skipped_fields += 1
                print(f"跳过字段（全部为null）: {table_name}.{column_name}")
                continue
            
            filtered_columns.append(column)
        
        # 如果没有有效字段，跳过该表
        if not filtered_columns:
            print(f"跳过表（无有效字段）: {table_name}")
            continue
        
        # 更新表的列信息
        table_data_with_filtered = table_data.copy()
        table_data_with_filtered["columns"] = filtered_columns
        
        # 按 token 数量切分表
        split_tables = split_table_by_tokens(
            table_data_with_filtered,
            tokenizer,
            max_tokens,
            indent,
            delimiter
        )
        
        # 为每个切分后的表生成文件
        safe_table = sanitize_filename(table_name)
        for idx, split_table in enumerate(split_tables):
            if len(split_tables) > 1:
                # 如果表被切分了，添加序号后缀
                output_file = output_dir / f"{safe_table}_{idx + 1}.txt"
            else:
                # 如果表没有被切分，使用原表名
                output_file = output_dir / f"{safe_table}.txt"
            
            # 写入 TOON 格式文件
            write_table_to_toon(split_table, output_file, indent=indent, delimiter=delimiter)
            total_files += 1
            
            # 计算并显示 token 数（用于调试）
            from toon_format import encode  # type: ignore
            options = {"indent": indent, "delimiter": delimiter}
            encoded = encode(split_table, options)
            token_count = count_tokens_with_qwen(encoded, tokenizer)
            print(f"已生成: {output_file.name} (token数: {token_count})")

    if skipped_fields > 0:
        print(f"已跳过 {skipped_fields} 个全部为null的字段")
    
    return total_files


def main() -> None:
    """主函数"""
    total_files = generate_schema_and_export(
        schema_path=SCHEMA_PATH,
        output_dir=OUTPUT_DIR,
        indent=TOON_INDENT,
        delimiter=TOON_DELIMITER,
        database_url=DATABASE_URL,
        skip_all_null=SKIP_ALL_NULL,
        max_tokens=MAX_TOKENS_PER_FILE,
        qwen_model_name=QWEN_MODEL_NAME
    )
    print(f"已生成 {total_files} 个表的 TOON 文本文件，输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

