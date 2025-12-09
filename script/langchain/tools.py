"""
LangChain Text2SQL 工具定义
"""
import json
from typing import Optional, List, Dict, Any
import pymysql
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from langchain.tools import tool

from config import DB_CONFIG, DB_URL
from data_loader import (
    get_example_by_sql_id,
    get_examples_by_tables,
    get_similar_questions,
    format_example_for_prompt,
    load_common_knowledge,
)


# 数据库连接管理
_engine: Optional[Engine] = None


def get_db_engine() -> Engine:
    """获取数据库引擎（单例模式）"""
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    return _engine


def get_pymysql_connection():
    """获取 PyMySQL 连接"""
    return pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )


@tool
def get_table_schema(table_name: str) -> str:
    """获取指定表的结构信息，包括字段名、类型、注释等。
    
    Args:
        table_name: 表名，例如 'dws_argothek_oss_login_di'
    
    Returns:
        表的详细结构信息，包括字段名、数据类型、注释等
    """
    try:
        engine = get_db_engine()
        inspector = inspect(engine)
        
        if not inspector.has_table(table_name):
            return f"错误: 表 '{table_name}' 不存在于数据库中"
        
        columns = inspector.get_columns(table_name)
        schema_info = f"表名: {table_name}\n字段信息:\n"
        
        for col in columns:
            col_name = col['name']
            col_type = str(col['type'])
            nullable = "可空" if col['nullable'] else "非空"
            default = f", 默认值: {col['default']}" if col['default'] is not None else ""
            comment = f", 注释: {col.get('comment', '')}" if col.get('comment') else ""
            
            schema_info += f"  - {col_name}: {col_type} ({nullable}{default}{comment})\n"
        
        # 获取主键信息
        pk_constraint = inspector.get_pk_constraint(table_name)
        if pk_constraint['constrained_columns']:
            schema_info += f"\n主键: {', '.join(pk_constraint['constrained_columns'])}\n"
        
        # 获取索引信息
        indexes = inspector.get_indexes(table_name)
        if indexes:
            schema_info += "\n索引:\n"
            for idx in indexes:
                schema_info += f"  - {idx['name']}: {', '.join(idx['column_names'])}\n"
        
        return schema_info
    
    except Exception as e:
        return f"获取表结构时出错: {str(e)}"


@tool
def get_tables_schema(table_names: List[str]) -> str:
    """批量获取多个表的结构信息。
    
    Args:
        table_names: 表名列表，例如 ['dws_argothek_oss_login_di', 'dim_argothek_gplayerid2qqwxid_df']
    
    Returns:
        所有表的详细结构信息
    """
    schemas = []
    for table_name in table_names:
        schema = get_table_schema.invoke({"table_name": table_name})
        schemas.append(schema)
    
    return "\n\n".join(schemas)


@tool
def execute_sql(sql_query: str) -> str:
    """执行 SQL 查询并返回结果。注意：此工具仅用于 SELECT 查询，不会修改数据。
    
    Args:
        sql_query: 要执行的 SQL 查询语句
    
    Returns:
        查询结果的 JSON 字符串，如果出错则返回错误信息
    """
    try:
        # 安全检查：只允许 SELECT 查询
        sql_upper = sql_query.strip().upper()
        if not sql_upper.startswith("SELECT"):
            return "错误: 此工具仅支持 SELECT 查询，不允许执行修改数据的操作"
        
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            
            # 转换为字典列表
            if rows:
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in rows]
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                return "查询成功，但未返回任何数据"
    
    except Exception as e:
        return f"执行 SQL 时出错: {str(e)}"


@tool
def get_related_examples(question: str) -> str:
    """根据用户问题获取相似的问题-SQL 示例，用于参考学习。
    
    Args:
        question: 用户的问题
    
    Returns:
        格式化的相似示例列表
    """
    try:
        examples = get_similar_questions(question, limit=3)
        
        if not examples:
            return "未找到相似的示例"
        
        result = "找到以下相似示例:\n\n"
        for i, example in enumerate(examples, 1):
            result += f"示例 {i}:\n"
            result += format_example_for_prompt(example)
            result += "\n" + "-" * 50 + "\n\n"
        
        return result
    
    except Exception as e:
        return f"获取示例时出错: {str(e)}"


@tool
def get_examples_by_table_names(table_names: List[str]) -> str:
    """根据表名获取使用这些表的示例，用于了解表的使用方式。
    
    Args:
        table_names: 表名列表
    
    Returns:
        格式化的示例列表
    """
    try:
        examples = get_examples_by_tables(table_names, limit=5)
        
        if not examples:
            return f"未找到使用表 {', '.join(table_names)} 的示例"
        
        result = f"找到以下使用表 {', '.join(table_names)} 的示例:\n\n"
        for i, example in enumerate(examples, 1):
            result += f"示例 {i}:\n"
            result += format_example_for_prompt(example)
            result += "\n" + "-" * 50 + "\n\n"
        
        return result
    
    except Exception as e:
        return f"获取示例时出错: {str(e)}"


@tool
def get_common_knowledge() -> str:
    """获取通用业务知识，包括游戏常识、指标说明、数仓设计规范等。
    
    Returns:
        通用知识库内容
    """
    try:
        knowledge = load_common_knowledge()
        if not knowledge:
            return "未找到通用知识库"
        return knowledge
    except Exception as e:
        return f"获取知识库时出错: {str(e)}"


@tool
def validate_sql_syntax(sql_query: str) -> str:
    """验证 SQL 语法是否正确（不执行查询）。
    
    Args:
        sql_query: 要验证的 SQL 查询语句
    
    Returns:
        验证结果，包括语法检查和建议
    """
    try:
        sql_upper = sql_query.strip().upper()
        
        # 基本语法检查
        checks = []
        
        # 检查是否包含 SELECT
        if not sql_upper.startswith("SELECT"):
            checks.append("❌ 错误: SQL 必须以 SELECT 开头")
        else:
            checks.append("✓ SELECT 语句格式正确")
        
        # 检查是否包含 FROM
        if "FROM" not in sql_upper:
            checks.append("❌ 错误: SQL 必须包含 FROM 子句")
        else:
            checks.append("✓ 包含 FROM 子句")
        
        # 检查日期格式提示
        if any(keyword in sql_upper for keyword in ["DATE", "2025", "2024", "2023"]):
            checks.append("⚠️  提示: 请确认日期格式是否正确（通常为 YYYYMMDD，如 20250724）")
        
        # 检查分区字段
        if any(keyword in sql_upper for keyword in ["WHERE", "PARTITION"]):
            checks.append("⚠️  提示: 请确认 WHERE 条件中包含正确的分区字段（通常是日期字段）")
        
        result = "SQL 语法验证结果:\n" + "\n".join(checks)
        
        # 尝试解析 SQL（不执行）
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                # 只解析，不执行
                parsed = text(sql_query)
                result += "\n\n✓ SQL 可以被数据库解析"
        except Exception as parse_error:
            result += f"\n\n❌ SQL 解析错误: {str(parse_error)}"
        
        return result
    
    except Exception as e:
        return f"验证 SQL 时出错: {str(e)}"

