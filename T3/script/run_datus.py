"""批量处理 Text2SQL 任务的脚本，支持用户认证和结果导出。

此脚本的主要功能：
1. 从 JSON 数据集加载任务
2. 获取数据库 schema 和业务知识
3. 调用 workflow 服务生成 SQL 并执行
4. 统计并导出结果
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from toon import encode as toon_encode


# ============================================================================
# 配置常量
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# 文件路径配置
DATASET_PATH = BASE_DIR / "data" / "final_dataset.json"
COMMON_KNOWLEDGE_PATH = BASE_DIR / "data" / "common_knowledge.md"
SCHEMA_PATH = BASE_DIR / "data" / "schema.json"
OUTPUT_PATH = BASE_DIR / "upload" / "dataset_exe_result.json"

# API 服务配置
API_URL = "http://localhost:6080/workflows/run"
TOKEN_URL = "http://localhost:6080/auth/token"

# 认证和工作流配置
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "client"
WORKFLOW_NAME = "reflection"
NAMESPACE = "game"

# 数据库信息配置
CATALOG_NAME = "default_catalog"
DATABASE_NAME = "database_main"
SCHEMA_NAME = ""

# API 请求超时时间（秒）
AUTH_TIMEOUT = 30
WORKFLOW_TIMEOUT = 120

# 缓存（全局状态）
_ACCESS_TOKEN: str | None = None
_COMMON_KNOWLEDGE_CACHE: str | None = None
_SCHEMA_CACHE: Dict[str, Dict[str, Any]] | None = None


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TaskResult:
    """单个任务的执行结果。"""
    sql_id: str
    sql: Optional[str]
    result: Optional[Any]


@dataclass
class ProcessingStats:
    """批处理的统计信息。"""
    total: int
    success: int
    failed: int
    success_rate: str
    failed_ids: List[str]


# ============================================================================
# 工具函数
# ============================================================================

def print_block(title: str, content: str) -> None:
    """打印格式化的信息块。
    
    Args:
        title: 信息块的标题
        content: 信息块的内容
    """
    print(f"\n[{title}]")
    print(content)


def dump_compact_json(data: Any) -> Optional[str]:
    """将数据序列化为单行紧凑 JSON 字符串。"""
    if data is None:
        return None
    if isinstance(data, (list, dict)) and not data:
        return None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def get_schema_cache() -> Dict[str, Dict[str, Any]]:
    """加载并缓存数据库 schema 定义。
    
    将 schema JSON 文件中的表定义按表名索引，便于快速查询。
    
    Returns:
        按表名（小写）索引的 schema 字典。如果文件不存在，返回空字典。
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    if not SCHEMA_PATH.exists():
        _SCHEMA_CACHE = {}
        return _SCHEMA_CACHE

    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        raw_schema = json.load(handle)

    cache: Dict[str, Dict[str, Any]] = {}
    for table in raw_schema:
        table_name = table.get("table_name")
        if not table_name:
            continue
        
        columns = table.get("columns") or []
        processed_columns = [
            {
                "col": col.get("col"),
                "type": col.get("type"),
                "description": col.get("description", ""),
            }
            for col in columns
            if col.get("col")
        ]
        
        cache[table_name.lower()] = {
            "table_name": table_name,
            "table_description": table.get("table_description", ""),
            "columns": processed_columns,
        }

    _SCHEMA_CACHE = cache
    return _SCHEMA_CACHE


def collect_table_schemas(table_names: List[str]) -> List[Dict[str, Any]]:
    """获取指定表的 schema 信息。
    
    Args:
        table_names: 表名列表
        
    Returns:
        指定表的 schema 片段列表
    """
    if not table_names:
        return []

    schema_cache = get_schema_cache()
    collected = []

    for table_name in table_names:
        schema = schema_cache.get(table_name.lower())
        if schema:
            collected.append(schema)

    return collected


def authenticate() -> str:
    """获取 OAuth2 访问令牌。
    
    使用客户端凭证流程从认证服务获取访问令牌，并缓存以供后续使用。
    
    Returns:
        访问令牌字符串
        
    Raises:
        requests.exceptions.RequestException: 认证请求失败时抛出
    """
    global _ACCESS_TOKEN
    if _ACCESS_TOKEN:
        return _ACCESS_TOKEN

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=AUTH_TIMEOUT,
    )
    response.raise_for_status()
    token_payload = response.json()
    _ACCESS_TOKEN = token_payload.get("access_token")
    return _ACCESS_TOKEN


def get_common_knowledge() -> str:
    """获取缓存的公共知识文档。
    
    首次调用时从文件读取，之后返回缓存结果。
    
    Returns:
        公共知识文档的内容
    """
    global _COMMON_KNOWLEDGE_CACHE
    if _COMMON_KNOWLEDGE_CACHE is None:
        _COMMON_KNOWLEDGE_CACHE = COMMON_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return _COMMON_KNOWLEDGE_CACHE


def format_knowledge_blob(knowledge_blob: Dict[str, Any]) -> Optional[str]:
    """将知识字典转换为 JSON 字符串。
    
    使用单行紧凑格式（无换行符），便于 API 处理。
    
    Args:
        knowledge_blob: 包含公共和业务知识的字典
        
    Returns:
        格式化后的 JSON 字符串（单行格式），如果输入为空则返回 None
    """
    return dump_compact_json(knowledge_blob)


def build_prompt_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    """为 API 构建提示词、schema 和知识的负载。
    
    从数据集条目中提取并整理：
    - 问题/查询
    - 涉及的表名
    - 公共和业务知识
    - 复杂度（仅用于日志，不发送到 API）
    
    Args:
        entry: 数据集中的单个条目，包含 question、table_list、knowledge、复杂度 等字段
        
    Returns:
        包含 prompt、schema_payload、ext_knowledge、database_name、complexity 的字典
    """
    question = (entry.get("question") or "").strip()
    tables = entry.get("table_list") or []
    knowledge = entry.get("knowledge") or ""
    complexity = entry.get("复杂度", "未知")

    knowledge_blob: Dict[str, Any] = {"common": get_common_knowledge()}
    if knowledge:
        knowledge_blob["business"] = knowledge

    prompt_payload: Dict[str, Any] = {
        "question": question,
        "table_list": tables,
    }

    # 移除空字段
    if not question:
        prompt_payload.pop("question")
    if not tables:
        prompt_payload.pop("table_list")

    schema_snippets = collect_table_schemas(tables)
    schema_payload = dump_compact_json(schema_snippets)

    catalog_name = (entry.get("catalog_name") or CATALOG_NAME or "").strip()
    database_name = (entry.get("database_name") or DATABASE_NAME or "").strip()
    schema_name = entry.get("schema_name")
    if schema_name is None:
        schema_name = SCHEMA_NAME
    schema_name = (schema_name or "").strip()

    return {
        "prompt": prompt_payload,
        "schema_payload": schema_payload,
        "ext_knowledge": format_knowledge_blob(knowledge_blob),
        "database_name": database_name,
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "complexity": complexity,
    }


def run_text2sql_task(
    prompt: Dict[str, Any],
    schema_payload: Optional[str],
    ext_knowledge: Optional[str],
    catalog_name: Optional[str],
    database_name: Optional[str],
    schema_name: Optional[str],
) -> Dict[str, Any]:
    """调用 workflow 服务执行 Text2SQL 任务。
    
    发送提示词、schema、知识和数据库信息到服务，获取生成的 SQL 和执行结果。
    
    Args:
        prompt: 提示词字典，包含问题和表列表
    schema_payload: 格式化后的 schema JSON 字符串
    ext_knowledge: 格式化后的知识 JSON 字符串
    catalog_name: 数据库 catalog 名称
    database_name: 数据库名称
    schema_name: schema 名称
        
    Returns:
        服务返回的 JSON 响应，包含 sql 和 result 字段
        
    Raises:
        requests.exceptions.RequestException: API 请求失败时抛出
    """
    headers = {"Authorization": f"Bearer {authenticate()}"}
    
    # 内部维护 JSON 结构
    payload = {
        "workflow": WORKFLOW_NAME,
        "namespace": NAMESPACE,
        "task": prompt,
        "mode": "async",
    }
    if catalog_name:
        payload["catalog_name"] = catalog_name
    if ext_knowledge:
        payload["ext_knowledge"] = ext_knowledge
    if database_name:
        payload["database_name"] = database_name
    if schema_name:
        payload["schema_name"] = schema_name

    # 先解析字符串参数为对象
    schema_obj = json.loads(schema_payload) if schema_payload else None
    knowledge_obj = json.loads(ext_knowledge) if ext_knowledge else None

    # 打印时采用 toon 格式
    debug_view = {
        "url": API_URL,
        "workflow": payload["workflow"],
        "namespace": payload["namespace"],
        "mode": payload["mode"],
    }
    if schema_obj:
        debug_view["schema_payload"] = toon_encode(schema_obj, {"indent": 2})
    if knowledge_obj:
        debug_view["ext_knowledge"] = toon_encode(knowledge_obj, {"indent": 2})
    if catalog_name:
        debug_view["catalog_name"] = catalog_name
    if database_name:
        debug_view["database_name"] = database_name
    if schema_name:
        debug_view["schema_name"] = schema_name

    print_block("POST 信息", toon_encode(debug_view, {"indent": 2}))

    # 发送 POST 请求前将所有参数分别用 toon 压缩转换为字符串
    request_payload = {
        "workflow": toon_encode(WORKFLOW_NAME),
        "namespace": toon_encode(NAMESPACE),
        "task": toon_encode(payload["task"], {"indent": 2}),
        "mode": toon_encode("sync"),
    }
    if catalog_name:
        request_payload["catalog_name"] = toon_encode(catalog_name)
    if knowledge_obj:
        request_payload["ext_knowledge"] = toon_encode(knowledge_obj, {"indent": 2})
    if database_name:
        request_payload["database_name"] = toon_encode(database_name)
    if schema_name:
        request_payload["schema_name"] = toon_encode(schema_name)
    if schema_obj:
        request_payload["schema_payload"] = toon_encode(schema_obj, {"indent": 2})

    response = requests.post(API_URL, headers=headers, json=request_payload, timeout=WORKFLOW_TIMEOUT)
    if not response.ok:
        try:
            error_preview = response.json()
        except ValueError:
            error_preview = response.text
        print_block("请求失败", json.dumps(error_preview, ensure_ascii=False, indent=2))
    response.raise_for_status()
    return response.json()


def process_single_task(
    entry: Dict[str, Any],
    current_idx: int,
    total_count: int,
) -> TaskResult:
    """处理单个任务。
    
    Args:
        entry: 数据集条目
        current_idx: 当前处理的索引（从 1 开始）
        total_count: 总任务数
        
    Returns:
        任务执行结果
    """
    sql_id = entry.get("sql_id", "<unknown>")
    prompt_bundle = build_prompt_payload(entry)
    prompt_str = toon_encode(prompt_bundle["prompt"], {"indent": 2})
    complexity = prompt_bundle.get("complexity", "未知")
    
    print(f"\nProcessing {sql_id} ({current_idx}/{total_count}) [复杂度: {complexity}]")
    print_block("模型提示词", prompt_str)

    result = run_text2sql_task(
        prompt_bundle["prompt"],
        prompt_bundle.get("schema_payload"),
        prompt_bundle.get("ext_knowledge"),
        prompt_bundle.get("catalog_name"),
        prompt_bundle.get("database_name"),
        prompt_bundle.get("schema_name"),
    )

    sql_text = result.get("sql")
    query_result = result.get("result")

    result_view = toon_encode({
        "sql": sql_text,
        "result": query_result,
    }, {"indent": 2})
    print_block("返回结果", result_view)

    return TaskResult(sql_id=sql_id, sql=sql_text, result=query_result)


def print_task_stats(task_result: TaskResult, success_count: int, current_idx: int, total_count: int) -> None:
    """打印单个任务的执行统计。
    
    Args:
        task_result: 任务执行结果
        success_count: 到目前为止的成功次数
        current_idx: 当前处理的索引（从 1 开始）
        total_count: 总任务数
    """
    is_success = bool(task_result.sql and task_result.result is not None)
    current_rate = f"{(success_count / current_idx * 100):.2f}%"
    
    stats_lines = [
        f"本次执行是否成功: {'是' if is_success else '否'}",
        f"当前进度: {current_idx}/{total_count}",
        f"累计成功: {success_count}/{current_idx}",
        f"当前成功率: {current_rate}",
    ]
    print_block("准确率", "\n".join(stats_lines))


def batch_process(dataset: List[Dict[str, Any]]) -> tuple[List[TaskResult], ProcessingStats]:
    """批量处理数据集中的所有任务。
    
    Args:
        dataset: 包含多个任务条目的列表
        
    Returns:
        (任务结果列表, 处理统计信息) 的元组
    """
    results: List[TaskResult] = []
    success_count = 0
    failed_ids: List[str] = []
    
    for idx, entry in enumerate(dataset, 1):
        task_result = process_single_task(entry, idx, len(dataset))
        results.append(task_result)
        
        is_success = bool(task_result.sql and task_result.result is not None)
        if is_success:
            success_count += 1
        else:
            failed_ids.append(task_result.sql_id)
        
        print_task_stats(task_result, success_count, idx, len(dataset))

    total_count = len(dataset)
    failed_count = total_count - success_count
    success_rate = f"{(success_count / total_count * 100):.2f}%" if total_count > 0 else "0%"
    
    stats = ProcessingStats(
        total=total_count,
        success=success_count,
        failed=failed_count,
        success_rate=success_rate,
        failed_ids=failed_ids,
    )

    return results, stats


def export_results(results: List[TaskResult], stats: ProcessingStats) -> None:
    """将处理结果导出到 JSON 文件。
    
    Args:
        results: 任务结果列表
        stats: 处理统计信息
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化的格式
    results_dict = [
        {
            "sql_id": r.sql_id,
            "sql": r.sql,
            "result": r.result,
        }
        for r in results
    ]
    
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, ensure_ascii=False, indent=4)
    
    print(f"\n✓ 结果已导出到: {OUTPUT_PATH}")
    print(f"  - 总数: {stats.total}")
    print(f"  - 成功: {stats.success}")
    print(f"  - 失败: {stats.failed}")
    print(f"  - 成功率: {stats.success_rate}")

def main() -> None:
    """主程序入口。
    
    流程：
    1. 从文件加载数据集
    2. 批量处理所有任务
    3. 导出结果到文件
    """
    print("=" * 70)
    print("开始批量处理 Text2SQL 任务")
    print("=" * 70)
    
    try:
        # 加载数据集
        print(f"\n📂 加载数据集: {DATASET_PATH}")
        with DATASET_PATH.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        print(f"✓ 已加载 {len(dataset)} 条任务")

        # 批量处理
        print("\n🔄 开始处理任务...\n")
        results, stats = batch_process(dataset)

        # 导出结果
        print("\n📝 导出结果...")
        export_results(results, stats)
        
        print("\n" + "=" * 70)
        print("✓ 处理完成！")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误：文件未找到 - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"\n❌ 错误：JSON 解析失败 - {e}")
        raise
    except Exception as e:
        print(f"\n❌ 错误：{type(e).__name__} - {e}")
        raise


if __name__ == "__main__":
    main()
