"""改进的 Datus 脚本 - 集成 Agent Lightning 优化能力

此版本在原 run_datus.py 的基础上增加了 Agent Lightning 集成。
主要改动：
1. 导入 Agent Lightning 包装器
2. 在 process_single_task 中添加追踪包装
3. 优化完成后保存追踪数据和生成分析报告
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from toon import encode as toon_encode

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# ✨ Agent Lightning 集成导入
from agent_lightning_wrapper import DatusAgentOptimizer, color_text


def color_text(text: str, *, color: Optional[str] = None, style: Optional[str] = None) -> str:
    """Apply ANSI coloring when colorama is available."""
    segments: List[str] = []
    if color:
        segments.append(color)
    if style:
        segments.append(style)
    segments.append(text)
    segments.append(Style.RESET_ALL)
    return "".join(segments)


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
DATABASE_VERSION = "4.0.0"
DATABASE_TYPE = "StarRocks"

# API 请求超时时间（秒）
AUTH_TIMEOUT = 40
WORKFLOW_TIMEOUT = 300

# ✨ Agent Lightning 配置
ENABLE_AGENT_LIGHTNING = True  # 是否启用 Agent Lightning
LIGHTNING_STORE_PATH = BASE_DIR / "lightning_store"

# 缓存（全局状态）
_ACCESS_TOKEN: str | None = None
_COMMON_KNOWLEDGE_CACHE: str | None = None
_SCHEMA_CACHE: Dict[str, TableSchema] | None = None
_SELECTED_GOLDEN_IDS: List[str] = []

# ✨ Agent Lightning 全局实例
_AGENT_OPTIMIZER: DatusAgentOptimizer | None = None


# ============================================================================
# 数据类定义 - 统一的内部数据格式
# ============================================================================

@dataclass
class DatabaseInfo:
    """数据库信息。"""
    database_type: str
    database_version: str


@dataclass
class ColumnSchema:
    """列定义。"""
    col: str
    type: str
    description: str = ""


@dataclass
class TableSchema:
    """表的 schema 定义。"""
    table_name: str
    table_description: str
    columns: List[ColumnSchema] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """知识库信息。"""
    common: str
    business: Optional[str] = None


@dataclass
class GoldenExample:
    """标准答案示例。"""
    question: str
    sql: str
    table_list: List[str]


@dataclass
class TaskPrompt:
    """单个任务的完整提示词（包含所有上下文）。"""
    question: str
    table_list: List[str]
    schema_context: List[TableSchema]
    knowledge: KnowledgeBase
    golden_examples: List[GoldenExample]
    database: DatabaseInfo
    complexity: str


@dataclass
class TaskRequest:
    """发送给 workflow 服务的请求体。"""
    workflow: str
    namespace: str
    task: TaskPrompt
    mode: str = "sync"


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


@dataclass
class Resources:
    """应用启动时加载的资源。"""
    dataset: List[Dict[str, Any]]
    golden_examples: List[GoldenExample]


# ============================================================================
# 工具函数
# ============================================================================

def extract_golden_examples(dataset: List[Dict[str, Any]]) -> List[GoldenExample]:
    """提取带有标准答案的示例。"""
    global _SELECTED_GOLDEN_IDS
    
    candidates: List[Dict[str, Any]] = []
    for entry in dataset:
        if not entry.get("golden_sql"):
            continue

        question = (entry.get("question") or "").strip()
        sql_text = (entry.get("sql") or "").strip()
        if not question or not sql_text:
            continue

        candidates.append(
            {
                "sql_id": entry.get("sql_id", "<unknown>"),
                "question": question,
                "sql": sql_text,
                "table_list": entry.get("table_list") or [],
            }
        )

    max_examples = min(1, len(candidates))
    selected = random.sample(candidates, max_examples) if candidates else []
    
    _SELECTED_GOLDEN_IDS = [ex["sql_id"] for ex in selected]
    
    return [
        GoldenExample(
            question=ex["question"],
            sql=ex["sql"],
            table_list=ex["table_list"],
        )
        for ex in selected
    ]


def load_resources() -> Resources:
    """从磁盘加载所有必需的资源文件。"""
    global _SELECTED_GOLDEN_IDS
    
    print(
        f"\n{color_text('📂 加载数据集: ' + str(DATASET_PATH), color=Fore.CYAN, style=Style.BRIGHT)}"
    )
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    print(
        color_text(
            f"✓ 已加载 {len(dataset)} 条任务",
            color=Fore.GREEN,
            style=Style.BRIGHT,
        )
    )

    golden_examples = extract_golden_examples(dataset)
    if golden_examples:
        print(
            color_text(
                f"✓ 随机选择 {len(golden_examples)} 条标准答案示例作为 few-shot",
                color=Fore.GREEN,
            )
        )
        print(
            color_text(
                f"  选中示例 ID: {', '.join(_SELECTED_GOLDEN_IDS)}",
                color=Fore.CYAN,
            )
        )

    return Resources(dataset=dataset, golden_examples=golden_examples)


def print_block(title: str, content: str) -> None:
    """打印格式化的信息块。"""
    colored_title = color_text(title, color=Fore.CYAN, style=Style.BRIGHT)
    processed_content = content.replace("\\n", "\n")
    print(f"\n[{colored_title}]")
    print(color_text(processed_content, color=Fore.WHITE))


def get_schema_cache() -> Dict[str, TableSchema]:
    """加载并缓存数据库 schema 定义。"""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    if not SCHEMA_PATH.exists():
        _SCHEMA_CACHE = {}
        return _SCHEMA_CACHE

    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        raw_schema = json.load(handle)

    cache: Dict[str, TableSchema] = {}
    for table in raw_schema:
        table_name = table.get("table_name")
        if not table_name:
            continue
        
        columns = table.get("columns") or []
        processed_columns = [
            ColumnSchema(
                col=col.get("col"),
                type=col.get("type"),
                description=col.get("description", ""),
            )
            for col in columns
            if col.get("col")
        ]
        
        cache[table_name.lower()] = TableSchema(
            table_name=table_name,
            table_description=table.get("table_description", ""),
            columns=processed_columns,
        )

    _SCHEMA_CACHE = cache
    return _SCHEMA_CACHE


def collect_table_schemas(table_names: List[str]) -> List[TableSchema]:
    """获取指定表的 schema 信息。"""
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
    """获取 OAuth2 访问令牌。"""
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
    """获取缓存的公共知识文档。"""
    global _COMMON_KNOWLEDGE_CACHE
    if _COMMON_KNOWLEDGE_CACHE is None:
        _COMMON_KNOWLEDGE_CACHE = COMMON_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return _COMMON_KNOWLEDGE_CACHE


def build_prompt_payload(entry: Dict[str, Any], golden_examples: Optional[List[GoldenExample]] = None) -> TaskPrompt:
    """为 API 构建包含全部上下文的提示词。"""
    question = (entry.get("question") or "").strip()
    tables = entry.get("table_list") or []
    knowledge = entry.get("knowledge") or ""
    complexity = entry.get("复杂度", "未知")

    knowledge_base = KnowledgeBase(
        common=get_common_knowledge(),
        business=knowledge if knowledge else None,
    )

    schema_snippets = collect_table_schemas(tables)

    db_info = DatabaseInfo(
        database_type=DATABASE_TYPE,
        database_version=DATABASE_VERSION,
    )

    return TaskPrompt(
        question=question,
        table_list=tables,
        schema_context=schema_snippets,
        knowledge=knowledge_base,
        golden_examples=golden_examples or [],
        database=db_info,
        complexity=complexity,
    )


def run_text2sql_task(request: TaskRequest) -> Optional[Dict[str, Any]]:
    """调用 workflow 服务执行 Text2SQL 任务。"""
    try:
        headers = {"Authorization": f"Bearer {authenticate()}"}
        
        request_dict = asdict(request)
        toon_encoded_request = toon_encode(request_dict, {"indent": 2})
        
        print_block("POST 信息 (toon 编码)", toon_encoded_request)
        
        payload = {
            "workflow": toon_encode(request.workflow),
            "namespace": toon_encode(request.namespace),
            "task": toon_encode(asdict(request.task), {"indent": 2}),
            "mode": toon_encode(request.mode),
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=WORKFLOW_TIMEOUT)
        if not response.ok:
            error_preview = response.json() if response.text else response.text
            print_block("请求失败", json.dumps(error_preview, ensure_ascii=False, indent=2))
        response.raise_for_status()
        return response.json()
    
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
        error_type = type(e).__name__
        print_block(
            f"❌ 请求异常: {error_type}",
            f"错误详情：{str(e)}\n将此任务当作 null 处理并继续执行"
        )
        return None


def process_single_task(
    entry: Dict[str, Any],
    current_idx: int,
    total_count: int,
    golden_examples: Optional[List[GoldenExample]] = None,
) -> TaskResult:
    """处理单个任务（集成 Agent Lightning 追踪）。"""
    sql_id = entry.get("sql_id", "<unknown>")
    task_prompt = build_prompt_payload(entry, golden_examples)
    complexity = task_prompt.complexity
    
    status_line = color_text(
        f"Processing {sql_id} ({current_idx}/{total_count}) [复杂度: {complexity}]",
        color=Fore.YELLOW,
        style=Style.BRIGHT,
    )
    print(f"\n{status_line}")
    
    prompt_dict = asdict(task_prompt)
    prompt_str = toon_encode(prompt_dict, {"indent": 2})
    print_block("模型提示词", prompt_str)

    # ✨ 定义任务执行函数（用于 Agent Lightning 包装）
    def execute_task():
        request = TaskRequest(
            workflow=WORKFLOW_NAME,
            namespace=NAMESPACE,
            task=task_prompt,
            mode="sync",
        )
        result = run_text2sql_task(request)
        
        if result is None:
            print_block(
                "任务结果",
                color_text(
                    "超时或网络错误，将此任务记录为失败",
                    color=Fore.YELLOW,
                    style=Style.BRIGHT
                )
            )
            return {
                "sql": None,
                "result": None,
                "row_count": None,
                "sql_result": None,
                "finished": False,
                "result_path": None,
                "raw_response": None,
            }
        
        sql_text = result.get("sql") or result.get("gen_sql")
        result_path = result.get("result")
        sql_result_text = result.get("sql_result")
        row_count = result.get("row_count")
        finished = result.get("finished", False)

        preview_text = sql_result_text
        if preview_text and len(preview_text) > 500:
            preview_text = preview_text[:500] + "...(truncated)"
        
        result_view = toon_encode({
            "generated_sql": sql_text,
            "result_path": result_path,
            "row_count": row_count,
            "sql_result_preview": preview_text,
        }, {"indent": 2})
        print_block("返回结果", result_view)
        
        return {
            "sql": sql_text,
            "result": result_path,
            "row_count": row_count,
            "sql_result": sql_result_text,
            "finished": finished,
            "result_path": result_path,
            "raw_response": result,
        }

    # ✨ 使用 Agent Lightning 包装（如果启用）
    if ENABLE_AGENT_LIGHTNING and _AGENT_OPTIMIZER:
        result = _AGENT_OPTIMIZER.wrap_task_execution(
            task_fn=execute_task,
            task_id=sql_id,
            question=task_prompt.question,
            gold_sql=entry.get("golden_sql"),  # 传递金标 SQL
            complexity=complexity,
            table_list=task_prompt.table_list,
        )
    else:
        result = execute_task()

    return TaskResult(sql_id=sql_id, sql=result.get("sql"), result=result.get("result"))


def print_task_stats(task_result: TaskResult, success_count: int, current_idx: int, total_count: int) -> None:
    """打印单个任务的执行统计。"""
    is_success = bool(task_result.sql and task_result.result is not None)
    current_rate = f"{(success_count / current_idx * 100):.2f}%"
    status_word = color_text(
        "是" if is_success else "否",
        color=Fore.GREEN if is_success else Fore.RED,
        style=Style.BRIGHT,
    )
    colored_rate = color_text(
        current_rate,
        color=Fore.GREEN if is_success else Fore.YELLOW,
        style=Style.BRIGHT if is_success else None,
    )
    
    stats_lines = [
        f"本次执行是否成功: {status_word}",
        f"当前进度: {current_idx}/{total_count}",
        f"累计成功: {success_count}/{current_idx}",
        f"当前成功率: {colored_rate}",
    ]
    print_block("准确率", "\n".join(stats_lines))


def batch_process(
    dataset: List[Dict[str, Any]],
    golden_examples: Optional[List[GoldenExample]] = None,
) -> tuple[List[TaskResult], ProcessingStats]:
    """批量处理数据集中的所有任务。"""
    results: List[TaskResult] = []
    success_count = 0
    failed_ids: List[str] = []
    
    for idx, entry in enumerate(dataset, 1):
        task_result = process_single_task(entry, idx, len(dataset), golden_examples)
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
    """将处理结果导出到 JSON 文件。"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    print(
        f"\n{color_text('✓ 结果已导出到: ' + str(OUTPUT_PATH), color=Fore.GREEN, style=Style.BRIGHT)}"
    )
    print(color_text(f"  - 总数: {stats.total}", color=Fore.WHITE))
    print(color_text(f"  - 成功: {stats.success}", color=Fore.GREEN))
    print(
        color_text(
            f"  - 失败: {stats.failed}",
            color=Fore.RED if stats.failed else Fore.GREEN,
            style=Style.BRIGHT if stats.failed else None,
        )
    )
    print(
        color_text(
            f"  - 成功率: {stats.success_rate}",
            color=Fore.GREEN if stats.success == stats.total else Fore.YELLOW,
            style=Style.BRIGHT,
        )
    )


def main() -> None:
    """主程序入口（集成 Agent Lightning）。"""
    global _AGENT_OPTIMIZER
    
    print("=" * 70)
    print("开始批量处理 Text2SQL 任务")
    if ENABLE_AGENT_LIGHTNING:
        print("✨ 已启用 Agent Lightning 优化")
    print("=" * 70)
    
    try:
        # ✨ 初始化 Agent Lightning 优化器
        if ENABLE_AGENT_LIGHTNING:
            _AGENT_OPTIMIZER = DatusAgentOptimizer(
                store_path=LIGHTNING_STORE_PATH,
                enable_training=True,
            )
        
        # 加载所有资源
        resources = load_resources()

        # 批量处理
        print(color_text("\n🔄 开始处理任务...\n", color=Fore.CYAN))
        results, stats = batch_process(resources.dataset, resources.golden_examples)

        # 导出结果
        print(color_text("\n📝 导出结果...", color=Fore.CYAN))
        export_results(results, stats)
        
        # ✨ 保存追踪数据和生成分析报告
        if ENABLE_AGENT_LIGHTNING and _AGENT_OPTIMIZER:
            print(color_text("\n⚡ 保存 Agent Lightning 数据...", color=Fore.CYAN))
            _AGENT_OPTIMIZER.save_traces()
            _AGENT_OPTIMIZER.export_analysis()
            _AGENT_OPTIMIZER.print_summary()
        
        print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
        print(color_text("✓ 处理完成！", color=Fore.GREEN, style=Style.BRIGHT))
        print(color_text("=" * 70, color=Fore.MAGENTA))
        
    except FileNotFoundError as e:
        print(color_text(f"\n❌ 错误：文件未找到 - {e}", color=Fore.RED, style=Style.BRIGHT))
        raise
    except json.JSONDecodeError as e:
        print(color_text(f"\n❌ 错误：JSON 解析失败 - {e}", color=Fore.RED, style=Style.BRIGHT))
        raise
    except Exception as e:
        print(color_text(f"\n❌ 错误：{type(e).__name__} - {e}", color=Fore.RED, style=Style.BRIGHT))
        raise


if __name__ == "__main__":
    main()
