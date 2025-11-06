"""批量处理 Text2SQL 任务的脚本，支持用户认证和结果导出。

主要步骤：
1. 从指定提示词目录加载纯文本提示词
2. 调用 workflow 服务执行任务
3. 打印提示词与结果，统计成功率并导出 JSON 文件
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)


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

# 提示词版本配置（与输入目录名称保持一致）
PROMPT_VERSION = "V1"

# 提示词文件夹路径
PROMPTS_INPUT_DIR = BASE_DIR / "script" / "prompt" / "input" / PROMPT_VERSION

# 输出文件路径配置
OUTPUT_PATH = BASE_DIR / "upload" / "dataset_exe_result.json"

# API 服务配置
API_URL = "http://localhost:6080/workflows/run"
TOKEN_URL = "http://localhost:6080/auth/token"

# 认证和工作流配置
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "client"
WORKFLOW_NAME = "reflection"
NAMESPACE = "game"

# API 请求超时时间（秒）
AUTH_TIMEOUT = 40
WORKFLOW_TIMEOUT = 300

# 缓存（全局状态）
_ACCESS_TOKEN: str | None = None

# 磁盘读取缓存与跳过列表
_PROMPT_CONTENT_CACHE: Dict[Path, str] = {}
_SKIPPED_PROMPT_IDS: List[str] = []


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TaskRequest:
    """发送给 workflow 服务的请求体。"""
    workflow: str
    namespace: str
    task: Dict[str, Any]  # 包含 prompt 等上下文信息
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
    skipped: int = 0


# ============================================================================
# 工具函数
# ============================================================================

def print_block(title: str, content: str) -> None:
    """打印格式化的信息块。
    
    Args:
        title: 信息块的标题
        content: 信息块的内容
    """
    colored_title = color_text(title, color=Fore.CYAN, style=Style.BRIGHT)
    processed_content = content.replace("\\n", "\n")
    print(f"\n[{colored_title}]")
    print(color_text(processed_content, color=Fore.WHITE))


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


# ============================================================================
# 资源加载
# ============================================================================

def load_prompt_files() -> List[Path]:
    """加载所有提示词文件路径。
    
    Returns:
        提示词文件路径列表
        
    Raises:
        FileNotFoundError: 如果提示词目录不存在
    """
    if not PROMPTS_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"提示词目录不存在: {PROMPTS_INPUT_DIR}\n"
            f"请先准备提示词文件"
        )
    
    # 获取所有 .txt 文件（排除 metadata.json）
    prompt_files = sorted(PROMPTS_INPUT_DIR.glob("*.txt"))
    
    if not prompt_files:
        raise FileNotFoundError(
            f"提示词目录中没有找到 .txt 文件: {PROMPTS_INPUT_DIR}\n"
            f"请先准备提示词文件"
        )
    
    return prompt_files


def load_resources() -> List[Path]:
    """从磁盘收集提示词文件列表。
    
    Returns:
        提示词文件路径列表
        
    Raises:
        FileNotFoundError: 如果提示词文件不存在
    """
    print(
        f"\n{color_text('📂 加载提示词资源...', color=Fore.CYAN, style=Style.BRIGHT)}"
    )
    print(
        color_text(
            f"  版本: {PROMPT_VERSION}",
            color=Fore.WHITE,
        )
    )
    print(
        color_text(
            f"  目录: {PROMPTS_INPUT_DIR}",
            color=Fore.WHITE,
        )
    )

    # 加载提示词文件
    prompt_files = load_prompt_files()

    total_files = len(prompt_files)
    valid_prompt_files: List[Path] = []
    skipped_ids: List[str] = []

    for prompt_file in prompt_files:
        content = load_prompt_from_file(prompt_file)
        if should_skip_prompt(content):
            skipped_ids.append(prompt_file.stem)
            continue
        valid_prompt_files.append(prompt_file)

    _SKIPPED_PROMPT_IDS.clear()
    _SKIPPED_PROMPT_IDS.extend(skipped_ids)

    print(
        color_text(
            f"✓ 有效提示词: {len(valid_prompt_files)} / {total_files}",
            color=Fore.GREEN,
            style=Style.BRIGHT,
        )
    )

    if skipped_ids:
        preview = ", ".join(skipped_ids[:10])
        if len(skipped_ids) > 10:
            preview += ", ..."
        print(
            color_text(
                f"⚠ 已跳过 {len(skipped_ids)} 个 golden_sql=true 的样本: {preview}",
                color=Fore.YELLOW,
                style=Style.BRIGHT,
            )
        )

    return valid_prompt_files


# ============================================================================
# 提示词处理
# ============================================================================

def load_prompt_from_file(prompt_file: Path) -> str:
    """从文件加载提示词内容。"""
    if prompt_file in _PROMPT_CONTENT_CACHE:
        return _PROMPT_CONTENT_CACHE[prompt_file]
    content = prompt_file.read_text(encoding="utf-8")
    _PROMPT_CONTENT_CACHE[prompt_file] = content
    return content


def _contains_golden_sql_flag(data: Any) -> bool:
    """递归检测 JSON 结构中是否包含 golden_sql=true。"""
    if isinstance(data, dict):
        if data.get("golden_sql") is True:
            return True
        return any(_contains_golden_sql_flag(value) for value in data.values())
    if isinstance(data, list):
        return any(_contains_golden_sql_flag(item) for item in data)
    return False


def should_skip_prompt(content: str) -> bool:
    """判断提示词内容是否应基于 golden_sql 标记而跳过。"""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False
    return _contains_golden_sql_flag(parsed)


# ============================================================================
# API 调用
# ============================================================================

def run_text2sql_task(request: TaskRequest) -> Optional[Dict[str, Any]]:
    """调用 workflow 服务执行 Text2SQL 任务。"""
    try:
        headers = {"Authorization": f"Bearer {authenticate()}"}
        if isinstance(request.task, dict):
            task_preview = dict(request.task)
            prompt_text = task_preview.get("prompt")
            if isinstance(prompt_text, str) and len(prompt_text) > 200:
                task_preview["prompt"] = f"{prompt_text[:200]}... (truncated)"
        else:
            task_preview = request.task

        payload_preview = {
            "workflow": request.workflow,
            "namespace": request.namespace,
            "task": task_preview,
            "mode": request.mode,
        }

        print_block(
            "POST 信息",
            json.dumps(payload_preview, ensure_ascii=False, indent=2)
        )

        payload = {
            "workflow": request.workflow,
            "namespace": request.namespace,
            "task": request.task,
            "mode": request.mode,
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


# ============================================================================
# 任务处理
# ============================================================================

def process_single_task(
    prompt_file: Path,
    current_idx: int,
    total_count: int,
) -> TaskResult:
    """处理单个任务。
    
    Args:
        prompt_file: 提示词文件路径
        current_idx: 当前处理的索引（从 1 开始）
        total_count: 总任务数
        
    Returns:
        任务执行结果。如果请求超时或失败，返回包含 None 值的 TaskResult
    """
    # 加载提示词内容
    prompt_content = load_prompt_from_file(prompt_file)

    sql_id = prompt_file.stem
    
    status_line = color_text(
        f"Processing {sql_id} ({current_idx}/{total_count})",
        color=Fore.YELLOW,
        style=Style.BRIGHT,
    )
    print(f"\n{status_line}")

    print_block("模型提示词", prompt_content)

    # 构建请求对象并发送
    request = TaskRequest(
        workflow=WORKFLOW_NAME,
        namespace=NAMESPACE,
        task={
            "sql_id": sql_id,
            "prompt": prompt_content,
        },
        mode="sync",
    )
    result = run_text2sql_task(request)

    # 如果请求失败或超时，result 为 None
    if result is None:
        print_block(
            "任务结果",
            color_text(
                "超时或网络错误，将此任务记录为失败",
                color=Fore.YELLOW,
                style=Style.BRIGHT
            )
        )
        return TaskResult(sql_id=sql_id, sql=None, result=None)

    sql_text = result.get("sql")
    query_result = result.get("result")

    result_view = json.dumps(
        {
            "sql": sql_text,
            "result": query_result,
        },
        ensure_ascii=False,
        indent=2,
    )
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
    prompt_files: List[Path],
) -> tuple[List[TaskResult], ProcessingStats]:
    """批量处理所有提示词文件。
    
    Args:
        prompt_files: 提示词文件路径列表
        
    Returns:
        (任务结果列表, 处理统计信息) 的元组
    """
    results: List[TaskResult] = []
    success_count = 0
    failed_ids: List[str] = []
    
    for idx, prompt_file in enumerate(prompt_files, 1):
        task_result = process_single_task(prompt_file, idx, len(prompt_files))
        results.append(task_result)
        
        is_success = bool(task_result.sql and task_result.result is not None)
        if is_success:
            success_count += 1
        else:
            failed_ids.append(task_result.sql_id)
        
        print_task_stats(task_result, success_count, idx, len(prompt_files))

    total_count = len(prompt_files)
    failed_count = total_count - success_count
    success_rate = f"{(success_count / total_count * 100):.2f}%" if total_count > 0 else "0%"
    
    stats = ProcessingStats(
        total=total_count,
        success=success_count,
        failed=failed_count,
        success_rate=success_rate,
        failed_ids=failed_ids,
        skipped=len(_SKIPPED_PROMPT_IDS),
    )

    return results, stats


# ============================================================================
# 结果导出
# ============================================================================

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
    if stats.skipped:
        print(
            color_text(
                f"  - 跳过样本: {stats.skipped}",
                color=Fore.YELLOW,
            )
        )


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    """主程序入口。
    
    流程：
    1. 从文件加载提示词资源
    2. 批量处理所有任务
    3. 导出结果到文件
    """
    print("=" * 70)
    print(color_text("开始批量处理 Text2SQL 任务", color=Fore.MAGENTA, style=Style.BRIGHT))
    print(color_text(f"提示词版本: {PROMPT_VERSION}", color=Fore.CYAN))
    print("=" * 70)
    
    try:
        # 加载所有资源
        prompt_files = load_resources()

        # 批量处理
        print(color_text("\n🔄 开始处理任务...\n", color=Fore.CYAN))
        results, stats = batch_process(prompt_files)

        # 导出结果
        print(color_text("\n📝 导出结果...", color=Fore.CYAN))
        export_results(results, stats)
        
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
