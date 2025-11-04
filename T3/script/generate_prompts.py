"""预处理脚本：生成带版本标识的提示词文件。

此脚本的主要功能：
1. 从 JSON 数据集加载任务
2. 为每个任务生成完整的提示词
3. 将提示词保存为独立的 txt 文件
4. 支持版本标识，便于管理和追踪
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from toon import encode as toon_encode
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

# 文件路径配置
DATASET_PATH = BASE_DIR / "data" / "final_dataset.json"
COMMON_KNOWLEDGE_PATH = BASE_DIR / "data" / "common_knowledge.md"
SCHEMA_PATH = BASE_DIR / "data" / "schema.json"

# 提示词版本配置
PROMPT_VERSION = "v1.0.0"  # 修改此版本号来标识不同的提示词格式

# 输出目录配置
PROMPTS_OUTPUT_DIR = BASE_DIR / "prompts" / PROMPT_VERSION

# 数据库信息配置
DATABASE_VERSION = "4.0.0"
DATABASE_TYPE = "StarRocks"

# 缓存（全局状态）
_COMMON_KNOWLEDGE_CACHE: str | None = None
_SCHEMA_CACHE: Dict[str, TableSchema] | None = None
_SELECTED_GOLDEN_IDS: List[str] = []


# ============================================================================
# 数据类定义
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
    version: str  # 提示词版本
    generated_at: str  # 生成时间
    sql_id: str  # 任务 ID
    question: str
    table_list: List[str]
    schema_context: List[TableSchema]
    knowledge: KnowledgeBase
    golden_examples: List[GoldenExample]
    database: DatabaseInfo
    complexity: str


@dataclass
class PromptMetadata:
    """提示词元数据。"""
    version: str
    generated_at: str
    total_prompts: int
    golden_example_ids: List[str]
    database_type: str
    database_version: str


# ============================================================================
# 工具函数
# ============================================================================

def get_common_knowledge() -> str:
    """获取缓存的公共知识文档。"""
    global _COMMON_KNOWLEDGE_CACHE
    if _COMMON_KNOWLEDGE_CACHE is None:
        _COMMON_KNOWLEDGE_CACHE = COMMON_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return _COMMON_KNOWLEDGE_CACHE


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


def extract_golden_examples(dataset: List[Dict[str, Any]]) -> List[GoldenExample]:
    """提取带有标准答案的示例，随机选择最多 1 条作为 few-shot 提示。"""
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

    # 随机选择最多 1 条
    max_examples = min(1, len(candidates))
    selected = random.sample(candidates, max_examples) if candidates else []
    
    # 记录选中的 ID
    _SELECTED_GOLDEN_IDS = [ex["sql_id"] for ex in selected]
    
    # 转换为 GoldenExample 对象
    return [
        GoldenExample(
            question=ex["question"],
            sql=ex["sql"],
            table_list=ex["table_list"],
        )
        for ex in selected
    ]


def build_task_prompt(
    entry: Dict[str, Any],
    golden_examples: List[GoldenExample],
    version: str,
    timestamp: str,
) -> TaskPrompt:
    """为单个任务构建完整的提示词。"""
    sql_id = entry.get("sql_id", "<unknown>")
    question = (entry.get("question") or "").strip()
    tables = entry.get("table_list") or []
    knowledge = entry.get("knowledge") or ""
    complexity = entry.get("复杂度", "未知")

    # 构建知识库对象
    knowledge_base = KnowledgeBase(
        common=get_common_knowledge(),
        business=knowledge if knowledge else None,
    )

    # 收集 schema 信息
    schema_snippets = collect_table_schemas(tables)

    # 构建数据库信息对象
    db_info = DatabaseInfo(
        database_type=DATABASE_TYPE,
        database_version=DATABASE_VERSION,
    )

    return TaskPrompt(
        version=version,
        generated_at=timestamp,
        sql_id=sql_id,
        question=question,
        table_list=tables,
        schema_context=schema_snippets,
        knowledge=knowledge_base,
        golden_examples=golden_examples,
        database=db_info,
        complexity=complexity,
    )


def save_prompt_to_file(prompt: TaskPrompt, output_dir: Path) -> None:
    """将提示词保存为 toon 编码的 txt 文件。
    
    注意：提示词在生成阶段就使用 toon 编码，执行阶段只需要解码即可。
    这样可以确保提示词的一致性，避免执行时重复编码。
    """
    # 使用 toon 编码格式化提示词
    prompt_dict = asdict(prompt)
    prompt_str = toon_encode(prompt_dict, {"indent": 2})
    
    # 文件名使用 sql_id
    filename = f"{prompt.sql_id}.txt"
    filepath = output_dir / filename
    
    filepath.write_text(prompt_str, encoding="utf-8")


def generate_all_prompts(dataset: List[Dict[str, Any]], golden_examples: List[GoldenExample]) -> int:
    """为数据集中的所有任务生成提示词文件。"""
    # 创建输出目录
    PROMPTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().isoformat()
    
    print(
        color_text(
            f"\n📝 开始生成提示词...",
            color=Fore.CYAN,
            style=Style.BRIGHT,
        )
    )
    print(
        color_text(
            f"  版本: {PROMPT_VERSION}",
            color=Fore.WHITE,
        )
    )
    print(
        color_text(
            f"  输出目录: {PROMPTS_OUTPUT_DIR}",
            color=Fore.WHITE,
        )
    )
    
    # 为每个任务生成提示词
    for idx, entry in enumerate(dataset, 1):
        sql_id = entry.get("sql_id", f"<unknown_{idx}>")
        prompt = build_task_prompt(entry, golden_examples, PROMPT_VERSION, timestamp)
        save_prompt_to_file(prompt, PROMPTS_OUTPUT_DIR)
        
        if idx % 10 == 0 or idx == len(dataset):
            print(
                color_text(
                    f"  进度: {idx}/{len(dataset)}",
                    color=Fore.YELLOW,
                )
            )
    
    return len(dataset)


def save_metadata(total_prompts: int) -> None:
    """保存提示词生成的元数据。"""
    metadata = PromptMetadata(
        version=PROMPT_VERSION,
        generated_at=datetime.now().isoformat(),
        total_prompts=total_prompts,
        golden_example_ids=_SELECTED_GOLDEN_IDS,
        database_type=DATABASE_TYPE,
        database_version=DATABASE_VERSION,
    )
    
    metadata_path = PROMPTS_OUTPUT_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, ensure_ascii=False, indent=4)
    
    print(
        color_text(
            f"\n✓ 元数据已保存到: {metadata_path}",
            color=Fore.GREEN,
        )
    )


def main() -> None:
    """主程序入口。"""
    print("=" * 70)
    print(color_text("提示词预处理脚本", color=Fore.MAGENTA, style=Style.BRIGHT))
    print(color_text(f"版本: {PROMPT_VERSION}", color=Fore.CYAN))
    print("=" * 70)
    
    try:
        # 加载数据集
        print(
            color_text(
                f"\n📂 加载数据集: {DATASET_PATH}",
                color=Fore.CYAN,
                style=Style.BRIGHT,
            )
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

        # 提取标准答案示例
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

        # 生成所有提示词
        total_prompts = generate_all_prompts(dataset, golden_examples)

        # 保存元数据
        save_metadata(total_prompts)

        print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
        print(
            color_text(
                f"✓ 完成！共生成 {total_prompts} 个提示词文件",
                color=Fore.GREEN,
                style=Style.BRIGHT,
            )
        )
        print(color_text("=" * 70, color=Fore.MAGENTA))

    except FileNotFoundError as e:
        print(
            color_text(
                f"\n❌ 错误：文件未找到 - {e}",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
        raise
    except json.JSONDecodeError as e:
        print(
            color_text(
                f"\n❌ 错误：JSON 解析失败 - {e}",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
        raise
    except Exception as e:
        print(
            color_text(
                f"\n❌ 错误：{type(e).__name__} - {e}",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
        raise


if __name__ == "__main__":
    main()
