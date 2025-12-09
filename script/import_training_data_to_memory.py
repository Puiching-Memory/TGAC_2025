"""
将正确的问题-SQL对导入到 vanna agent memory 中用于训练

该脚本会：
1. 从 ckpt 目录和 final_dataset.json 收集所有正确示例（得分为1或golden_sql）
2. 将这些示例添加到 agent_memory 中，供 agent 学习参考
"""

from __future__ import annotations

import asyncio
import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

# 导入 vanna 相关模块
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.capabilities.agent_memory import ToolMemory
from vanna.core.tool import ToolContext
from vanna.core.user.models import User

# ==== 配置常量 ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "ckpt"
FINAL_DATASET_PATH = REPO_ROOT / "data" / "final_dataset.json"
CHROMA_DB_PATH = REPO_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "vanna_memory"
# ==================


def load_json(path: Path) -> Any:
    """加载 JSON 文件"""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_question_lookup(final_dataset: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """构建 sql_id 到 question 的映射"""
    lookup: Dict[str, str] = {}
    for item in final_dataset:
        sql_id = item.get("sql_id")
        question = item.get("question")
        if isinstance(sql_id, str) and isinstance(question, str):
            lookup[sql_id] = question
    return lookup


def load_correct_ids(score_path: Path) -> set[str]:
    """加载正确示例的SQL ID（得分为1的记录）"""
    correct_ids: set[str] = set()
    with score_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            sql_id = row.get("SQL ID") or row.get("sql_id")
            score = row.get("得分") or row.get("score")
            if not isinstance(sql_id, str):
                continue
            if isinstance(score, str) and score.strip() == "1":
                correct_ids.add(sql_id.strip())
    return correct_ids


def collect_correct_from_ckpt(
    question_lookup: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """从ckpt目录收集正确示例"""
    records: List[Dict[str, Any]] = []

    for score_path in sorted(CKPT_DIR.glob("*/score.csv")):
        version = score_path.parent.name
        correct_ids = load_correct_ids(score_path)
        dataset_path = score_path.parent / "dataset_exe_result.json"
        if not dataset_path.exists():
            continue

        dataset_entries = load_json(dataset_path)
        if not isinstance(dataset_entries, list):
            continue

        for entry in dataset_entries:
            if not isinstance(entry, Mapping):
                continue
            sql_id = entry.get("sql_id")
            if sql_id not in correct_ids:
                continue
            sql = entry.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                continue

            question = question_lookup.get(sql_id)
            if not question:
                continue

            records.append({
                "sql_id": sql_id,
                "question": question,
                "sql": sql,
                "source": f"ckpt/{version}",
            })

    return records


def collect_golden_from_final_dataset(
    final_dataset: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """从final_dataset收集golden SQL示例"""
    records: List[Dict[str, Any]] = []
    for item in final_dataset:
        if not isinstance(item, Mapping):
            continue
        if not item.get("golden_sql"):
            continue
        sql_id = item.get("sql_id")
        sql = item.get("sql")
        question = item.get("question")
        if not isinstance(sql_id, str) or not isinstance(sql, str):
            continue
        if not isinstance(question, str):
            continue

        records.append({
            "sql_id": sql_id,
            "question": question,
            "sql": sql,
            "source": "final_dataset",
        })
    return records


def merge_duplicates(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并重复记录，保留第一个"""
    unique: Dict[str, Dict[str, Any]] = {}
    for record in records:
        # 使用 sql_id 作为唯一键
        sql_id = record.get("sql_id")
        if not sql_id:
            continue
        if sql_id not in unique:
            unique[sql_id] = record
        else:
            # 合并来源信息
            existing = unique[sql_id]
            existing_source = existing.get("source", "")
            new_source = record.get("source", "")
            if new_source and new_source not in existing_source:
                existing["source"] = f"{existing_source}, {new_source}"
    return list(unique.values())


async def import_to_memory(
    records: List[Dict[str, Any]],
    agent_memory: ChromaAgentMemory,
    batch_size: int = 10,
) -> None:
    """将记录导入到 agent memory"""
    # 创建用户上下文
    user = User(
        id="admin",
        email="admin@example.com",
        group_memberships=["admin"]
    )
    # 创建 ToolContext，需要提供所有必需字段
    context = ToolContext(
        user=user,
        conversation_id="training_import",  # 训练导入的会话ID
        request_id=str(uuid.uuid4()),  # 生成唯一的请求ID
        agent_memory=agent_memory,  # 传入 agent_memory
    )

    total = len(records)
    success_count = 0
    error_count = 0

    print(f"开始导入 {total} 个训练示例...")

    for i, record in enumerate(records, 1):
        try:
            question = record["question"]
            sql = record["sql"]
            sql_id = record.get("sql_id", "")
            source = record.get("source", "")
            metadata = {
                "sql_id": sql_id,
                "source": source,
            }

            # 创建 ToolMemory 对象
            tool_memory = ToolMemory(
                question=question,
                tool_name="run_sql",
                args={"sql": sql},
            )

            # 打印完整的导入 context
            print(f"\n[{i}/{total}] 导入记录:")
            print(f"  SQL ID: {sql_id}")
            print(f"  来源: {source}")
            print(f"  问题: {tool_memory.question}")
            print(f"  SQL: {tool_memory.args.get('sql', '')}")
            print(f"  工具名称: {tool_memory.tool_name}")
            print(f"  参数: {json.dumps(tool_memory.args, ensure_ascii=False, indent=2)}")
            print(f"  元数据: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
            print(f"  上下文用户: id={context.user.id}, email={context.user.email}, groups={context.user.group_memberships}")
            print(f"  成功标记: True")

            # 使用 ToolMemory 调用 save_tool_usage
            await agent_memory.save_tool_usage(
                question=tool_memory.question,
                tool_name=tool_memory.tool_name,
                args=tool_memory.args,
                context=context,
                success=True,  # 标记为成功
                metadata=metadata,
            )
            success_count += 1

            if i % batch_size == 0:
                print(f"\n已导入 {i}/{total} 个示例...")

        except Exception as e:
            error_count += 1
            print(f"导入失败 (sql_id={record.get('sql_id', 'unknown')}): {e}")

    print(f"\n导入完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  总计: {total}")


async def main() -> None:
    """主函数"""
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"未找到 final_dataset.json: {FINAL_DATASET_PATH}")

    print("加载数据集...")
    final_dataset = load_json(FINAL_DATASET_PATH)
    if not isinstance(final_dataset, list):
        raise ValueError("final_dataset.json 格式错误，应为数组。")

    # 构建问题查找表
    question_lookup = build_question_lookup(final_dataset)

    # 收集正确示例
    print("收集正确示例...")
    ckpt_records = collect_correct_from_ckpt(question_lookup)
    golden_records = collect_golden_from_final_dataset(final_dataset)

    # 合并去重
    all_records = merge_duplicates(ckpt_records + golden_records)
    print(f"共收集到 {len(all_records)} 个唯一示例")

    if not all_records:
        print("没有找到可导入的示例")
        return

    # 初始化 agent memory
    print(f"初始化 ChromaDB (路径: {CHROMA_DB_PATH})...")
    agent_memory = ChromaAgentMemory(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH)
    )

    # 导入到 memory
    await import_to_memory(all_records, agent_memory)

    print(f"\n训练数据已成功导入到 agent memory!")
    print(f"ChromaDB 路径: {CHROMA_DB_PATH}")
    print(f"Collection 名称: {CHROMA_COLLECTION_NAME}")


if __name__ == "__main__":
    asyncio.run(main())

