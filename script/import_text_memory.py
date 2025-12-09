"""
将文本知识文档导入到 vanna agent memory 中

该脚本会：
1. 读取 common_knowledge.md 和 补充知识.md
2. 将这些文档内容添加到 agent_memory 中作为文本记忆
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import List

# 导入 vanna 相关模块
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.tool import ToolContext
from vanna.core.user.models import User

# ==== 配置常量 ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DB_PATH = REPO_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "vanna_memory"

# 要导入的文档文件
DOCUMENT_FILES = [
    REPO_ROOT / "data" / "common_knowledge.md",
    # REPO_ROOT / "data" / "补充知识.md",
]
# ==================


def load_document(file_path: Path) -> str:
    """加载文档文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文档文件: {file_path}")
    
    with file_path.open("r", encoding="utf-8") as fp:
        return fp.read()


async def import_text_memory(
    agent_memory: ChromaAgentMemory,
    document_files: List[Path],
) -> None:
    """将文档导入到 agent memory 作为文本记忆"""
    # 创建用户上下文
    user = User(
        id="admin",
        email="admin@example.com",
        group_memberships=["admin"]
    )
    
    total = len(document_files)
    success_count = 0
    error_count = 0

    print(f"开始导入 {total} 个文档...")

    for i, doc_path in enumerate(document_files, 1):
        try:
            # 读取文档内容
            content = load_document(doc_path)
            
            # 创建 ToolContext，需要提供所有必需字段
            context = ToolContext(
                user=user,
                conversation_id="text_memory_import",  # 文本记忆导入的会话ID
                request_id=str(uuid.uuid4()),  # 生成唯一的请求ID
                agent_memory=agent_memory,  # 传入 agent_memory
            )

            # 打印导入信息
            print(f"\n[{i}/{total}] 导入文档:")
            print(f"  文件: {doc_path.name}")
            print(f"  内容长度: {len(content)} 字符")
            print(f"  上下文用户: id={context.user.id}, email={context.user.email}, groups={context.user.group_memberships}")

            # 调用 save_text_memory
            await agent_memory.save_text_memory(
                content=content,
                context=context,
            )
            
            success_count += 1
            print(f"  ✓ 导入成功")

        except Exception as e:
            error_count += 1
            print(f"  ✗ 导入失败 ({doc_path.name}): {e}")

    print(f"\n导入完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  总计: {total}")


async def main() -> None:
    """主函数"""
    # 检查文档文件是否存在
    missing_files = [f for f in DOCUMENT_FILES if not f.exists()]
    if missing_files:
        print("警告: 以下文档文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n将只导入存在的文件...")
        document_files = [f for f in DOCUMENT_FILES if f.exists()]
    else:
        document_files = DOCUMENT_FILES

    if not document_files:
        print("没有找到可导入的文档文件")
        return

    # 初始化 agent memory
    print(f"初始化 ChromaDB (路径: {CHROMA_DB_PATH})...")
    agent_memory = ChromaAgentMemory(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH)
    )

    # 导入文档
    await import_text_memory(agent_memory, document_files)

    print(f"\n文本记忆已成功导入到 agent memory!")
    print(f"ChromaDB 路径: {CHROMA_DB_PATH}")
    print(f"Collection 名称: {CHROMA_COLLECTION_NAME}")


if __name__ == "__main__":
    asyncio.run(main())

