from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from openai import OpenAI

# ==== 开发者可根据实际路径调整以下常量 ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECT_EXAMPLES_DIR = REPO_ROOT / "data" / "correct_examples"
ERROR_EXAMPLES_DIR = REPO_ROOT / "data" / "error_examples"
CORRECT_CHROMA_DIR = REPO_ROOT / "chroma_db" / "correct_examples"
ERROR_CHROMA_DIR = REPO_ROOT / "chroma_db" / "error_examples"
CORRECT_COLLECTION_NAME = "sql_correct_examples"
ERROR_COLLECTION_NAME = "sql_error_examples"

# DashScope 配置
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"
# ==========================================


class DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma 自定义嵌入函数，使用阿里云 DashScope OpenAI 兼容接口。"""

    def __init__(
        self,
        *,
        model: str = DASHSCOPE_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL,
    ) -> None:
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY 环境变量，无法调用阿里云嵌入服务。")

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-DashScope-SSE": "disable"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(input),
        )
        return [list(item.embedding) for item in response.data]

    def get_config(self) -> dict[str, Optional[str]]:
        return {
            "model": self._model,
            "api_base": DASHSCOPE_COMPATIBLE_BASE_URL,
        }


def parse_example_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """解析示例文件，提取关键信息"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None

    # 提取 SQL ID
    sql_id_match = re.search(r"- \*\*SQL ID\*\*: `([^`]+)`", content)
    sql_id = sql_id_match.group(1) if sql_id_match else None

    # 提取用户问题
    question_match = re.search(r"## 用户问题\n\n(.*?)\n\n---", content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # 提取使用的表
    tables_match = re.search(r"## 使用的表\n\n(.*?)\n\n---", content, re.DOTALL)
    tables_text = tables_match.group(1) if tables_match else ""
    tables = re.findall(r"- `([^`]+)`", tables_text)

    # 提取 SQL 语句
    sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
    sql = sql_match.group(1).strip() if sql_match else ""

    # 提取运行结果
    result_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    result = result_match.group(1).strip() if result_match else ""

    if not sql_id or not sql:
        print(f"跳过无效文件 {file_path}: 缺少 SQL ID 或 SQL 语句")
        return None

    return {
        "sql_id": sql_id,
        "question": question,
        "tables": tables,
        "sql": sql,
        "result": result,
        "file_path": str(file_path),
    }


def build_document_text(example: Dict[str, Any]) -> str:
    """构建用于向量化的文档文本"""
    parts = []
    
    if example["question"]:
        parts.append(f"用户问题: {example['question']}")
    
    if example["tables"]:
        tables_str = ", ".join(example["tables"])
        parts.append(f"使用的表: {tables_str}")
    
    if example["sql"]:
        parts.append(f"SQL 语句:\n{example['sql']}")
    
    return "\n\n".join(parts)


def build_metadata(example: Dict[str, Any], is_error: bool) -> Dict[str, Any]:
    """构建元数据"""
    metadata = {
        "sql_id": example["sql_id"],
        "file_path": example["file_path"],
        "is_error": str(is_error),
        "tables": ", ".join(example["tables"]) if example["tables"] else "",
    }
    
    if example["question"]:
        metadata["question"] = example["question"][:500]  # 限制长度
    
    return metadata


def import_examples_to_chroma(
    examples_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    is_error: bool = False,
) -> int:
    """将示例文件导入到 ChromaDB"""
    if not examples_dir.exists():
        print(f"示例目录不存在: {examples_dir}")
        return 0

    # 初始化 ChromaDB 客户端
    embedding_function = DashScopeEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # 获取或创建集合
    try:
        # 尝试获取现有集合
        existing_collection = client.get_collection(name=collection_name)
        # 如果集合存在，删除它以便重新导入
        client.delete_collection(name=collection_name)
        print(f"已删除现有集合: {collection_name}")
    except Exception:
        # 集合不存在，继续创建
        pass

    # 创建新集合
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )
    print(f"已创建集合: {collection_name}")

    # 读取并解析所有示例文件
    example_files = sorted(examples_dir.glob("*.txt"))
    print(f"找到 {len(example_files)} 个示例文件")

    examples: List[Dict[str, Any]] = []
    for file_path in example_files:
        example = parse_example_file(file_path)
        if example:
            examples.append(example)

    if not examples:
        print("没有有效的示例可以导入")
        return 0

    print(f"成功解析 {len(examples)} 个有效示例")

    # 准备批量插入的数据
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    skipped_count = 0

    for example in examples:
        doc_id = f"{example['sql_id']}_{Path(example['file_path']).stem}"
        doc_text = build_document_text(example)
        metadata = build_metadata(example, is_error)

        # 检查文档长度，超过8192字符的直接丢弃
        if len(doc_text) > 8192:
            skipped_count += 1
            print(f"跳过过长文档: {doc_id} (长度: {len(doc_text)} 字符，超过8192限制)")
            continue
        
        # 确保文档不为空（API要求至少1个字符）
        if not doc_text or len(doc_text.strip()) == 0:
            skipped_count += 1
            print(f"跳过空文档: {doc_id}")
            continue

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(metadata)

    if skipped_count > 0:
        print(f"总共跳过了 {skipped_count} 个无效文档")

    if not ids:
        print("没有有效的文档可以导入")
        return 0

    # 分批插入到 ChromaDB（DashScope API 限制每批最多 10 个）
    batch_size = 10
    total_added = 0
    
    try:
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            total_added += len(batch_ids)
            print(f"已导入 {total_added}/{len(ids)} 个示例...", end="\r")
        
        print(f"\n成功导入 {total_added} 个示例到集合 {collection_name}")
        if skipped_count > 0:
            print(f"（跳过了 {skipped_count} 个过长或无效的文档）")
        return total_added
    except Exception as e:
        print(f"\n导入失败: {e}")
        print(f"已成功导入 {total_added} 个示例")
        raise


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("开始导入示例到 ChromaDB")
    print("=" * 60)

    # 导入正确案例
    print("\n[1/2] 导入正确案例...")
    try:
        correct_count = import_examples_to_chroma(
            examples_dir=CORRECT_EXAMPLES_DIR,
            chroma_dir=CORRECT_CHROMA_DIR,
            collection_name=CORRECT_COLLECTION_NAME,
            is_error=False,
        )
        print(f"✓ 正确案例导入完成: {correct_count} 个示例")
    except Exception as e:
        print(f"✗ 正确案例导入失败: {e}")
        correct_count = 0

    # 导入错误案例
    print("\n[2/2] 导入错误案例...")
    try:
        error_count = import_examples_to_chroma(
            examples_dir=ERROR_EXAMPLES_DIR,
            chroma_dir=ERROR_CHROMA_DIR,
            collection_name=ERROR_COLLECTION_NAME,
            is_error=True,
        )
        print(f"✓ 错误案例导入完成: {error_count} 个示例")
    except Exception as e:
        print(f"✗ 错误案例导入失败: {e}")
        error_count = 0

    print("\n" + "=" * 60)
    print("导入完成")
    print("=" * 60)
    print(f"正确案例: {correct_count} 个")
    print(f"错误案例: {error_count} 个")
    print(f"总计: {correct_count + error_count} 个")
    print("\n数据库位置:")
    print(f"  正确案例: {CORRECT_CHROMA_DIR}")
    print(f"  错误案例: {ERROR_CHROMA_DIR}")


if __name__ == "__main__":
    main()

