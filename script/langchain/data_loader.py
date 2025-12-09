"""
数据加载工具：加载数据集和知识库
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import FINAL_DATASET_PATH, COMMON_KNOWLEDGE_PATH, REPO_ROOT


def load_dataset() -> List[Dict[str, Any]]:
    """加载 final_dataset.json 数据集"""
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"数据集文件不存在: {FINAL_DATASET_PATH}")
    
    with open(FINAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_common_knowledge() -> str:
    """加载通用知识库"""
    if not COMMON_KNOWLEDGE_PATH.exists():
        return ""
    
    with open(COMMON_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_example_by_sql_id(sql_id: str) -> Optional[Dict[str, Any]]:
    """根据 sql_id 获取示例"""
    dataset = load_dataset()
    for item in dataset:
        if item.get("sql_id") == sql_id:
            return item
    return None


def get_examples_by_tables(table_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    """根据表名获取相关示例"""
    dataset = load_dataset()
    examples = []
    
    for item in dataset:
        table_list = item.get("table_list", [])
        if any(table in table_list for table in table_names):
            examples.append(item)
            if len(examples) >= limit:
                break
    
    return examples


def get_similar_questions(question: str, limit: int = 3) -> List[Dict[str, Any]]:
    """根据问题相似度获取相关示例（简单实现：关键词匹配）"""
    dataset = load_dataset()
    question_lower = question.lower()
    
    # 简单的关键词匹配
    keywords = set(question_lower.split())
    scored_examples = []
    
    for item in dataset:
        item_question = item.get("question", "").lower()
        item_keywords = set(item_question.split())
        
        # 计算关键词重叠度
        overlap = len(keywords & item_keywords)
        if overlap > 0:
            scored_examples.append((overlap, item))
    
    # 按重叠度排序
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    
    return [item for _, item in scored_examples[:limit]]


def format_example_for_prompt(example: Dict[str, Any]) -> str:
    """将示例格式化为提示词格式"""
    question = example.get("question", "")
    table_list = example.get("table_list", [])
    knowledge = example.get("knowledge", "")
    
    formatted = f"问题: {question}\n"
    formatted += f"涉及表: {', '.join(table_list)}\n"
    if knowledge:
        formatted += f"业务知识: {knowledge}\n"
    
    return formatted


