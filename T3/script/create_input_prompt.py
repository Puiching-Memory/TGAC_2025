"""
生成SQL生成任务的提示词
"""
import json
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer


def create_enhanced_prompt(task: Dict[str, Any]) -> str:
    """
    创建提示词
    
    Args:
        task: 任务信息
    """
    raw_question = (task.get("question", "") or "").strip()
    table_list = task.get("table_list", [])
    knowledge = (task.get("knowledge", "") or "").strip()
    if knowledge:
        question = f"{raw_question}\n\n【任务补充】\n{knowledge}"
    else:
        question = raw_question
    
    # 构建提示词（Markdown格式）
    prompt_parts = []
    
    # 角色提示
    prompt_parts.append("你是一个StarRocks数据库专家，擅长根据业务需求编写准确、高效的SQL查询语句。\n\n")
    
    # 标题
    prompt_parts.append("# SQL生成任务\n")
    
    # 用户问题
    prompt_parts.append("## 用户问题\n")
    prompt_parts.append(f"{question}\n")
    
    # 涉及表名
    # if table_list:
    #     prompt_parts.append("## 涉及表名\n")
    #     table_list_str = ", ".join([f"`{t}`" for t in table_list]) if isinstance(table_list, list) else str(table_list)
    #     prompt_parts.append(f"{table_list_str}\n")
    
    return "".join(prompt_parts).rstrip() + "\n"


def create_refined_task_prompt(task: Dict[str, Any]) -> str:
    """
    创建引导 Agent 先调研再重写需求的提示词。
    """
    sql_id = task.get("sql_id", "")
    raw_question = (task.get("question", "") or "").strip()
    table_list = task.get("table_list", [])
    knowledge = (task.get("knowledge", "") or "").strip()
    if knowledge:
        question = f"{raw_question}\n\n【任务补充】\n{knowledge}"
    else:
        question = raw_question

    prompt_parts = []
    prompt_parts.append("你是一名资深游戏数据分析师，准备重写用户的不一定准确的提问。你的输出将直接替代用户提问，请通过广泛且仔细的搜索后给出答案。\n\n")
    prompt_parts.append("## 工作流程\n")
    prompt_parts.append(
        "1. 围绕用户需求中的业务概念、指标口径、时间范围、过滤条件等，检索并梳理可能涉及的知识点和依赖关系。\n"
        "2. 将检索到的知识与已知补充知识进行融合，检查逻辑上的一致性和缺漏。\n"
        "3. 以专业视角重新描述用户需求，明确目标、输出口径、假设前提和必须遵守的业务规则。\n"
        "4. 在重写后的需求中显式嵌入所有必须知道的知识和概念，避免遗漏任何关键约束。\n"
    )

    prompt_parts.append("\n## 原始需求\n")
    prompt_parts.append(f"{question}\n")

    # if table_list:
    #     prompt_parts.append("\n## 相关表\n")
    #     table_list_str = ", ".join([f"`{t}`" for t in table_list]) if isinstance(table_list, list) else str(table_list)
    #     prompt_parts.append(f"{table_list_str}\n")

    return "".join(prompt_parts).rstrip() + "\n"


# 主程序
if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    t3_root = script_path.parents[1]  # T3/
    dataset_path = t3_root / "data" / "final_dataset.json"
    
    # 加载数据集
    with dataset_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    
    # 处理每个任务
    for t in json_data:
        if t.get("golden_sql"):
            continue
        
        sql_id = t.get("sql_id", "")
        if not sql_id:
            continue
        
        # 生成提示词
        prompt_v1 = create_enhanced_prompt(t)
        prompt_v2 = create_refined_task_prompt(t)
        
        # 计算token数量
        token_count_v1 = len(tokenizer.encode(prompt_v1))
        token_count_v2 = len(tokenizer.encode(prompt_v2))
        
        print(f"\n{'='*60}")
        print(f"SQL ID: {sql_id}")
        print(f"Token count (V1): {token_count_v1}")
        print(f"{'='*60}")
        print(prompt_v1)
        
        # 保存 V1 提示词
        output_dir_v1 = t3_root / "prompt" / "input" / "V1"
        output_dir_v1.mkdir(parents=True, exist_ok=True)
        output_path_v1 = output_dir_v1 / f"{sql_id}.txt"
        
        with output_path_v1.open("w", encoding="utf-8") as f:
            f.write(prompt_v1)
        
        print(f"已保存到: {output_path_v1}")
        
        print(f"\n{'='*60}")
        print(f"Token count (V2): {token_count_v2}")
        print(f"{'='*60}")
        print(prompt_v2)
        
        # 保存 V2 提示词
        output_dir_v2 = t3_root / "prompt" / "input" / "V2"
        output_dir_v2.mkdir(parents=True, exist_ok=True)
        output_path_v2 = output_dir_v2 / f"{sql_id}.txt"
        
        with output_path_v2.open("w", encoding="utf-8") as f:
            f.write(prompt_v2)
        
        print(f"已保存到: {output_path_v2}")
