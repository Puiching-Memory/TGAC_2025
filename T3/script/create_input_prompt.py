"""
生成SQL生成任务的提示词
"""
import json
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer


def extract_engineer_info(existing_content: str) -> str:
    """
    从现有文件中提取 `## 工程师提供信息` section及其后续内容
    
    Args:
        existing_content: 现有文件内容
        
    Returns:
        `## 工程师提供信息` section及其后续内容的文本，如果没有则返回空字符串
    """
    target_section = "## 工程师提供信息"
    lines = existing_content.split('\n')
    
    # 反向查找，从文件末尾往前查找目标section
    for i in range(len(lines) - 1, -1, -1):
        line_stripped = lines[i].strip()
        if line_stripped == target_section or line_stripped.startswith(target_section):
            # 找到目标section，提取从该行开始到文件末尾的所有内容
            custom_content = '\n'.join(lines[i:])
            # 确保以换行符结尾
            if custom_content and not custom_content.endswith('\n'):
                custom_content += '\n'
            return custom_content
    
    return ""


def create_enhanced_prompt(task: Dict[str, Any], existing_content: str = "") -> str:
    """
    创建提示词
    
    Args:
        task: 任务信息
    """
    raw_question = (task.get("question", "") or "").strip()
    table_list = task.get("table_list", [])
    knowledge = (task.get("knowledge", "") or "").strip()
    sql_id = (task.get("sql_id", "") or "").strip()
    
    # 构建提示词（Markdown格式）
    prompt_parts = []
    
    # 角色与原则
    prompt_parts.append("# 角色定位\n")
    prompt_parts.append("你是一位 StarRocks 数据库的专家分析师。当前数据库正处于**全局架构升级期**，请确保所有分析结论都明确说明你已基于升级背景做出判断。请严格遵循以下工作流程：\n\n")
    prompt_parts.append("1. **需求解析**：先总结用户问题、时间范围、业务口径及指标口径。\n")
    prompt_parts.append("2. **智能检索**：\n")
    prompt_parts.append("   - 当对表结构、字段含义、实体关系或历史 SQL 成功案例不够确定时，调用 `query_lightrag` 获取线索。\n")
    prompt_parts.append("   - 若 SQL 运行失败、返回列缺失或结果不符合预期，必须重新检索或补充信息后再尝试。\n")
    prompt_parts.append("3. **SQL 设计**：在掌握充分结构信息后再编写完整 SQL，清晰说明过滤条件、关联逻辑与指标计算方式。\n")
    prompt_parts.append("4. **验证闭环**：\n")
    prompt_parts.append("   - 在输出 SQL 前，先调用 `run_sql_query` 进行验证（如 `EXPLAIN` 或 LIMIT 结果），记录执行是否成功。\n")
    prompt_parts.append("   - 连续两次失败必须回溯信息、重新检索或调整思路，避免重复尝试同一错误 SQL。\n")
    prompt_parts.append("   - 如果查询结果缺少数值或全为 NULL，优先反思 SQL 逻辑、字段选择、过滤条件或表关联是否正确，而不是生成假的示例数据。\n")
    prompt_parts.append("   - 输出回答时需显式说明验证结果，并指出哪些字段在升级策略下经过实际数据检查。\n\n")

    prompt_parts.append("## 全局架构升级背景\n")
    prompt_parts.append("当前数据库处于全局架构升级期，编写 SQL 时需处理字段迁移、双写等场景：\n\n")
    prompt_parts.append("- **用户标识统一**：67 张表将 `playerid`/`sqq`/`swxid` 统一为 `suserid`，处于双写阶段（如 `dim_argothek_gplayerid2qqwxid_df` 在 2025.3-7 月双写）。\n")
    prompt_parts.append("- **功能重构**：`dwd_jordass_roundflow_hi`、`dwd_argothek_playermatchdetail_hi` 等表字段体系更新。\n")
    prompt_parts.append("- **等级评价现代化**：`seasonratinglevel` 取代 `newsegmentlevel`/`ilevel`/`iviplevel`（`dwd_jordass_roundflow_hi` 在 2024 下半年-2025 年升级）。\n\n")
    prompt_parts.append("**处理方式**：通过日期过滤、字段存在性检查或样本 COUNT 验证迁移状态；回答需附上「升级观察结论」说明如何兼容迁移数据。\n\n")
    prompt_parts.append("保持逻辑严谨、步骤透明，确保每次输出都能复现与追踪。\n\n")

    # 任务说明
    prompt_parts.append("## 结果保存\n")
    prompt_parts.append(
        "- **重要**：在调用 `run_sql_query` 时，**必须且只能使用**任务中给定的 `sql_id` 参数保存结果：\n"
        "  ```json\n"
        "  {\n"
        '    "query": "你的SQL语句",\n'
        f'    "sql_id": "{sql_id}"\n'
        "  }\n"
        "  ```\n"
        f"- **严禁**创建新的sql_id字段（如sql_id_xxx、sql_id_test等），**每个任务的所有查询都必须使用同一个sql_id：`{sql_id}`**\n"
        "- 工具会自动获取所有结果（不受行数限制）并保存到 `dataset_exe_result.json`\n"
        "- 工具返回摘要信息（行数、列名等），完整数据已保存到文件\n"
        "- 保存成功后再向用户汇报最终结论\n\n"
    )

    prompt_parts.append(f"# SQL 生成任务\n")
    prompt_parts.append(f"sql_id: {sql_id}\n")
    prompt_parts.append("## 用户问题\n")
    prompt_parts.append(f"{raw_question}\n")
    
    if knowledge:
        prompt_parts.append("## 任务补充\n")
        prompt_parts.append("**注意**：任务补充信息的准确性高于用户问题。如果用户问题的语义不准确或存在歧义，请优先以任务补充为准。\n\n")
        prompt_parts.append(f"{knowledge}\n")
    
    if table_list:
        prompt_parts.append("## 涉及表名\n")
        table_list_str = ", ".join([f"`{t}`" for t in table_list]) if isinstance(table_list, list) else str(table_list)
        prompt_parts.append(f"{table_list_str}\n")
    
    # 统一写入优先级说明（无论是否有工程师信息，都写入完整说明）
    prompt_parts.append("## 信息优先级\n")
    prompt_parts.append("**优先级说明**：工程师提供信息的准确性最高，其次是任务补充，最后是用户问题。如果存在冲突，请严格按照以下优先级执行：**工程师提供信息 > 任务补充 > 用户问题**。\n\n")
    
    # 生成基础提示词
    base_prompt = "".join(prompt_parts).rstrip() + "\n"
    
    # 如果有现有内容，提取并追加 `## 工程师提供信息` section
    if existing_content:
        engineer_info = extract_engineer_info(existing_content)
        if engineer_info:
            # 确保基础提示词和自定义section之间有空行
            if not base_prompt.endswith('\n\n'):
                if base_prompt.endswith('\n'):
                    base_prompt += '\n'
                else:
                    base_prompt += '\n\n'
            base_prompt += engineer_info
    
    return base_prompt


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
        
        # 检查现有文件是否存在，如果存在则读取内容
        output_dir_v1 = t3_root / "prompt" / "input" / "V1"
        output_dir_v1.mkdir(parents=True, exist_ok=True)
        output_path_v1 = output_dir_v1 / f"{sql_id}.txt"
        
        existing_content = ""
        if output_path_v1.exists():
            with output_path_v1.open("r", encoding="utf-8") as f:
                existing_content = f.read()
            print(f"检测到已存在文件: {output_path_v1}")
            print(f"将保留其中的 `## 工程师提供信息` section")
        
        # 生成提示词（传入现有内容以提取自定义section）
        prompt_v1 = create_enhanced_prompt(t, existing_content)
        
        # 计算token数量
        token_count_v1 = len(tokenizer.encode(prompt_v1))
        
        print(f"\n{'='*60}")
        print(f"SQL ID: {sql_id}")
        print(f"Token count (V1): {token_count_v1}")
        print(f"{'='*60}")
        print(prompt_v1)
        
        # 保存 V1 提示词
        with output_path_v1.open("w", encoding="utf-8") as f:
            f.write(prompt_v1)
        
        print(f"已保存到: {output_path_v1}")
