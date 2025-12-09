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
    创建优化的提示词
    
    Args:
        task: 任务信息
        existing_content: 现有文件内容（用于提取工程师提供信息）
    """
    raw_question = (task.get("question", "") or "").strip()
    table_list = task.get("table_list", [])
    knowledge = (task.get("knowledge", "") or "").strip()
    sql_id = (task.get("sql_id", "") or "").strip()
    
    # 构建提示词（Markdown格式）
    prompt_parts = []
    
    # 角色定位
    prompt_parts.append("# 角色定位\n\n")
    prompt_parts.append("你是 **StarRocks 数据库专家分析师**，处理**全局架构升级期**的数据库分析任务。\n")
    prompt_parts.append("**原则**：基于升级背景做判断；只提供验证过的正确答案；禁止生成虚假数据。\n\n")
    
    # 任务背景
    prompt_parts.append("# 任务背景\n\n")
    prompt_parts.append("⚠️ **重要背景信息**：\n")
    prompt_parts.append("1. **所有题目都一定有答案**：当前任务是 SQL 题目，每个题目都经过验证确保存在正确答案。如果你的查询结果为空或不符合预期，一定是你的错误，需要重新检索信息并修复。\n")
    prompt_parts.append("2. **查询结果为 NULL 是绝对错误的**：如果执行 SQL 后结果全为 NULL、缺少数值或数据异常，说明你的 SQL 逻辑存在错误，必须修复 SQL，严禁生成假的示例数据。\n\n")
    
    # 固定工作流程
    prompt_parts.append("# 固定工作流程\n\n")
    prompt_parts.append("⚠️ **必须按顺序执行，不得跳过或颠倒**\n\n")
    
    # 步骤1：需求解析
    prompt_parts.append("## 步骤 1：需求解析\n")
    prompt_parts.append("阅读并总结：用户问题、时间范围、业务口径、指标口径、涉及表名和字段。\n")
    prompt_parts.append("**信息优先级**：工程师提供信息 > 任务补充 > 用户问题\n\n")
    
    # 步骤2：信息检索
    prompt_parts.append("## 步骤 2：信息检索（不确定时按顺序执行）\n")
    prompt_parts.append("**2.1 搜索历史记录**（优先）：查找相似问题的解决方案和成功案例\n")
    prompt_parts.append("**2.2 搜索 RAG 系统**：调用 `query_lightrag` 获取表结构、字段含义、实体关系、架构升级信息\n\n")
    
    # 步骤3：SQL 设计与验证
    prompt_parts.append("## 步骤 3：SQL 设计与验证\n")
    prompt_parts.append("**3.1 编写 SQL**：编写完整可执行 SQL，说明过滤条件、关联逻辑、指标计算；考虑架构升级兼容性（字段迁移、双写阶段、新老字段共存）\n")
    prompt_parts.append("**3.2 验证 SQL**：调用 `run_sql_query` 验证（可用 `EXPLAIN` 或 `LIMIT` 测试），检查执行是否成功、结果是否符合预期\n\n")
    
    # 步骤4：错误修复
    prompt_parts.append("## 步骤 4：错误修复（如需要）\n")
    prompt_parts.append("**SQL 执行失败**：分析错误信息 → 返回步骤 2 重新检索 → 修复 SQL → 重新验证。⚠️ 连续两次失败必须回溯调整思路，避免重复错误\n")
    prompt_parts.append("**结果异常**（缺少数值或全为 NULL）：⚠️ **查询结果为 NULL 是绝对错误的**，所有题目都一定有答案。反思 SQL 逻辑和架构升级兼容性（字段选择、过滤条件、表关联、架构升级兼容性处理等）→ 修复后重新验证。⚠️ 严禁生成假的示例数据\n\n")
    
    # 步骤5：保存结果
    prompt_parts.append("## 步骤 5：保存结果（SQL 验证成功后）\n")
    prompt_parts.append(f"使用 `run_sql_query` 保存结果，**必须且只能使用** `sql_id: {sql_id}`（禁止创建新 sql_id）。工具自动保存到 `dataset_exe_result.json`，保存成功后再汇报，需附上「升级观察结论」。\n\n")
    
    # 步骤6：结果格式验证
    prompt_parts.append("## 步骤 6：结果格式验证（保存完成后必须执行）\n")
    prompt_parts.append(f"从 `dataset_exe_result.json` 读取保存的结果（使用 `sql_id: {sql_id}` 查找），验证输出格式（字段名、顺序、数量、数据类型、日期格式等）是否与用户要求完全一致。格式不一致时修复 SQL 后重新执行步骤 3→5→6，⚠️ **只有格式完全一致后才能结束任务**\n\n")
    
    # 全局架构升级背景
    prompt_parts.append("# 全局架构升级背景\n")
    prompt_parts.append("- **用户标识统一**：67 张表将 `playerid`/`sqq`/`swxid` 统一为 `suserid`，处于双写阶段\n")
    prompt_parts.append("- **功能重构**：`dwd_jordass_roundflow_hi`、`dwd_argothek_playermatchdetail_hi` 等表字段体系更新\n")
    prompt_parts.append("- **等级评价现代化**：`seasonratinglevel` 取代 `newsegmentlevel`/`ilevel`/`iviplevel`\n")
    prompt_parts.append("**处理方式**：通过日期过滤、字段存在性检查、样本 COUNT 验证迁移状态\n\n")
    
    # 结果保存规范
    prompt_parts.append("# 结果保存规范\n")
    prompt_parts.append("```json\n")
    prompt_parts.append("{\n")
    prompt_parts.append('  "query": "你的SQL语句",\n')
    prompt_parts.append(f'  "sql_id": "{sql_id}"\n')
    prompt_parts.append("}\n")
    prompt_parts.append("```\n\n")
    
    # 任务信息
    prompt_parts.append(f"# SQL 生成任务 (sql_id: {sql_id})\n\n")
    prompt_parts.append("## 用户问题\n")
    prompt_parts.append(f"{raw_question}\n\n")
    
    if knowledge:
        prompt_parts.append("## 任务补充\n")
        prompt_parts.append("> ⚠️ 任务补充信息的准确性高于用户问题，存在冲突时优先以任务补充为准。\n\n")
        prompt_parts.append(f"{knowledge}\n\n")
    
    if table_list:
        prompt_parts.append("## 涉及表名\n")
        table_list_str = ", ".join([f"`{t}`" for t in table_list]) if isinstance(table_list, list) else str(table_list)
        prompt_parts.append(f"{table_list_str}\n\n")
    
    # 生成基础提示词
    base_prompt = "".join(prompt_parts).rstrip() + "\n\n"
    
    # 如果有现有内容，提取并追加 `## 工程师提供信息` section
    if existing_content:
        engineer_info = extract_engineer_info(existing_content)
        if engineer_info:
            base_prompt += engineer_info
    
    return base_prompt


# 主程序
if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]  # 项目根目录
    dataset_path = project_root / "data" / "final_dataset.json"
    
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
        output_dir_v1 = project_root / "prompt" / "input" / "V1"
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
