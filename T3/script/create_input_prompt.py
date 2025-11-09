"""
直接从CKPT记录中提取正负样本，添加到提示词中
因为每个任务（sql_id）是固定不变的，所以可以直接从记录中查找
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer


def load_ckpt_cases(sql_id: str, ckpt_root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从CKPT目录中加载指定sql_id的成功和失败案例
    
    Args:
        sql_id: SQL ID
        ckpt_root: CKPT根目录路径
    
    Returns:
        (成功案例列表, 失败案例列表)
    """
    success_cases = []
    failed_cases = []
    
    if not ckpt_root.exists():
        return success_cases, failed_cases
    
    # 遍历所有ckpt版本目录，按时间倒序（最新的优先）
    ckpt_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir()], reverse=True)
    
    for ckpt_dir in ckpt_dirs:
        score_path = ckpt_dir / "score.csv"
        result_path = ckpt_dir / "dataset_exe_result.json"
        
        if not score_path.exists() or not result_path.exists():
            continue
        
        # 加载得分映射
        score_map = {}
        try:
            with score_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_sql_id = row.get("SQL ID") or row.get("sql_id") or row.get("\ufeffSQL ID", "").strip()
                    score_str = row.get("得分") or row.get("score", "0")
                    if row_sql_id:
                        try:
                            score_map[row_sql_id.strip()] = int(score_str)
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            continue
        
        # 加载执行结果
        try:
            with result_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            continue
        
        if not isinstance(results, list):
            continue
        
        # 查找该sql_id的记录
        for record in results:
            if not isinstance(record, dict):
                continue
            
            record_sql_id = record.get("sql_id")
            if not record_sql_id or str(record_sql_id).strip() != sql_id:
                continue
            
            sql_text = record.get("sql", "")
            if not sql_text or not sql_text.strip():
                continue
            
            score = score_map.get(sql_id, 0)
            success = record.get("success", True)
            error = record.get("error")
            
            case_info = {
                "sql": sql_text.strip(),
                "score": score,
                "success": success,
                "error": error or "结果与预期不匹配",
                "ckpt_version": ckpt_dir.name,
                "result": record.get("result"),
            }
            
            # 判断是成功还是失败
            is_success = score > 0 and success and not error
            is_failed = score == 0 or not success or error
            
            if is_success:
                # 成功案例：只保留最新的几个
                if len(success_cases) < 2:  # 最多保留2个成功案例
                    success_cases.append(case_info)
            elif is_failed:
                # 失败案例：只保留最新的1个
                if len(failed_cases) == 0:
                    failed_cases.append(case_info)
                    break  # 找到第一个失败案例就够了
    
    return success_cases, failed_cases


def format_case_text(case: Dict[str, Any], case_type: str, index: int) -> str:
    """
    格式化案例文本（Markdown格式）
    
    Args:
        case: 案例信息
        case_type: "positive" 或 "negative"
        index: 案例索引
    
    Returns:
        格式化后的文本
    """
    if case_type == "positive":
        case_text = f"### 成功案例 {index + 1}\n"
        
        # 元信息列表
        meta_items = []
        meta_items.append(f"- **SQL ID**: `{case.get('sql_id', 'N/A')}`")
        if case.get("score"):
            meta_items.append(f"- **得分**: {case.get('score')}")
        if case.get("ckpt_version"):
            meta_items.append(f"- **来源**: CKPT `{case.get('ckpt_version')}`")
        
        if meta_items:
            case_text += "\n".join(meta_items) + "\n"
        
        sql_text = case.get("sql", "").strip()
        if sql_text:
            case_text += f"\n**SQL:**\n\n```sql\n{sql_text}\n```\n"
        
        return case_text
    else:  # negative
        case_text = f"### 失败案例 {index + 1}\n"
        
        # 元信息列表
        meta_items = []
        meta_items.append(f"- **SQL ID**: `{case.get('sql_id', 'N/A')}`")
        if case.get("ckpt_version"):
            meta_items.append(f"- **来源**: CKPT `{case.get('ckpt_version')}`")
        
        if meta_items:
            case_text += "\n".join(meta_items) + "\n"
        
        sql_text = case.get("sql", "").strip()
        if sql_text:
            case_text += f"\n**错误SQL:**\n\n```sql\n{sql_text}\n```\n"
        
        error_msg = case.get("error", "")
        if error_msg:
            case_text += f"\n> **错误原因**: {error_msg}\n"
        
        return case_text


def create_enhanced_prompt(task: Dict[str, Any], ckpt_root: Path) -> str:
    """
    创建提示词，包含成功和失败案例信息
    
    Args:
        task: 任务信息
        ckpt_root: CKPT根目录路径
    """
    question = task.get("question", "")
    table_list = task.get("table_list", [])
    knowledge = task.get("knowledge", "") or "无"
    sql_id = task.get("sql_id", "")
    
    # 直接从CKPT记录中提取该sql_id的成功和失败案例
    success_cases, failed_cases = load_ckpt_cases(sql_id, ckpt_root)
    
    # 为案例添加sql_id信息
    for case in success_cases:
        case["sql_id"] = sql_id
    for case in failed_cases:
        case["sql_id"] = sql_id
    
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
    if table_list:
        prompt_parts.append("## 涉及表名\n")
        table_list_str = ", ".join([f"`{t}`" for t in table_list]) if isinstance(table_list, list) else str(table_list)
        prompt_parts.append(f"{table_list_str}\n")
    
    # 提示信息
    if knowledge and knowledge != "无":
        prompt_parts.append("## 提示信息\n")
        # 如果knowledge包含多行，保持格式
        knowledge_lines = knowledge.split('\n')
        for line in knowledge_lines:
            if line.strip():
                prompt_parts.append(f"{line}\n")
        prompt_parts.append("")
    
    # 如果有成功案例，添加到提示词中（作为参考）
    if success_cases:
        prompt_parts.append("## 成功案例参考\n")
        prompt_parts.append("以下案例执行成功，可作为参考：\n")
        for idx, case in enumerate(success_cases):
            case_text = format_case_text(case, "positive", idx)
            prompt_parts.append(case_text)
            if idx < len(success_cases) - 1:
                prompt_parts.append("---\n")  # 案例之间分隔线
    
    # 如果有失败案例，添加到提示词中（作为避免错误的参考）
    if failed_cases:
        prompt_parts.append("## 失败案例参考\n")
        prompt_parts.append("以下案例曾出现过错误，请避免类似问题：\n")
        for idx, case in enumerate(failed_cases):
            case_text = format_case_text(case, "negative", idx)
            prompt_parts.append(case_text)
    
    return "".join(prompt_parts).rstrip() + "\n"


# 主程序
if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    t3_root = script_path.parents[1]  # T3/
    dataset_path = t3_root / "data" / "final_dataset.json"
    ckpt_root = t3_root / "ckpt"
    
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
        prompt = create_enhanced_prompt(t, ckpt_root)
        
        # 计算token数量
        token_count = len(tokenizer.encode(prompt))
        
        print(f"\n{'='*60}")
        print(f"SQL ID: {sql_id}")
        print(f"Token count: {token_count}")
        print(f"{'='*60}")
        print(prompt)
        
        # 保存提示词
        output_path = t3_root / "prompt" / "input" / "V1" / f"{sql_id}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write(prompt)
        
        print(f"已保存到: {output_path}")
