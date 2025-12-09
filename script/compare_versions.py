r"""
比对ckpt中任意两个版本之间，新增加了哪些正确题目

使用方法:
    python compare_versions.py [旧版本目录] [新版本目录] [选项]

示例:
    # 使用默认版本比对（V8.1 vs V8.2）
    python compare_versions.py
    
    # 比对V8和V9版本
    python compare_versions.py V8_36.05_1113 V9_40.70_1116
    
    # 比对并输出详细信息到文件
    python compare_versions.py V8_36.05_1113 V9_40.70_1116 --output diff_result.json
    
    # 只显示SQL ID列表
    python compare_versions.py V8_36.05_1113 V9_40.70_1116 --simple
"""
import csv
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
try:
    from colorama import Fore, Style, init
    # 初始化colorama以支持Windows终端颜色
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    # 如果没有安装colorama，使用空字符串作为颜色
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
    class Style:
        RESET_ALL = ""
    HAS_COLORAMA = False

CKPT_BASE_DIR = Path("ckpt")


def load_score_map(score_path: Path) -> Dict[str, int]:
    """
    加载score.csv文件，返回 {sql_id: 得分} 的字典
    
    Args:
        score_path: score.csv文件路径
        
    Returns:
        包含SQL ID和得分的字典，得分为1表示正确，0表示错误
    """
    scores: Dict[str, int] = {}
    if not score_path.exists():
        print(f"{Fore.YELLOW}警告: Score文件不存在: {score_path}{Style.RESET_ALL}")
        return scores
    
    try:
        with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sql_id = (row.get("SQL ID") or "").strip()
                if not sql_id:
                    continue
                try:
                    scores[sql_id] = int(row.get("得分", 0))
                except (TypeError, ValueError):
                    scores[sql_id] = 0
    except Exception as e:
        print(f"{Fore.RED}错误: 读取score文件失败: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    return scores


def load_dataset_results(result_path: Path) -> Dict[str, dict]:
    """
    加载dataset_exe_result.json文件，返回 {sql_id: 结果数据} 的字典
    
    Args:
        result_path: dataset_exe_result.json文件路径
        
    Returns:
        包含SQL ID和完整结果数据的字典
    """
    results: Dict[str, dict] = {}
    if not result_path.exists():
        print(f"{Fore.YELLOW}警告: 结果文件不存在: {result_path}{Style.RESET_ALL}")
        return results
    
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                for item in data:
                    sql_id = item.get("sql_id", "")
                    if sql_id:
                        results[sql_id] = item
    except Exception as e:
        print(f"{Fore.RED}错误: 读取结果文件失败: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    return results


def find_new_correct_questions(
    old_scores: Dict[str, int],
    new_scores: Dict[str, int]
) -> List[str]:
    """
    找出新增加的正确题目
    
    Args:
        old_scores: 旧版本的得分字典
        new_scores: 新版本的得分字典
        
    Returns:
        新增加的正确题目SQL ID列表
    """
    new_correct: List[str] = []
    
    for sql_id, score in new_scores.items():
        old_score = old_scores.get(sql_id, 0)
        # 新版本得分为1（正确），且旧版本得分为0或不存在
        if score == 1 and old_score != 1:
            new_correct.append(sql_id)
    
    # 按SQL ID排序（自然排序）
    new_correct.sort(key=lambda x: (int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0, x))
    
    return new_correct


def print_summary(old_version: str, new_version: str, new_correct: List[str]):
    """打印摘要信息"""
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}版本比对结果{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}旧版本: {old_version}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}新版本: {new_version}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}新增加的正确题目数量: {len(new_correct)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")


def print_simple_list(new_correct: List[str]):
    """简单模式：只打印SQL ID列表"""
    if not new_correct:
        print(f"{Fore.YELLOW}没有新增加的正确题目{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}新增加的正确题目列表:{Style.RESET_ALL}")
    for sql_id in new_correct:
        print(f"  - {sql_id}")


def print_detailed_info(
    new_correct: List[str],
    new_results: Dict[str, dict],
    old_results: Dict[str, dict]
):
    """详细模式：打印每个题目的基本信息（不包含SQL和结果）"""
    if not new_correct:
        print(f"{Fore.YELLOW}没有新增加的正确题目{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}新增加的正确题目列表:{Style.RESET_ALL}\n")
    
    for idx, sql_id in enumerate(new_correct, 1):
        new_result = new_results.get(sql_id, {})
        old_result = old_results.get(sql_id, {})
        
        # 显示基本信息
        status_info = []
        if old_result:
            status_info.append("旧版本: 错误")
        else:
            status_info.append("旧版本: 不存在")
        
        if "result" in new_result:
            result_count = len(new_result["result"])
            status_info.append(f"新版本: 正确 (结果{result_count}行)")
        else:
            status_info.append("新版本: 正确")
        
        print(f"  [{idx:3d}] {sql_id:15s} - {' | '.join(status_info)}")


def save_to_json(
    output_path: Path,
    old_version: str,
    new_version: str,
    new_correct: List[str],
    new_results: Dict[str, dict],
    old_results: Dict[str, dict]
):
    """将比对结果保存到JSON文件"""
    output_data = {
        "old_version": old_version,
        "new_version": new_version,
        "new_correct_count": len(new_correct),
        "new_correct_questions": []
    }
    
    for sql_id in new_correct:
        question_data = {
            "sql_id": sql_id,
            "old_version_status": "error" if old_results.get(sql_id) else "not_exist",
            "new_version_result": new_results.get(sql_id, {})
        }
        output_data["new_correct_questions"].append(question_data)
    
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}结果已保存到: {output_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}保存文件失败: {e}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(
        description="比对ckpt中任意两个版本之间，新增加了哪些正确题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "old_version",
        type=str,
        nargs='?',
        default="V8.1_38.70_1114",
        help="旧版本目录名称（例如: V8_36.05_1113，默认为 V8.1_38.70_1114）"
    )
    parser.add_argument(
        "new_version",
        type=str,
        nargs='?',
        default="V8.2_40.70_1116",
        help="新版本目录名称（例如: V9_40.70_1116，默认为 V8.2_40.70_1116）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="将详细结果保存到JSON文件"
    )
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="简单模式：只显示SQL ID列表"
    )
    
    args = parser.parse_args()
    
    # 构建文件路径
    old_dir = CKPT_BASE_DIR / args.old_version
    new_dir = CKPT_BASE_DIR / args.new_version
    
    # 检查目录是否存在
    if not old_dir.exists():
        print(f"{Fore.RED}错误: 旧版本目录不存在: {old_dir}{Style.RESET_ALL}")
        sys.exit(1)
    
    if not new_dir.exists():
        print(f"{Fore.RED}错误: 新版本目录不存在: {new_dir}{Style.RESET_ALL}")
        sys.exit(1)
    
    # 加载数据
    print(f"{Fore.CYAN}正在加载数据...{Style.RESET_ALL}")
    old_scores = load_score_map(old_dir / "score.csv")
    new_scores = load_score_map(new_dir / "score.csv")
    old_results = load_dataset_results(old_dir / "dataset_exe_result.json")
    new_results = load_dataset_results(new_dir / "dataset_exe_result.json")
    
    # 找出新增加的正确题目
    new_correct = find_new_correct_questions(old_scores, new_scores)
    
    # 打印摘要
    print_summary(args.old_version, args.new_version, new_correct)
    
    # 根据模式输出
    if args.simple:
        print_simple_list(new_correct)
    else:
        print_detailed_info(new_correct, new_results, old_results)
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        save_to_json(
            output_path,
            args.old_version,
            args.new_version,
            new_correct,
            new_results,
            old_results
        )


if __name__ == "__main__":
    main()

