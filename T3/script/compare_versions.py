"""版本对比工具：对比不同版本提示词的执行结果。

此脚本的主要功能：
1. 加载两个版本的执行结果
2. 对比成功率、失败的任务等
3. 生成对比报告
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set
from dataclasses import dataclass

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


def color_text(text: str, *, color: str | None = None, style: str | None = None) -> str:
    """Apply ANSI coloring."""
    segments: List[str] = []
    if color:
        segments.append(color)
    if style:
        segments.append(style)
    segments.append(text)
    segments.append(Style.RESET_ALL)
    return "".join(segments)


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class VersionStats:
    """单个版本的统计信息。"""
    version: str
    total: int
    success: int
    failed: int
    success_rate: float
    failed_ids: Set[str]
    result_file: Path


def load_results(result_path: Path) -> List[Dict[str, Any]]:
    """加载执行结果文件。"""
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_results(results: List[Dict[str, Any]], version: str, result_file: Path) -> VersionStats:
    """分析执行结果并生成统计信息。"""
    total = len(results)
    success = 0
    failed_ids: Set[str] = set()
    
    for result in results:
        sql_id = result.get("sql_id", "unknown")
        sql = result.get("sql")
        query_result = result.get("result")
        
        if sql and query_result is not None:
            success += 1
        else:
            failed_ids.add(sql_id)
    
    failed = total - success
    success_rate = (success / total * 100) if total > 0 else 0
    
    return VersionStats(
        version=version,
        total=total,
        success=success,
        failed=failed,
        success_rate=success_rate,
        failed_ids=failed_ids,
        result_file=result_file,
    )


def print_version_stats(stats: VersionStats) -> None:
    """打印单个版本的统计信息。"""
    print(f"\n{color_text('━' * 70, color=Fore.CYAN)}")
    print(color_text(f"版本: {stats.version}", color=Fore.CYAN, style=Style.BRIGHT))
    print(color_text('━' * 70, color=Fore.CYAN))
    print(f"  总任务数: {stats.total}")
    print(color_text(f"  成功: {stats.success}", color=Fore.GREEN))
    print(color_text(f"  失败: {stats.failed}", color=Fore.RED if stats.failed > 0 else Fore.GREEN))
    print(
        color_text(
            f"  成功率: {stats.success_rate:.2f}%",
            color=Fore.GREEN if stats.success_rate >= 90 else Fore.YELLOW,
            style=Style.BRIGHT,
        )
    )
    print(f"  结果文件: {stats.result_file}")


def compare_versions(stats1: VersionStats, stats2: VersionStats) -> None:
    """对比两个版本的结果。"""
    print(f"\n{color_text('=' * 70, color=Fore.MAGENTA)}")
    print(color_text("版本对比分析", color=Fore.MAGENTA, style=Style.BRIGHT))
    print(color_text('=' * 70, color=Fore.MAGENTA))
    
    # 成功率对比
    print(f"\n{color_text('📊 成功率对比:', color=Fore.CYAN, style=Style.BRIGHT)}")
    rate_diff = stats2.success_rate - stats1.success_rate
    if rate_diff > 0:
        print(
            color_text(
                f"  {stats2.version} 比 {stats1.version} 高 {rate_diff:.2f}%",
                color=Fore.GREEN,
                style=Style.BRIGHT,
            )
        )
    elif rate_diff < 0:
        print(
            color_text(
                f"  {stats2.version} 比 {stats1.version} 低 {abs(rate_diff):.2f}%",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
    else:
        print(color_text(f"  两个版本成功率相同", color=Fore.YELLOW))
    
    # 失败任务对比
    print(f"\n{color_text('❌ 失败任务分析:', color=Fore.CYAN, style=Style.BRIGHT)}")
    
    # 两个版本都失败的任务
    both_failed = stats1.failed_ids & stats2.failed_ids
    if both_failed:
        print(
            color_text(
                f"  两个版本都失败: {len(both_failed)} 个任务",
                color=Fore.RED,
            )
        )
        print(color_text(f"    {', '.join(sorted(both_failed))}", color=Fore.WHITE))
    
    # 只有第一个版本失败的任务
    only_v1_failed = stats1.failed_ids - stats2.failed_ids
    if only_v1_failed:
        print(
            color_text(
                f"  仅 {stats1.version} 失败: {len(only_v1_failed)} 个任务",
                color=Fore.YELLOW,
            )
        )
        print(color_text(f"    {', '.join(sorted(only_v1_failed))}", color=Fore.WHITE))
    
    # 只有第二个版本失败的任务
    only_v2_failed = stats2.failed_ids - stats1.failed_ids
    if only_v2_failed:
        print(
            color_text(
                f"  仅 {stats2.version} 失败: {len(only_v2_failed)} 个任务",
                color=Fore.YELLOW,
            )
        )
        print(color_text(f"    {', '.join(sorted(only_v2_failed))}", color=Fore.WHITE))
    
    # 改进和退步统计
    print(f"\n{color_text('📈 变化统计:', color=Fore.CYAN, style=Style.BRIGHT)}")
    improved = len(only_v1_failed)
    regressed = len(only_v2_failed)
    
    if improved > 0:
        print(
            color_text(
                f"  ✓ {stats2.version} 修复了 {improved} 个任务",
                color=Fore.GREEN,
            )
        )
    if regressed > 0:
        print(
            color_text(
                f"  ✗ {stats2.version} 新增了 {regressed} 个失败任务",
                color=Fore.RED,
            )
        )
    
    net_improvement = improved - regressed
    if net_improvement > 0:
        print(
            color_text(
                f"\n  总体: 净改进 {net_improvement} 个任务",
                color=Fore.GREEN,
                style=Style.BRIGHT,
            )
        )
    elif net_improvement < 0:
        print(
            color_text(
                f"\n  总体: 净退步 {abs(net_improvement)} 个任务",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
    else:
        print(
            color_text(
                f"\n  总体: 没有净变化",
                color=Fore.YELLOW,
            )
        )
    
    # 推荐
    print(f"\n{color_text('💡 推荐:', color=Fore.CYAN, style=Style.BRIGHT)}")
    if stats2.success_rate > stats1.success_rate and net_improvement > 0:
        print(
            color_text(
                f"  建议使用 {stats2.version}（成功率更高，改进明显）",
                color=Fore.GREEN,
                style=Style.BRIGHT,
            )
        )
    elif stats2.success_rate < stats1.success_rate or net_improvement < 0:
        print(
            color_text(
                f"  建议继续使用 {stats1.version}（性能更好）",
                color=Fore.YELLOW,
                style=Style.BRIGHT,
            )
        )
    else:
        print(
            color_text(
                f"  两个版本性能相近，可根据其他因素选择",
                color=Fore.YELLOW,
            )
        )


def main() -> None:
    """主程序入口。"""
    print("=" * 70)
    print(color_text("版本对比工具", color=Fore.MAGENTA, style=Style.BRIGHT))
    print("=" * 70)
    
    # 配置要对比的版本
    # 修改这里来对比不同的版本
    version1 = "v1.0.0"
    version2 = "v1.1.0"
    
    # 结果文件路径（可以根据需要修改）
    result_file1 = BASE_DIR / "upload" / f"dataset_exe_result_{version1}.json"
    result_file2 = BASE_DIR / "upload" / f"dataset_exe_result_{version2}.json"
    
    # 如果文件名不包含版本号，使用默认名称
    if not result_file1.exists():
        result_file1 = BASE_DIR / "upload" / "dataset_exe_result.json"
        print(
            color_text(
                f"\n⚠ 警告: 使用默认结果文件作为 {version1}",
                color=Fore.YELLOW,
            )
        )
    
    try:
        # 加载和分析第一个版本
        print(f"\n{color_text('📂 加载版本 1 结果...', color=Fore.CYAN)}")
        results1 = load_results(result_file1)
        stats1 = analyze_results(results1, version1, result_file1)
        print_version_stats(stats1)
        
        # 加载和分析第二个版本
        print(f"\n{color_text('📂 加载版本 2 结果...', color=Fore.CYAN)}")
        results2 = load_results(result_file2)
        stats2 = analyze_results(results2, version2, result_file2)
        print_version_stats(stats2)
        
        # 对比分析
        compare_versions(stats1, stats2)
        
        print(f"\n{color_text('=' * 70, color=Fore.MAGENTA)}")
        print(color_text("对比完成！", color=Fore.GREEN, style=Style.BRIGHT))
        print(color_text('=' * 70, color=Fore.MAGENTA))
        
    except FileNotFoundError as e:
        print(
            color_text(
                f"\n❌ 错误：文件未找到 - {e}",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
        )
        print(
            color_text(
                "\n请确保：",
                color=Fore.YELLOW,
            )
        )
        print("  1. 已运行 run_datus.py 生成结果文件")
        print("  2. 结果文件路径正确")
        print("  3. 版本号配置正确")
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
