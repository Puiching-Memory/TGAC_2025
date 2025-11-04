"""Agent Lightning 优化训练脚本

此脚本可以独立运行，用于在收集到足够的追踪数据后启动优化训练。
流程：
1. 加载已保存的追踪数据
2. 分析失败模式和低奖励任务
3. 运行优化算法生成改进的提示词
4. 生成详细的优化报告
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

from agent_lightning_wrapper import (
    DatusAgentOptimizer,
    TraceCollector,
    color_text,
)


def analyze_traces(store_path: Path = Path("./lightning_store")) -> None:
    """分析已收集的追踪数据
    
    Args:
        store_path: LightningStore 路径
    """
    print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
    print(color_text("📊 追踪数据分析", color=Fore.CYAN, style=Style.BRIGHT))
    print(color_text("=" * 70, color=Fore.MAGENTA))
    
    collector = TraceCollector(store_path=store_path)
    traces = collector.load_traces()
    
    if not traces:
        print(color_text("⚠ 没有找到追踪数据", color=Fore.YELLOW))
        print(color_text("请先运行: python run_datus_with_agent_lightning.py", color=Fore.CYAN))
        return
    
    print(color_text(f"✓ 加载了 {len(traces)} 条追踪记录", color=Fore.GREEN))
    
    # 统计成功率
    successful = sum(1 for t in traces if t.get("is_successful"))
    success_rate = successful / len(traces) if traces else 0
    
    print(color_text(f"\n📈 整体成功率: {success_rate:.2%}", color=Fore.YELLOW, style=Style.BRIGHT))
    print(color_text(f"   成功任务: {successful}/{len(traces)}", color=Fore.GREEN))
    
    # 分析失败模式
    print(color_text("\n🔍 失败模式分析", color=Fore.CYAN))
    failure_patterns = collector.analyze_failure_patterns(traces)
    
    if failure_patterns:
        for error, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(color_text(f"   {error}: {count} 次", color=Fore.YELLOW))
    else:
        print(color_text("   ✓ 没有失败任务", color=Fore.GREEN))
    
    # 低奖励任务
    print(color_text("\n⚠️ 需要改进的任务（低奖励）", color=Fore.CYAN))
    low_reward_tasks = collector.get_low_reward_tasks(traces, threshold=0.5)
    
    print(color_text(f"   共 {len(low_reward_tasks)} 条任务需要改进", color=Fore.YELLOW))
    
    if low_reward_tasks:
        print(color_text("\n   前 5 条低奖励任务:", color=Fore.CYAN))
        for task in low_reward_tasks[:5]:
            print(
                color_text(
                    f"   • [{task['task_id']}] 奖励: {task['reward']:.3f}",
                    color=Fore.YELLOW
                )
            )
            print(color_text(f"     问题: {task['question'][:50]}...", color=Fore.WHITE))
    
    # 按复杂度统计
    print(color_text("\n📊 按复杂度统计", color=Fore.CYAN))
    complexity_stats: Dict[str, Dict[str, int]] = {}
    
    for trace in traces:
        complexity = trace.get("metadata", {}).get("complexity", "unknown")
        if complexity not in complexity_stats:
            complexity_stats[complexity] = {"total": 0, "successful": 0, "avg_reward": 0.0}
        
        complexity_stats[complexity]["total"] += 1
        if trace.get("is_successful"):
            complexity_stats[complexity]["successful"] += 1
        complexity_stats[complexity]["avg_reward"] += trace.get("reward", 0)
    
    # 计算平均奖励
    for stats in complexity_stats.values():
        if stats["total"] > 0:
            stats["avg_reward"] /= stats["total"]
    
    for complexity, stats in sorted(complexity_stats.items()):
        success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        color = Fore.GREEN if success_rate > 0.7 else (Fore.YELLOW if success_rate > 0.5 else Fore.RED)
        print(
            color_text(
                f"   {complexity}: {success_rate:.2%} ({stats['successful']}/{stats['total']})"
                f" | 平均奖励: {stats['avg_reward']:.3f}",
                color=color,
            )
        )


def simulate_optimization(store_path: Path = Path("./lightning_store")) -> None:
    """模拟优化过程（演示）
    
    在真实场景中，这里会使用 GRPO/DPO 等算法进行优化。
    当前版本仅演示数据收集和分析流程。
    
    Args:
        store_path: LightningStore 路径
    """
    print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
    print(color_text("⚡ Agent Lightning 优化模拟", color=Fore.CYAN, style=Style.BRIGHT))
    print(color_text("=" * 70, color=Fore.MAGENTA))
    
    optimizer = DatusAgentOptimizer(store_path=store_path, enable_training=False)
    collector = TraceCollector(store_path=store_path)
    traces = collector.load_traces()
    
    if not traces:
        print(color_text("⚠ 没有追踪数据，无法进行优化", color=Fore.YELLOW))
        return
    
    print(color_text("\n🔄 模拟优化流程...", color=Fore.CYAN))
    
    # 步骤 1: 识别关键问题
    print(color_text("\n  [步骤 1] 识别关键问题", color=Fore.YELLOW, style=Style.BRIGHT))
    low_reward = collector.get_low_reward_tasks(traces, threshold=0.5)
    print(color_text(f"  ✓ 识别了 {len(low_reward)} 条低奖励任务", color=Fore.GREEN))
    
    # 步骤 2: 分析模式
    print(color_text("\n  [步骤 2] 分析失败模式", color=Fore.YELLOW, style=Style.BRIGHT))
    patterns = collector.analyze_failure_patterns(traces)
    print(color_text(f"  ✓ 识别了 {len(patterns)} 种失败模式", color=Fore.GREEN))
    
    # 步骤 3: 生成优化建议
    print(color_text("\n  [步骤 3] 生成优化建议", color=Fore.YELLOW, style=Style.BRIGHT))
    suggestions = _generate_optimization_suggestions(traces, patterns)
    
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(color_text(f"\n  建议 {i}: {suggestion['title']}", color=Fore.CYAN))
        print(color_text(f"  描述: {suggestion['description']}", color=Fore.WHITE))
        print(color_text(f"  预期改进: {suggestion['expected_improvement']:.2%}", color=Fore.GREEN))
    
    # 步骤 4: 提示词优化
    print(color_text("\n  [步骤 4] 生成优化的提示词", color=Fore.YELLOW, style=Style.BRIGHT))
    improved_prompts = _generate_improved_prompts(traces, suggestions)
    print(color_text(f"  ✓ 生成了 {len(improved_prompts)} 个改进的提示词模板", color=Fore.GREEN))
    
    # 保存优化结果
    print(color_text("\n  [步骤 5] 保存优化结果", color=Fore.YELLOW, style=Style.BRIGHT))
    output_dir = store_path / "optimized_prompts"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "prompt_improvements.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "suggestions": suggestions,
            "improved_prompts": improved_prompts,
        }, f, ensure_ascii=False, indent=2)
    
    print(color_text(f"  ✓ 优化结果已保存到: {output_file}", color=Fore.GREEN))
    
    print(color_text("\n✨ 优化模拟完成！", color=Fore.GREEN, style=Style.BRIGHT))
    print(color_text(
        "   在真实场景中，改进的提示词会被用于下一轮推理。",
        color=Fore.CYAN,
    ))


def _generate_optimization_suggestions(
    traces: List[Dict[str, Any]],
    patterns: Dict[str, int],
) -> List[Dict[str, Any]]:
    """生成优化建议
    
    Args:
        traces: 追踪数据
        patterns: 失败模式
        
    Returns:
        优化建议列表
    """
    suggestions = []
    
    # 分析低成功率的复杂度级别
    complexity_success = {}
    for trace in traces:
        complexity = trace.get("metadata", {}).get("complexity", "unknown")
        if complexity not in complexity_success:
            complexity_success[complexity] = {"total": 0, "successful": 0}
        complexity_success[complexity]["total"] += 1
        if trace.get("is_successful"):
            complexity_success[complexity]["successful"] += 1
    
    for complexity, stats in complexity_success.items():
        success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        if success_rate < 0.7:
            suggestions.append({
                "title": f"提高 {complexity} 难度 SQL 的成功率",
                "description": f"当前 {complexity} SQL 的成功率仅为 {success_rate:.2%}，"
                              "建议加强提示词中对复杂 SQL 结构的指导。",
                "target": complexity,
                "expected_improvement": min(0.15, 0.7 - success_rate),
            })
    
    # 基于失败模式的建议
    if patterns:
        top_error = max(patterns.items(), key=lambda x: x[1])
        suggestions.append({
            "title": f"解决最常见的错误",
            "description": f"最常见的错误是 '{top_error[0]}'（{top_error[1]} 次），"
                          "建议在提示词中添加相应的错误处理指导。",
            "target": top_error[0],
            "expected_improvement": 0.1,
        })
    
    # 通用改进建议
    suggestions.append({
        "title": "增强提示词的清晰度",
        "description": "通过添加更多示例和明确的指导，提高模型对 SQL 生成任务的理解。",
        "target": "general",
        "expected_improvement": 0.08,
    })
    
    return suggestions


def _generate_improved_prompts(
    traces: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """生成改进的提示词模板
    
    Args:
        traces: 追踪数据
        suggestions: 优化建议
        
    Returns:
        改进的提示词列表
    """
    improved_prompts = []
    
    # 基于成功的任务生成提示词
    successful_traces = [t for t in traces if t.get("is_successful")]
    
    for i, trace in enumerate(successful_traces[:3]):
        improved_prompts.append({
            "id": f"improved_prompt_{i+1}",
            "based_on": trace["task_id"],
            "template": {
                "role": "SQL 生成专家",
                "task": trace["question"],
                "schema_context": "数据库 schema 信息",
                "examples": [trace["generated_sql"]],
                "guidelines": [
                    "确保 SQL 语法正确",
                    "充分利用 schema 中的表和列信息",
                    "测试边界情况",
                ],
            },
        })
    
    return improved_prompts


def generate_report(store_path: Path = Path("./lightning_store")) -> Path:
    """生成完整的优化报告
    
    Args:
        store_path: LightningStore 路径
        
    Returns:
        报告文件路径
    """
    print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
    print(color_text("📄 生成优化报告", color=Fore.CYAN, style=Style.BRIGHT))
    print(color_text("=" * 70, color=Fore.MAGENTA))
    
    collector = TraceCollector(store_path=store_path)
    traces = collector.load_traces()
    
    if not traces:
        print(color_text("⚠ 没有追踪数据，无法生成报告", color=Fore.YELLOW))
        return store_path / "report.json"
    
    # 计算各项指标
    successful = sum(1 for t in traces if t.get("is_successful"))
    avg_reward = sum(t.get("reward", 0) for t in traces) / len(traces)
    
    report = {
        "title": "Agent Lightning 优化报告",
        "summary": {
            "total_traces": len(traces),
            "successful_tasks": successful,
            "success_rate": successful / len(traces),
            "average_reward": avg_reward,
        },
        "complexity_analysis": {},
        "recommendations": [],
    }
    
    # 按复杂度分析
    complexity_data = {}
    for trace in traces:
        complexity = trace.get("metadata", {}).get("complexity", "unknown")
        if complexity not in complexity_data:
            complexity_data[complexity] = {"tasks": [], "avg_reward": 0.0}
        complexity_data[complexity]["tasks"].append(trace)
    
    for complexity, data in complexity_data.items():
        avg = sum(t.get("reward", 0) for t in data["tasks"]) / len(data["tasks"])
        data["avg_reward"] = avg
        report["complexity_analysis"][complexity] = {
            "task_count": len(data["tasks"]),
            "success_count": sum(1 for t in data["tasks"] if t.get("is_successful")),
            "average_reward": avg,
        }
    
    # 生成建议
    if avg_reward < 0.6:
        report["recommendations"].append(
            "当前平均奖励较低，建议加强提示词引导，增加示例数量。"
        )
    
    low_complexity_success = min(
        (v.get("average_reward", 0) for v in report["complexity_analysis"].values()),
        default=1.0
    )
    
    if low_complexity_success < 0.5:
        report["recommendations"].append(
            "存在成功率较低的任务类别，建议有针对性地优化相应的提示词模板。"
        )
    
    # 保存报告
    report_file = store_path / "optimization_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(color_text(f"✓ 报告已生成: {report_file}", color=Fore.GREEN))
    
    return report_file


def main() -> None:
    """主程序入口"""
    print("=" * 70)
    print("Agent Lightning 优化训练脚本")
    print("=" * 70)
    
    store_path = Path("./lightning_store")
    
    # 分析追踪数据
    analyze_traces(store_path)
    
    # 模拟优化过程
    simulate_optimization(store_path)
    
    # 生成报告
    generate_report(store_path)
    
    print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
    print(color_text("✨ 优化流程完成", color=Fore.GREEN, style=Style.BRIGHT))
    print(color_text("=" * 70, color=Fore.MAGENTA))
    print(color_text(
        "\n下一步建议:",
        color=Fore.CYAN,
        style=Style.BRIGHT,
    ))
    print(color_text(
        "1. 查看 lightning_store/optimization_report.json 了解详细分析结果",
        color=Fore.WHITE,
    ))
    print(color_text(
        "2. 查看 lightning_store/optimized_prompts/prompt_improvements.json 获取改进建议",
        color=Fore.WHITE,
    ))
    print(color_text(
        "3. 使用改进的提示词重新运行 run_datus_with_agent_lightning.py 进行验证",
        color=Fore.WHITE,
    ))


if __name__ == "__main__":
    main()
