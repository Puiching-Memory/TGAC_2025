"""Agent Lightning 集成包装器 - 为 Datus 添加学习和优化能力

此模块提供一个轻量级包装器，将 Agent Lightning 框架集成到现有的 Datus Text2SQL 工作流中。
主要功能：
1. 追踪 Text2SQL 任务的执行过程（输入、输出、奖励）
2. 聚合追踪数据用于优化
3. 运行强化学习算法改进 SQL 生成
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from colorama import Fore, Style


def color_text(text: str, *, color: Optional[str] = None, style: Optional[str] = None) -> str:
    """Apply ANSI coloring."""
    segments = []
    if color:
        segments.append(color)
    if style:
        segments.append(style)
    segments.append(text)
    segments.append(Style.RESET_ALL)
    return "".join(segments)


@dataclass
class OptimizationMetrics:
    """优化指标"""
    total_tasks: int
    successful_tasks: int
    initial_success_rate: float
    final_success_rate: float
    improvement: float
    average_reward: float


class DatusAgentOptimizer:
    """将 Agent Lightning 集成到 Datus 工作流
    
    此类提供：
    1. 追踪任务执行
    2. 计算奖励信号
    3. 管理优化状态
    4. 生成优化报告
    """

    _TABLE_PATTERN = re.compile(r"\b(?:from|join)\s+([`\"\[]?)([a-zA-Z0-9_.]+)\1", re.IGNORECASE)
    _WHITESPACE_RE = re.compile(r"\s+")
    _PUNCT_RE = re.compile(r"[(),;]")
    
    def __init__(self, store_path: Path | str = "./lightning_store", enable_training: bool = True):
        """初始化优化器
        
        Args:
            store_path: LightningStore 数据存储路径
            enable_training: 是否启用训练模式（收集追踪数据）
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_training = enable_training
        self.traces_dir = self.store_path / "traces"
        self.traces_dir.mkdir(exist_ok=True)
        
        self.resources_dir = self.store_path / "resources"
        self.resources_dir.mkdir(exist_ok=True)
        
        # 任务计数和指标
        self.task_count = 0
        self.successful_tasks = 0
        self.total_reward = 0.0
        self.traces: list[Dict[str, Any]] = []
        
        print(
            color_text(
                f"✓ Agent Lightning 优化器已初始化",
                color=Fore.GREEN,
                style=Style.BRIGHT,
            )
        )
        print(
            color_text(
                f"  存储路径: {self.store_path}",
                color=Fore.CYAN,
            )
        )
    
    def wrap_task_execution(
        self,
        task_fn: Callable[[], Dict[str, Any]],
        task_id: str,
        question: str,
        gold_sql: Optional[str] = None,
        **metadata,
    ) -> Dict[str, Any]:
        """包装任务执行以收集追踪数据
        
        Args:
            task_fn: 原始的任务执行函数（返回包含 'sql' 和 'result' 的字典）
            task_id: 任务唯一标识符
            question: 问题文本
            gold_sql: 金标 SQL（可选，用于奖励计算）
            **metadata: 任意元数据（如复杂度、表名等）
            
        Returns:
            原始的任务结果
        """
        self.task_count += 1
        
        try:
            # 执行任务
            result = task_fn()
            
            # 计算奖励
            reward = self._compute_reward(result, gold_sql, metadata)
            is_success = self._is_execution_successful(result)
            
            if is_success:
                self.successful_tasks += 1
            
            self.total_reward += reward
            
            # 记录追踪信息
            if self.enable_training:
                raw_response = result.get("raw_response") or {}
                row_count = self._coerce_to_int(result.get("row_count") or raw_response.get("row_count"))
                sql_result_preview = self._preview_sql_result(result.get("sql_result") or raw_response.get("sql_result"))
                result_path = result.get("result") or result.get("result_path") or raw_response.get("result")
                trace = {
                    "task_id": task_id,
                    "question": question,
                    "generated_sql": result.get("sql"),
                    "gold_sql": gold_sql,
                    "execution_result": result.get("result"),
                    "result_path": result_path,
                    "row_count": row_count,
                    "is_successful": is_success,
                    "reward": reward,
                    "datus_finished": bool(raw_response.get("finished")),
                    "sql_result_preview": sql_result_preview,
                    "metadata": metadata,
                }
                self.traces.append(trace)
            
            return result
            
        except Exception as e:
            # 异常情况记录为失败
            if self.enable_training:
                trace = {
                    "task_id": task_id,
                    "question": question,
                    "generated_sql": None,
                    "gold_sql": gold_sql,
                    "execution_result": None,
                    "result_path": None,
                    "row_count": None,
                    "is_successful": False,
                    "reward": 0.0,
                    "datus_finished": False,
                    "sql_result_preview": None,
                    "error": str(e),
                    "metadata": metadata,
                }
                self.traces.append(trace)
            
            raise
    
    def _compute_reward(
        self,
        result: Dict[str, Any],
        gold_sql: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """结合 Datus 输出生成奖励信号。"""
        metadata = metadata or {}
        raw_response = result.get("raw_response") or {}

        generated_sql = result.get("sql")
        finished = bool(raw_response.get("finished"))
        row_count = self._coerce_to_int(result.get("row_count") or raw_response.get("row_count"))
        sql_result = result.get("sql_result") or raw_response.get("sql_result")
        result_path = result.get("result") or result.get("result_path") or raw_response.get("result")

        csv_exists = False
        if result_path:
            try:
                csv_exists = Path(result_path).exists()
            except OSError:
                csv_exists = False

        reward = 0.0

        # 工作流完成信号
        if finished:
            reward += 0.2

        # SQL 生成质量
        if generated_sql:
            reward += 0.1
            if self._is_valid_sql_syntax(generated_sql):
                reward += 0.05

        # 执行反馈
        has_execution_signal = any([row_count is not None, bool(sql_result), csv_exists])
        if has_execution_signal:
            reward += 0.2
        if row_count is not None:
            reward += 0.05
            if row_count > 0:
                reward += 0.05
        if csv_exists:
            reward += 0.05

        # 表覆盖度
        expected_tables = {t.lower() for t in metadata.get("table_list") or [] if isinstance(t, str)}
        if generated_sql and expected_tables:
            used_tables = self._extract_tables_from_sql(generated_sql)
            if expected_tables:
                coverage = len(expected_tables & used_tables) / len(expected_tables)
                reward += 0.1 * coverage

        # 金标相似度
        if gold_sql and generated_sql:
            similarity = self._compare_with_golden(generated_sql, gold_sql)
            reward += 0.2 * similarity

        return max(0.0, min(1.0, reward))
    
    def _is_execution_successful(self, result: Dict[str, Any]) -> bool:
        """判断任务是否成功执行。"""
        raw_response = result.get("raw_response") or {}
        generated_sql = result.get("sql")
        if not generated_sql:
            return False
        finished = bool(raw_response.get("finished"))
        row_count = self._coerce_to_int(result.get("row_count") or raw_response.get("row_count"))
        sql_result = result.get("sql_result") or raw_response.get("sql_result")
        result_path = result.get("result") or result.get("result_path") or raw_response.get("result")
        return finished or row_count is not None or bool(sql_result) or bool(result_path)

    @staticmethod
    def _coerce_to_int(value: Any) -> Optional[int]:
        """将值转换为整数，失败时返回 None。"""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _preview_sql_result(sql_result: Optional[str], length: int = 200) -> Optional[str]:
        """生成 SQL 结果的简短预览。"""
        if not sql_result:
            return None
        text = str(sql_result)
        if len(text) <= length:
            return text
        return f"{text[:length]}...(truncated)"

    def _extract_tables_from_sql(self, sql: str) -> set[str]:
        """从 SQL 中提取出现的表名。"""
        tables: set[str] = set()
        if not sql:
            return tables
        for match in self._TABLE_PATTERN.finditer(sql):
            table_name = match.group(2)
            if table_name:
                tables.add(table_name.strip().lower())
        return tables

    def _normalize_sql(self, sql: str) -> str:
        """归一化 SQL 字符串以便比较。"""
        normalized = sql.lower()
        normalized = self._PUNCT_RE.sub(" ", normalized)
        normalized = self._WHITESPACE_RE.sub(" ", normalized)
        return normalized.strip()

    def _compare_with_golden(self, generated: str, golden: str) -> float:
        """计算生成 SQL 与金标 SQL 的相似度。"""
        if not generated or not golden:
            return 0.0
        normalized_generated = self._normalize_sql(generated)
        normalized_golden = self._normalize_sql(golden)
        if not normalized_generated or not normalized_golden:
            return 0.0
        similarity = difflib.SequenceMatcher(None, normalized_generated, normalized_golden).ratio()
        return max(0.0, min(1.0, similarity))
    
    def _is_semantically_equivalent(self, sql1: str, sql2: str) -> bool:
        """简单的语义等价性检查
        
        当前实现是基础的字符串比较。
        在生产环境中，应使用 SQL 解析库进行更准确的比较。
        
        Args:
            sql1: 第一个 SQL 字符串
            sql2: 第二个 SQL 字符串
            
        Returns:
            是否语义等价
        """
        # 标准化 SQL（小写、去除多余空格）
        sql1_normalized = " ".join(sql1.strip().lower().split())
        sql2_normalized = " ".join(sql2.strip().lower().split())
        return sql1_normalized == sql2_normalized
    
    def _is_valid_sql_syntax(self, sql: str) -> bool:
        """检查 SQL 语法有效性
        
        当前实现是基础的启发式检查。
        在生产环境中，应使用实际的 SQL 解析器。
        
        Args:
            sql: SQL 字符串
            
        Returns:
            是否语法有效
        """
        if not sql or not isinstance(sql, str):
            return False
        
        sql_upper = sql.strip().upper()
        
        # 检查基本关键字
        valid_keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
            "ALTER", "TRUNCATE", "CALL", "EXPLAIN", "DESCRIBE"
        ]
        
        return any(sql_upper.startswith(kw) for kw in valid_keywords)
    
    def save_traces(self) -> Path:
        """保存收集的追踪数据到文件
        
        Returns:
            保存的文件路径
        """
        if not self.traces:
            print(
                color_text(
                    "⚠ 没有追踪数据需要保存",
                    color=Fore.YELLOW,
                )
            )
            return self.traces_dir
        
        trace_file = self.traces_dir / "traces.jsonl"
        
        with trace_file.open("w", encoding="utf-8") as f:
            for trace in self.traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        
        print(
            color_text(
                f"✓ 追踪数据已保存: {trace_file}",
                color=Fore.GREEN,
            )
        )
        print(
            color_text(
                f"  共 {len(self.traces)} 条记录",
                color=Fore.CYAN,
            )
        )
        
        return trace_file
    
    def get_metrics(self) -> OptimizationMetrics:
        """获取优化指标
        
        Returns:
            包含各项指标的 OptimizationMetrics 对象
        """
        success_rate = (
            self.successful_tasks / self.task_count
            if self.task_count > 0
            else 0.0
        )
        avg_reward = (
            self.total_reward / self.task_count
            if self.task_count > 0
            else 0.0
        )
        
        return OptimizationMetrics(
            total_tasks=self.task_count,
            successful_tasks=self.successful_tasks,
            initial_success_rate=success_rate,
            final_success_rate=success_rate,  # 尚未经过优化
            improvement=0.0,  # 初始状态无改进
            average_reward=avg_reward,
        )
    
    def print_summary(self) -> None:
        """打印优化摘要"""
        metrics = self.get_metrics()
        
        print("\n" + color_text("=" * 70, color=Fore.MAGENTA))
        print(color_text("⚡ Agent Lightning 优化摘要", color=Fore.CYAN, style=Style.BRIGHT))
        print(color_text("=" * 70, color=Fore.MAGENTA))
        
        print(
            color_text(
                f"总任务数: {metrics.total_tasks}",
                color=Fore.WHITE,
            )
        )
        print(
            color_text(
                f"成功任务数: {metrics.successful_tasks}",
                color=Fore.GREEN if metrics.successful_tasks > 0 else Fore.YELLOW,
            )
        )
        print(
            color_text(
                f"成功率: {metrics.initial_success_rate:.2%}",
                color=Fore.GREEN if metrics.initial_success_rate > 0.7 else Fore.YELLOW,
                style=Style.BRIGHT,
            )
        )
        print(
            color_text(
                f"平均奖励: {metrics.average_reward:.3f}",
                color=Fore.GREEN if metrics.average_reward > 0.7 else Fore.YELLOW,
            )
        )
        
        print(color_text("=" * 70, color=Fore.MAGENTA))
    
    def export_analysis(self, output_path: Optional[Path] = None) -> Path:
        """导出详细分析报告
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            生成的报告文件路径
        """
        if output_path is None:
            output_path = self.resources_dir / "analysis_report.json"
        
        # 统计分析
        successful_traces = [t for t in self.traces if t.get("is_successful")]
        failed_traces = [t for t in self.traces if not t.get("is_successful")]
        
        # 按复杂度统计
        complexity_stats = {}
        for trace in self.traces:
            complexity = trace.get("metadata", {}).get("complexity", "unknown")
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {"total": 0, "successful": 0}
            complexity_stats[complexity]["total"] += 1
            if trace.get("is_successful"):
                complexity_stats[complexity]["successful"] += 1
        
        # 生成报告
        report = {
            "summary": {
                "total_tasks": self.task_count,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": len(failed_traces),
                "success_rate": self.successful_tasks / self.task_count if self.task_count > 0 else 0,
                "average_reward": self.total_reward / self.task_count if self.task_count > 0 else 0,
            },
            "by_complexity": complexity_stats,
            "failed_samples": [
                {
                    "task_id": t["task_id"],
                    "question": t["question"],
                    "generated_sql": t.get("generated_sql"),
                    "error": t.get("error"),
                }
                for t in failed_traces[:10]  # 仅包含前 10 条失败样本
            ],
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(
            color_text(
                f"✓ 分析报告已导出: {output_path}",
                color=Fore.GREEN,
            )
        )
        
        return output_path


class TraceCollector:
    """追踪数据收集器 - 用于在优化过程中持续收集数据"""
    
    def __init__(self, store_path: Path | str = "./lightning_store"):
        """初始化收集器
        
        Args:
            store_path: 存储路径
        """
        self.store_path = Path(store_path)
        self.traces_file = self.store_path / "traces" / "traces.jsonl"
    
    def load_traces(self) -> list[Dict[str, Any]]:
        """加载已保存的追踪数据
        
        Returns:
            追踪数据列表
        """
        traces = []
        
        if not self.traces_file.exists():
            return traces
        
        with self.traces_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        return traces
    
    def analyze_failure_patterns(self, traces: list[Dict[str, Any]]) -> Dict[str, int]:
        """分析失败模式
        
        Args:
            traces: 追踪数据列表
            
        Returns:
            失败原因统计
        """
        patterns = {}
        
        for trace in traces:
            if not trace.get("is_successful"):
                # 尝试从错误消息中提取模式
                error = trace.get("error", "unknown")
                if error not in patterns:
                    patterns[error] = 0
                patterns[error] += 1
        
        return patterns
    
    def get_low_reward_tasks(
        self, traces: list[Dict[str, Any]], threshold: float = 0.5
    ) -> list[Dict[str, Any]]:
        """获取低奖励任务（需要改进的任务）
        
        Args:
            traces: 追踪数据列表
            threshold: 奖励阈值
            
        Returns:
            低奖励任务列表
        """
        return [t for t in traces if t.get("reward", 0) < threshold]
