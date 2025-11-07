from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vanna.core.lifecycle import LifecycleHook
from vanna.core.storage import Conversation
from vanna.core.tool import Tool
from vanna.core.tool.models import ToolContext, ToolResult
from vanna.tools import RunSqlTool
from vanna.capabilities.sql_runner import RunSqlToolArgs


class TGACRunSqlTool(RunSqlTool):
    """Extend the stock run_sql tool so we can retain the executed SQL string."""

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        result = await super().execute(context, args)
        # Persist the executed SQL for downstream consumers.
        try:
            result.metadata.setdefault("sql", args.sql)
        except AttributeError:
            # Fallback for unexpected args shape.
            result.metadata.setdefault("sql", "")
        return result


class SaveTGACResultHook(LifecycleHook):
    """Lifecycle hook that writes executed SQL results to the TGAC dataset file."""

    def __init__(self, output_path: str, seed_dataset_path: Optional[str] = None) -> None:
        self._output_path = Path(output_path).resolve()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._seed_dataset_path = Path(seed_dataset_path).resolve() if seed_dataset_path else None
        self._conversation_stack: List[Tuple[str, str]] = []
        self._pending_results: Dict[str, ToolResult] = {}
        self._lock = asyncio.Lock()
        self._seed_golden_entries()

    async def before_tool(self, tool: Tool[Any], context: ToolContext) -> None:
        # Track the conversation for the upcoming tool result.
        self._conversation_stack.append((context.conversation_id, tool.name))

    async def after_tool(self, result: ToolResult) -> Optional[ToolResult]:
        if not self._conversation_stack:
            return None
        conversation_id, tool_name = self._conversation_stack.pop()
        if tool_name == "run_sql":
            self._pending_results[conversation_id] = result
        return None

    async def after_message(self, conversation: Conversation) -> None:
        conversation_id = conversation.id
        tool_result = self._pending_results.pop(conversation_id, None)
        if not tool_result:
            return

        sql_text = self._extract_sql(tool_result, conversation)
        if not sql_text:
            return

        entry = self._build_entry(conversation_id, sql_text, tool_result)
        await self._persist_entry(entry)

    def _extract_sql(self, tool_result: ToolResult, conversation: Conversation) -> str:
        sql_text = tool_result.metadata.get("sql") if tool_result.metadata else None
        if sql_text:
            return sql_text

        for message in reversed(conversation.messages):
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if tool_call.arguments and "sql" in tool_call.arguments:
                    candidate = tool_call.arguments["sql"]
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate
        return ""

    def _build_entry(self, conversation_id: str, sql_text: str, result: ToolResult) -> Dict[str, Any]:
        sql_id = Path(conversation_id).stem
        metadata = result.metadata or {}

        raw_results = metadata.get("results")
        cleaned_results = self._clean_for_json(raw_results)
        if isinstance(cleaned_results, list) and len(cleaned_results) == 0:
            cleaned_results = None

        entry = OrderedDict(
            (
                ("sql_id", sql_id),
                ("sql", sql_text),
                ("result", cleaned_results),
                ("success", result.success),
                ("error", result.error if result.error else None),
                ("retry_steps", 0),
            )
        )
        return entry

    async def _persist_entry(self, entry: Dict[str, Any]) -> None:
        async with self._lock:
            data = self._load_entries()
            updated = False
            for idx, existing in enumerate(data):
                if existing.get("sql_id") == entry["sql_id"]:
                    data[idx] = entry
                    updated = True
                    break
            if not updated:
                data.append(entry)
            self._write_entries(data)

    def _merge_seed_entries(self, entries: Iterable[Dict[str, Any]]) -> None:
        data = self._load_entries()
        changed = False
        existing_ids = {item.get("sql_id") for item in data}
        for entry in entries:
            if entry["sql_id"] in existing_ids:
                continue
            data.append(entry)
            changed = True
        if changed:
            self._write_entries(data)

    def _load_entries(self) -> List[Dict[str, Any]]:
        if not self._output_path.exists():
            return []
        try:
            with self._output_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            pass
        return []

    def _write_entries(self, entries: List[Dict[str, Any]]) -> None:
        with self._output_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=4)
            fh.write("\n")

    def _clean_for_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._clean_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean_for_json(v) for v in value]
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    def _seed_golden_entries(self) -> None:
        if not self._seed_dataset_path or not self._seed_dataset_path.exists():
            return
        try:
            with self._seed_dataset_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            return

        if not isinstance(payload, list):
            return

        seed_entries: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not item.get("golden_sql"):
                continue
            sql_id = item.get("sql_id")
            sql_text = item.get("sql")
            if not sql_id or not isinstance(sql_id, str):
                continue
            if not sql_text or not isinstance(sql_text, str):
                continue
            entry = OrderedDict(
                (
                    ("sql_id", sql_id),
                    ("sql", sql_text),
                    ("result", None),
                    ("success", True),
                    ("error", None),
                    ("retry_steps", 0),
                )
            )
            seed_entries.append(entry)

        if seed_entries:
            self._merge_seed_entries(seed_entries)
