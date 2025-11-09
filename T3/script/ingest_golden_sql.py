from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from vanna.integrations.chromadb import (
    ChromaAgentMemory,
    create_sentence_transformer_embedding_function,
    get_device,
)
from vanna.core.user import User
from vanna.core.tool.models import ToolContext

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "final_dataset.json"
CKPT_ROOT = Path(__file__).resolve().parent.parent / "ckpt"
COLLECTION_NAME = "vanna_tool_memory"
PERSIST_DIR = "./T3/chroma_db"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CKPT_SCORE_FILENAME = "score.csv"
CKPT_RESULT_FILENAME = "dataset_exe_result.json"


def _load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Dataset payload is not a list")
    return payload


def _create_embedding_function() -> Optional[Any]:
    try:
        device = get_device()
        embedding_fn = create_sentence_transformer_embedding_function(
            model_name=EMBEDDING_MODEL,
            device=device,
        )
        print(
            f"Embedding function ready on device='{device}' model='{EMBEDDING_MODEL}'",
            flush=True,
        )
        return embedding_fn
    except ImportError:
        print(
            "sentence-transformers not available; using default Chroma embeddings",
            flush=True,
        )
    except Exception as exc:
        print(
            f"Failed to initialize GPU embedding model '{EMBEDDING_MODEL}': {exc}",
            flush=True,
        )
    return None


def _create_agent_memory() -> ChromaAgentMemory:
    embedding_fn = _create_embedding_function()
    return ChromaAgentMemory(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_fn,
    )


def _iterate_with_progress(items: List[Any], description: str) -> Iterable[Any]:
    if not items:
        return []

    if tqdm is None:
        return items

    progress = tqdm(total=len(items), desc=description)

    def _generator() -> Iterable[Any]:
        try:
            for item in items:
                yield item
                progress.update(1)
        finally:
            progress.close()

    return _generator()


def _build_dataset_index(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for entry in dataset:
        sql_id = entry.get("sql_id")
        if isinstance(sql_id, str) and sql_id:
            index[sql_id] = entry
    return index


def _build_golden_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sql_id": entry.get("sql_id"),
        "complexity": entry.get("复杂度"),
        "table_list": entry.get("table_list"),
        "knowledge": entry.get("knowledge"),
        "source_path": str(DATASET_PATH),
        "source": "golden_dataset",
        "label": "positive",
        "golden_sql": True,
    }


def _build_ckpt_metadata(
    base_entry: Optional[Dict[str, Any]],
    ckpt_run: str,
    score: int,
    result_preview: Optional[Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": "ckpt_history",
        "ckpt_version": ckpt_run,
        "score": score,
        "label": "positive",
        "result_preview": result_preview,
    }
    if base_entry:
        metadata.update(
            {
                "sql_id": base_entry.get("sql_id"),
                "complexity": base_entry.get("复杂度"),
                "table_list": base_entry.get("table_list"),
                "knowledge": base_entry.get("knowledge"),
            }
        )
    return metadata


def _build_tool_context(
    agent_memory: ChromaAgentMemory,
    identifier: str,
    ingest_mode: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ToolContext:
    user = User(
        id="offline_ingest",
        email="offline_ingest@local",
        group_memberships=["admin"],
    )
    metadata = {"ingest_mode": ingest_mode}
    if extra_metadata:
        metadata.update(extra_metadata)
    return ToolContext(
        user=user,
        conversation_id=f"{ingest_mode}-{identifier}",
        request_id=f"{ingest_mode}-request-{identifier}",
        agent_memory=agent_memory,
        metadata=metadata,
    )


def _sanitize_question(raw_question: Any) -> str:
    if not isinstance(raw_question, str):
        return ""
    return raw_question.strip()


def _load_score_map(score_path: Path) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    if not score_path.exists():
        return scores
    with score_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sql_id = (row.get("SQL ID") or "").strip()
            score_raw = row.get("得分")
            if not sql_id:
                continue
            try:
                scores[sql_id] = int(score_raw) if score_raw is not None else 0
            except ValueError:
                scores[sql_id] = 0
    return scores


def _truncate_result_preview(result: Any, limit: int = 5) -> Any:
    if not isinstance(result, list):
        return result
    if len(result) <= limit:
        return result
    return result[:limit]


async def _ingest_golden_sql(
    agent_memory: ChromaAgentMemory,
    dataset: Iterable[Dict[str, Any]],
) -> int:
    candidates: List[Dict[str, Any]] = []

    for entry in dataset:
        if entry.get("golden_sql") is not True:
            continue

        sql_text = entry.get("sql")
        question = _sanitize_question(entry.get("question"))

        if not sql_text or not question:
            continue

        candidates.append(
            {
                "sql_id": entry.get("sql_id") or f"golden-{len(candidates)}",
                "sql": sql_text,
                "question": question,
                "metadata": _build_golden_metadata(entry),
            }
        )

    inserted = 0
    for candidate in _iterate_with_progress(candidates, "Golden SQL"):
        sql_id = candidate["sql_id"]
        context = _build_tool_context(
            agent_memory,
            identifier=sql_id,
            ingest_mode="golden_sql_seed",
        )
        args = {"sql": candidate["sql"], "sql_id": sql_id}

        await agent_memory.save_tool_usage(
            question=candidate["question"],
            tool_name="run_sql",
            args=args,
            context=context,
            success=True,
            metadata=candidate["metadata"],
        )
        inserted += 1

    return inserted


async def _ingest_ckpt_history(
    agent_memory: ChromaAgentMemory,
    dataset_index: Dict[str, Dict[str, Any]],
) -> int:
    if not CKPT_ROOT.exists():
        return 0

    candidates: List[Dict[str, Any]] = []
    ckpt_runs = sorted([p for p in CKPT_ROOT.iterdir() if p.is_dir()])

    for run_dir in ckpt_runs:
        score_map = _load_score_map(run_dir / CKPT_SCORE_FILENAME)
        result_path = run_dir / CKPT_RESULT_FILENAME
        if not result_path.exists():
            continue

        with result_path.open("r", encoding="utf-8") as handle:
            run_payload = json.load(handle)

        if not isinstance(run_payload, list):
            continue

        for record in run_payload:
            sql_id = record.get("sql_id") if isinstance(record, dict) else None
            if not isinstance(sql_id, str) or not sql_id:
                continue

            dataset_entry = dataset_index.get(sql_id)
            question = _sanitize_question(dataset_entry.get("question") if dataset_entry else None)
            if not question:
                continue

            sql_text = record.get("sql") if isinstance(record, dict) else None
            sql_text_str = sql_text if isinstance(sql_text, str) else ""
            score_value = score_map.get(sql_id, 0)
            if score_value <= 0:
                continue

            label = "positive"

            candidates.append(
                {
                    "identifier": f"{run_dir.name}-{sql_id}",
                    "question": question,
                    "args": {
                        "sql": sql_text_str,
                        "sql_id": sql_id,
                        "ckpt_version": run_dir.name,
                        "label": label,
                    },
                    "metadata": _build_ckpt_metadata(
                        base_entry=dataset_entry,
                        ckpt_run=run_dir.name,
                        score=score_value,
                        result_preview=_truncate_result_preview(record.get("result")),
                    ),
                    "extra_context_metadata": {
                        "ckpt_version": run_dir.name,
                        "label": label,
                        "sql_id": sql_id,
                    },
                }
            )

    inserted = 0
    for candidate in _iterate_with_progress(candidates, "CKPT History"):
        context = _build_tool_context(
            agent_memory,
            identifier=candidate["identifier"],
            ingest_mode="ckpt_ingest",
            extra_metadata=candidate["extra_context_metadata"],
        )

        await agent_memory.save_tool_usage(
            question=candidate["question"],
            tool_name="run_sql",
            args=candidate["args"],
            context=context,
            success=True,
            metadata=candidate["metadata"],
        )
        inserted += 1

    return inserted


async def ingest() -> None:
    dataset = _load_dataset()
    dataset_index = _build_dataset_index(dataset)
    agent_memory = _create_agent_memory()

    golden_count = await _ingest_golden_sql(agent_memory, dataset)
    ckpt_count = await _ingest_ckpt_history(agent_memory, dataset_index)

    print(
        "Ingest summary:",
        f"golden_sql={golden_count}",
        f"ckpt_records={ckpt_count}",
        f"collection='{COLLECTION_NAME}'",
        sep=" ",
        flush=True,
    )


def main() -> None:
    asyncio.run(ingest())


if __name__ == "__main__":
    main()
