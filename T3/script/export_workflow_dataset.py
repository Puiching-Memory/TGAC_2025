from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

try:
    from toon import encode as toon_encode
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "python-toon is required for TOON output. Install it via 'pip install python-toon'."
    ) from exc

# ---------------------------------------------------------------------------
# Simple configuration tuned for one-off usage (edit directly if needed)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "workflow"))
OUTPUT_FILE_NAME_JSON = "workflow_dataset.json"
OUTPUT_FILE_NAME_TOON = "workflow_dataset.toon"
SPLIT_BY_SUBDIR = True  # Set False to emit a single JSON/TOON file at WORKFLOW_ROOT


# ---------------------------------------------------------------------------
# Workflow JSON helpers
# ---------------------------------------------------------------------------
def list_workflow_files(workflow_dir: str) -> List[str]:
    candidates: List[str] = []
    for root, _, files in os.walk(workflow_dir):
        for entry in files:
            if entry.endswith("_workflow.json"):
                candidates.append(os.path.join(root, entry))
    return sorted(candidates)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def simplify_inputs(inputs: Iterable[Dict[str, Any]], id_to_name: Dict[str, str]) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for raw in inputs or []:
        item: Dict[str, Any] = {
            "name": raw.get("Name"),
            "type": raw.get("Type"),
            "desc": strip_or_none(raw.get("Desc")),
        }
        input_blob = raw.get("Input") or {}
        input_type = input_blob.get("InputType")
        item["source_type"] = input_type
        if input_type == "REFERENCE_OUTPUT":
            reference = input_blob.get("Reference") or {}
            source_id = reference.get("NodeID")
            item["source_node_id"] = source_id
            item["source_node_name"] = id_to_name.get(source_id)
            item["json_path"] = reference.get("JsonPath")
        elif input_type == "CUSTOM_VARIABLE":
            item["custom_var_id"] = input_blob.get("CustomVarID")
        elif input_type == "USER_INPUT":
            user_input = input_blob.get("UserInputValue") or {}
            item["values"] = user_input.get("Values")
            item["file_names"] = user_input.get("FileNames")
        elif input_type == "NODE_INPUT_PARAM":
            item["param_name"] = input_blob.get("NodeInputParamName")
        elif input_type == "REFERENCE_VARIABLE":
            ref_var = input_blob.get("Reference") or {}
            item["variable_name"] = ref_var.get("Name")
        cleaned = {k: v for k, v in item.items() if v not in (None, [], {})}
        if cleaned:
            simplified.append(cleaned)
    return simplified


def simplify_outputs(outputs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for raw in outputs or []:
        entry: Dict[str, Any] = {
            "title": raw.get("Title"),
            "type": raw.get("Type"),
            "desc": strip_or_none(raw.get("Desc")),
        }
        required = raw.get("Required") or []
        if required:
            entry["required"] = required
        properties = raw.get("Properties") or []
        if properties:
            entry["properties"] = [
                {
                    "title": prop.get("Title"),
                    "type": prop.get("Type"),
                    "desc": strip_or_none(prop.get("Desc")),
                }
                for prop in properties
            ]
        cleaned = {k: v for k, v in entry.items() if v not in (None, [], {})}
        if cleaned:
            simplified.append(cleaned)
    return simplified


def coerce_2d(values: Any) -> List[List[Any]]:
    if values is None:
        return []
    if not isinstance(values, list):
        return [[values]]
    matrix: List[List[Any]] = []
    for row in values:
        if isinstance(row, list):
            matrix.append(row)
        else:
            matrix.append([row])
    return matrix


def sheet_to_records(values: Any) -> Optional[Dict[str, Any]]:
    matrix = coerce_2d(values)
    if not matrix:
        return None

    headers_raw = matrix[0]
    headers: List[str] = []
    for idx, header in enumerate(headers_raw):
        normalized = strip_or_none(str(header)) if header is not None else None
        headers.append(normalized or f"column_{idx + 1}")

    rows: List[Dict[str, Any]] = []
    for raw_row in matrix[1:]:
        raw_row = raw_row or []
        record: Dict[str, Any] = {}
        empty = True
        for col_idx, header in enumerate(headers):
            cell = raw_row[col_idx] if col_idx < len(raw_row) else None
            value = normalize_cell(cell)
            if value is not None:
                record[header] = value
                empty = False
        if not empty:
            rows.append(record)

    return {"headers": headers, "rows": rows}


def parse_excel_files(workflow_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        import xlwings as xw
    except ImportError as exc:  # pragma: no cover - explicit dependency check
        raise RuntimeError(
            "xlwings is required to parse Excel files. Install it via 'pip install xlwings'."
        ) from exc

    excel_files: List[str] = []
    for root, _, files in os.walk(workflow_dir):
        for entry in files:
            if entry.startswith("~$"):
                continue
            if entry.lower().endswith(".xlsx"):
                excel_files.append(os.path.join(root, entry))

    if not excel_files:
        return {}

    payload: Dict[str, List[Dict[str, Any]]] = {}
    app = None
    try:
        app = xw.App(visible=False)
        app.display_alerts = False
        app.screen_updating = False

        for full_path in sorted(excel_files):
            rel_path = os.path.relpath(full_path, workflow_dir)
            book = app.books.open(full_path, update_links=False, read_only=True)
            try:
                sheets_payload: List[Dict[str, Any]] = []
                for sheet in book.sheets:
                    sheet_data = sheet_to_records(sheet.used_range.value)
                    if not sheet_data:
                        continue
                    sheets_payload.append(
                        {
                            "sheet_name": sheet.name,
                            "headers": sheet_data["headers"],
                            "rows": sheet_data["rows"],
                        }
                    )
                if sheets_payload:
                    payload[rel_path] = sheets_payload
            finally:
                book.close()
    finally:
        if app is not None:
            app.quit()

    return payload


def simplify_node(node: Dict[str, Any], id_to_name: Dict[str, str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "node_id": node.get("NodeID"),
        "name": node.get("NodeName"),
        "type": node.get("NodeType"),
    }
    desc = strip_or_none(node.get("NodeDesc"))
    if desc:
        summary["desc"] = desc

    inputs = simplify_inputs(node.get("Inputs") or [], id_to_name)
    if inputs:
        summary["inputs"] = inputs

    outputs = simplify_outputs(node.get("Outputs") or [])
    if outputs:
        summary["outputs"] = outputs

    next_ids = node.get("NextNodeIDs") or []
    if next_ids:
        summary["next_nodes"] = [
            {"node_id": nxt, "name": id_to_name.get(nxt)} for nxt in next_ids
        ]

    node_type = node.get("NodeType")
    if node_type == "CODE_EXECUTOR":
        code_blob = node.get("CodeExecutorNodeData") or {}
        if code_blob:
            summary["code_executor"] = {
                "language": code_blob.get("Language"),
                "code": strip_or_none(code_blob.get("Code")),
            }
    elif node_type == "LLM":
        llm_blob = node.get("LLMNodeData") or {}
        if llm_blob:
            summary["llm"] = {
                "model": llm_blob.get("ModelName"),
                "prompt": strip_or_none(llm_blob.get("Prompt")),
                "system_prompt": strip_or_none(llm_blob.get("SystemPrompt")),
                "temperature": llm_blob.get("ModelParams", {}).get("Temperature"),
                "top_p": llm_blob.get("ModelParams", {}).get("TopP"),
                "max_tokens": llm_blob.get("ModelParams", {}).get("MaxTokens"),
                "output_format": llm_blob.get("OutputFormat"),
            }
    elif node_type == "LOGIC_EVALUATOR":
        logic_blob = node.get("LogicEvaluatorNodeData") or {}
        if logic_blob:
            summary["logic"] = logic_blob.get("Group")
    elif node_type == "ITERATION":
        iteration_blob = node.get("IterationNodeData") or {}
        if iteration_blob:
            summary["iteration"] = {
                "mode": iteration_blob.get("IterationMode"),
                "workflow_id": iteration_blob.get("WorkflowID"),
                "body_type": iteration_blob.get("BodyType"),
            }
    elif node_type == "VAR_AGGREGATION":
        var_blob = node.get("VarAggregationNodeData") or {}
        if var_blob:
            summary["var_aggregation"] = var_blob.get("Groups")

    exception = node.get("ExceptionHandling") or {}
    exception = {k: v for k, v in exception.items() if v not in (None, "", [], {})}
    if exception:
        summary["exception_handling"] = exception

    return {k: v for k, v in summary.items() if v not in (None, [], {})}


def simplify_workflow(raw_workflow: Dict[str, Any]) -> Dict[str, Any]:
    nodes = raw_workflow.get("Nodes") or []
    id_to_name = {node.get("NodeID"): node.get("NodeName") for node in nodes}

    simplified_nodes = [simplify_node(node, id_to_name) for node in nodes]

    result: Dict[str, Any] = {
        "workflow_id": raw_workflow.get("WorkflowID"),
        "workflow_name": raw_workflow.get("WorkflowName"),
        "workflow_desc": strip_or_none(raw_workflow.get("WorkflowDesc")),
        "proto_version": raw_workflow.get("ProtoVersion"),
        "nodes": simplified_nodes,
    }
    edges = raw_workflow.get("Edge")
    if edges:
        result["edges_raw"] = edges
    return result


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def write_artifacts(target_dir: str, export_blob: Dict[str, Any], workflow_count: int, spreadsheet_count: int) -> None:
    json_path = os.path.join(target_dir, OUTPUT_FILE_NAME_JSON)
    ensure_parent_dir(json_path)
    json_text = json.dumps(export_blob, ensure_ascii=False, indent=2) + "\n"
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(json_text)

    toon_path = os.path.join(target_dir, OUTPUT_FILE_NAME_TOON)
    toon_text = toon_encode(export_blob, {"indent": 2})
    if not toon_text.endswith("\n"):
        toon_text += "\n"
    with open(toon_path, "w", encoding="utf-8") as fh:
        fh.write(toon_text)

    print(
        "Exported "
        f"{workflow_count} workflows and {spreadsheet_count} spreadsheets "
        f"to {json_path} (JSON) and {toon_path} (TOON)"
    )


def main() -> None:
    if not os.path.isdir(WORKFLOW_ROOT):
        raise FileNotFoundError(f"Workflow directory not found: {WORKFLOW_ROOT}")

    workflow_files = list_workflow_files(WORKFLOW_ROOT)
    if not workflow_files:
        raise FileNotFoundError("No workflow definition files were found")

    workflow_records: List[tuple[str, Dict[str, Any]]] = []
    for workflow_path in workflow_files:
        rel_path = os.path.relpath(workflow_path, WORKFLOW_ROOT)
        raw = load_json(workflow_path)
        workflow_records.append((rel_path, simplify_workflow(raw)))

    spreadsheet_payload: Dict[str, List[Dict[str, Any]]] = parse_excel_files(WORKFLOW_ROOT)

    if not SPLIT_BY_SUBDIR:
        export_blob = {
            "workflows": [wf for _, wf in workflow_records],
            "spreadsheets": spreadsheet_payload,
        }
        write_artifacts(WORKFLOW_ROOT, export_blob, len(workflow_records), len(spreadsheet_payload))
        return

    grouped_workflows: Dict[str, List[Dict[str, Any]]] = {}
    for rel_path, workflow in workflow_records:
        parts = rel_path.split(os.sep)
        top_level = parts[0] if len(parts) > 1 else "."
        grouped_workflows.setdefault(top_level, []).append(workflow)

    grouped_spreadsheets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rel_path, payload in spreadsheet_payload.items():
        parts = rel_path.split(os.sep)
        top_level = parts[0] if len(parts) > 1 else "."
        grouped_spreadsheets.setdefault(top_level, {})[rel_path] = payload

    for top_level, workflows in grouped_workflows.items():
        target_dir = (
            WORKFLOW_ROOT if top_level == "." else os.path.join(WORKFLOW_ROOT, top_level)
        )
        spreadsheets = grouped_spreadsheets.get(top_level, {})
        export_blob = {
            "workflows": workflows,
            "spreadsheets": spreadsheets,
        }
        write_artifacts(target_dir, export_blob, len(workflows), len(spreadsheets))


if __name__ == "__main__":
    main()
