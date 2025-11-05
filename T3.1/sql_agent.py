# Copyright (c) Microsoft. All rights reserved.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import threading
from typing import Any, Dict, List, Literal, Optional, cast
from urllib.parse import quote_plus

import pandas as pd
import termcolor
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import agentlightning as agl

agl.configure_logger()

logger = agl.configure_logger(name=__name__)


DEFAULT_MYSQL_USER = "root"
DEFAULT_MYSQL_PASSWORD = ""
DEFAULT_MYSQL_HOST = "127.0.0.1"
DEFAULT_MYSQL_PORT = 9030
DEFAULT_MYSQL_DATABASE = "database_main"

DEBUG_PROMPT_DIR = Path("T3/script/prompt/input/V1")
DEBUG_PROMPT_LIMIT = 1
DEBUG_DB_URL: Optional[str] = None
SCHEMA_DIR = Path("T3/data")

DEFAULT_OPENAI_MODEL = "qwen3-coder-plus-2025-09-23"
DEFAULT_OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_API_KEY = "sk-21f31afa708c4c6f9bf6b73585788e41"

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_OUTPUT_PATH = REPO_ROOT / "T3" / "upload" / "dataset_exe_result.json"
MAX_LOGGED_ROWS = 200


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def resolve_db_url(task: Dict[str, Any]) -> Optional[str]:
    db_url = task.get("db_url")
    if db_url:
        return db_url

    database = task.get("db_id") or DEFAULT_MYSQL_DATABASE

    if not database:
        return None

    user = quote_plus(DEFAULT_MYSQL_USER)
    password = quote_plus(DEFAULT_MYSQL_PASSWORD)
    auth = f"{user}:{password}" if DEFAULT_MYSQL_PASSWORD else user

    return (
        f"mysql+pymysql://{auth}@{DEFAULT_MYSQL_HOST}:{DEFAULT_MYSQL_PORT}/{database}"
    )


def load_schema_for_db(db_id: Optional[str]) -> Optional[str]:
    if not db_id:
        logger.warning("No database id provided; unable to load schema.")
        return None

    schema_path = SCHEMA_DIR / f"mschema_{db_id}.json"
    if not schema_path.exists():
        logger.error("Schema file not found for database %s at %s", db_id, schema_path)
        return None

    try:
        with schema_path.open("r", encoding="utf-8") as file:
            schema_payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load schema file %s: %s", schema_path, exc)
        return None

    formatted = {
        "db_id": schema_payload.get("db_id", db_id),
        "tables": schema_payload.get("tables", {}),
    }
    return json.dumps(formatted, ensure_ascii=False, indent=2)


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)


CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasoable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    execution_payload: Any
    execution_error: Optional[str]
    execution_success: bool
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLAgent:

    @staticmethod
    def _infer_db_id(db_uri: str) -> Optional[str]:
        if "://" not in db_uri:
            return None
        tail = db_uri.rsplit("/", 1)[-1]
        if not tail:
            return None
        tail = tail.split("?", 1)[0]
        return tail or None

    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)  # type: ignore
        self.db_uri = db
        self._engine = create_engine(db)
        inferred_db_id = self._infer_db_id(db)
        self.db_schema = db_schema or load_schema_for_db(inferred_db_id)
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        api_key = DEFAULT_OPENAI_API_KEY.strip()
        if not api_key or api_key.upper().startswith("YOUR_API_KEY"):
            raise ValueError(
                "DEFAULT_OPENAI_API_KEY is not configured. Update the constant near the top of T3.1/sql_agent.py "
                "with a valid API key."
            )
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=api_key,
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
            )
        else:
            self.model_name = DEFAULT_OPENAI_MODEL
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or DEFAULT_OPENAI_API_BASE,
                openai_api_key=api_key,
                temperature=0,
                max_retries=1,
                max_tokens=2048,
            )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        if self.db_schema:
            schema_text = self.db_schema
            if len(schema_text) > self.table_info_truncate:
                return schema_text[: self.table_info_truncate] + "\n... (truncated)"
            return schema_text

        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            if self.db_schema:
                if len(self.db_schema) > self.table_info_truncate:
                    return self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                return self.db_schema
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            # FIXME: fallback to create a random trajectory
            result = self.llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def truncate_execuion(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        result: str | None = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):  # type: ignore
            result = match.group(1).strip()  # type: ignore
        return result  # type: ignore

    def write_query(self, state: State) -> State:
        """Generate SQL query to fetch information."""
        prompt: Any = WRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        query = self.parse_query(result) or result.content  # type: ignore

        return {  # type: ignore
            **state,
            "query": query,  # type: ignore
            "num_turns": 1,
            "messages": [*prompt.messages, result],
        }

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        query = state.get("query")
        if not query:
            message = "No query available to execute."
            truncated = self.truncate_execuion(message)
            return {
                **state,
                "execution": truncated,
                "execution_payload": None,
                "execution_error": message,
                "execution_success": False,
            }

        try:
            with self._engine.connect() as connection:
                dataframe = pd.read_sql_query(text(query), connection)
        except Exception as exc:
            error_message = str(exc)
            truncated = self.truncate_execuion(error_message)
            if self.debug:
                termcolor.cprint(truncated, "yellow")
            return {
                **state,
                "execution": truncated,
                "execution_payload": None,
                "execution_error": error_message,
                "execution_success": False,
            }

        sanitized = dataframe.astype(object).where(pd.notnull(dataframe), None)
        payload = sanitized.to_dict(orient="records")
        if isinstance(payload, list) and len(payload) > MAX_LOGGED_ROWS:
            payload = payload[:MAX_LOGGED_ROWS]

        preview_rows = payload[:5] if isinstance(payload, list) else payload
        preview_text = json.dumps(preview_rows, ensure_ascii=False, default=_json_default)
        execution_output = self.truncate_execuion(preview_text if preview_text else "[]")

        if self.debug:
            termcolor.cprint(execution_output, "yellow")

        return {
            **state,
            "execution": execution_output,
            "execution_payload": payload,
            "execution_error": None,
            "execution_success": True,
        }

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        prompt: Any = CHECK_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        res = {  # type: ignore
            **state,
            "feedback": result.content,  # type: ignore
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        return res  # type: ignore

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        prompt: Any = REWRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        rewritten_query = self.parse_query(result)  # type: ignore

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):  # type: ignore
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:  # type: ignore
                if "THE QUERY IS INCORRECT" in last_message.content:  # type: ignore
                    # Both correct and incorrect messages found
                    # See which is the last one
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")  # type: ignore
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")  # type: ignore
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "rewrite_query"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)  # type: ignore
        builder.add_node(self.execute_query)  # type: ignore
        builder.add_node(self.check_query)  # type: ignore
        builder.add_node(self.rewrite_query)  # type: ignore

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("rewrite_query", "execute_query")

        return builder.compile()  # type: ignore


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    engine = None
    try:
        engine = create_engine(database)
        with engine.connect() as connection:
            predicted_df = pd.read_sql_query(text(query), connection)
            ground_truth_df = pd.read_sql_query(text(ground_truth), connection)

        if set(predicted_df.columns) != set(ground_truth_df.columns):
            logger.debug("Column mismatch during evaluation: predicted=%s, ground_truth=%s", predicted_df.columns, ground_truth_df.columns)
            return 0.0

        def _records_counter(df: pd.DataFrame) -> Counter[str]:
            records = df.to_dict(orient="records")
            return Counter(json.dumps(record, sort_keys=True, default=str) for record in records)

        predicted_counter = _records_counter(predicted_df)
        ground_truth_counter = _records_counter(ground_truth_df)

        return 1.0 if predicted_counter == ground_truth_counter else 0.0
    except SQLAlchemyError as exc:
        if raise_on_error:
            raise
        logger.exception("SQLAlchemy error during evaluation: %s", exc)
        return 0.0
    except Exception as exc:
        if raise_on_error:
            raise
        logger.exception("Unexpected error during evaluation: %s", exc)
        return 0.0
    finally:
        if engine is not None:
            engine.dispose()


class LitSQLAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        self.result_output_path = RESULT_OUTPUT_PATH
        self.result_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._results_lock = threading.Lock()
        self._results: List[Dict[str, Any]] = self._load_existing_results()

    def _load_existing_results(self) -> List[Dict[str, Any]]:
        try:
            if not self.result_output_path.exists():
                return []
            with self.result_output_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse existing results at %s: %s", self.result_output_path, exc)
        except OSError as exc:
            logger.warning("Unable to read existing results at %s: %s", self.result_output_path, exc)
        return []

    def _write_results(self) -> None:
        try:
            with self.result_output_path.open("w", encoding="utf-8") as file:
                json.dump(self._results, file, ensure_ascii=False, indent=4, default=_json_default)
        except OSError as exc:
            logger.error("Failed to write inference results to %s: %s", self.result_output_path, exc)

    def _upsert_result(self, entry: Dict[str, Any]) -> None:
        sql_id = entry.get("sql_id")
        if sql_id is not None:
            for index, existing in enumerate(self._results):
                if existing.get("sql_id") == sql_id:
                    self._results[index] = entry
                    return
        self._results.append(entry)

    def _resolve_sql_id(self, task: Dict[str, Any]) -> str:
        sql_id = task.get("sql_id") or task.get("id")
        if isinstance(sql_id, str) and sql_id.strip():
            return sql_id.strip()
        question = task.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()[:32]
        return "unknown_sql_id"

    def _record_inference_result(
        self, task: Dict[str, Any], agent_state: Dict[str, Any], reward: float | None
    ) -> None:
        sql_id = self._resolve_sql_id(task)
        retry_steps = max(agent_state.get("num_turns", 1) - 1, 0)
        execution_success = bool(agent_state.get("execution_success"))
        success = execution_success
        if reward is not None:
            success = success and reward > 0
        error_value: Optional[str] = agent_state.get("execution_error")
        if not success and error_value is None and execution_success and reward is not None and reward <= 0:
            error_value = "Query result did not match ground truth."
        entry = {
            "sql_id": sql_id,
            "sql": agent_state.get("query"),
            "result": agent_state.get("execution_payload"),
            "success": success,
            "error": None if success else error_value,
            "retry_steps": retry_steps,
        }
        with self._results_lock:
            self._upsert_result(entry)
            self._write_results()

    def _record_failed_inference(self, task: Dict[str, Any], error_message: str) -> None:
        entry = {
            "sql_id": self._resolve_sql_id(task),
            "sql": None,
            "result": None,
            "success": False,
            "error": error_message,
            "retry_steps": 0,
        }
        with self._results_lock:
            self._upsert_result(entry)
            self._write_results()

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        verl_config = (
            {"model": llm.model, **llm.sampling_parameters}
            if rollout.mode == "train"
            else {
                "model": llm.model,
                "temperature": (
                    self.val_temperature
                    if self.val_temperature is not None
                    else llm.sampling_parameters.get("temperature", 0.0)
                ),
            }
        )
        endpoint_base = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)  # type: ignore

        ground_truth = task["query"]
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        db_url = resolve_db_url(task)
        if not db_url:
            logger.error("[Rollout %s] Unable to resolve database URL for task %s", rollout_id, task.get("db_id"))
            self._record_failed_inference(task, "Unable to resolve database URL.")
            return None

        schema_override = task.get("db_schema")
        target_db_id = task.get("db_id") or DEFAULT_MYSQL_DATABASE
        schema = schema_override or load_schema_for_db(target_db_id) or "No schema available."

        agent = SQLAgent(
            db_url,
            max_turns=self.max_turns,
            table_info_truncate=self.table_info_truncate,
            execution_truncate=self.execution_truncate,
            debug=False,
            db_schema=schema,
            endpoint=endpoint_base,
            verl_replacement=verl_config,
        ).graph()
        try:
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(  # type: ignore
                {"question": question},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as exc:
            logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {exc}")
            self._record_failed_inference(task, str(exc))
            return None

        logger.info(f"[Rollout {rollout_id}] Generated Query: {result.get('query')}")

        end_time_rollout = time.time()

        query_text = result.get("query")
        reward: float | None
        if query_text:
            reward = evaluate_query(query_text, ground_truth, db_url, raise_on_error=False)
        else:
            reward = None
            logger.warning("[Rollout %s] No query generated for evaluation.", rollout_id)
        logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        self._record_inference_result(task, result, reward)

        end_time_eval = time.time()
        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )

        return reward


def debug_sql_agent():
    prompt_dir = DEBUG_PROMPT_DIR
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory {prompt_dir} does not exist.")

    prompt_files = sorted(prompt_dir.glob("*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in {prompt_dir}.")

    selected_files = prompt_files[:DEBUG_PROMPT_LIMIT]

    prompts: List[Dict[str, str]] = []
    for path in selected_files:
        with path.open("r", encoding="utf-8") as file:
            prompts.append({"name": path.name, "question": file.read().strip()})

    task_meta = {
        "db_url": DEBUG_DB_URL,
        "db_id": DEFAULT_MYSQL_DATABASE,
    }
    db_url = resolve_db_url(task_meta)
    if not db_url:
        raise ValueError("Unable to determine database URL. Update DEBUG_DB_URL or DEFAULT_* constants.")

    debug_schema = load_schema_for_db(DEFAULT_MYSQL_DATABASE)
    if not debug_schema:
        logger.warning("Falling back to empty schema because formatted schema could not be loaded.")
        debug_schema = "No schema available."

    agent = SQLAgent(
        db_url,
        max_turns=5,
        table_info_truncate=4096,
        execution_truncate=4096,
        debug=True,
        db_schema=debug_schema,
    ).graph()

    for prompt in prompts:
        logger.info("[Debug] Processing %s", prompt["name"])
        result = agent.invoke({"question": prompt["question"]}, {"recursion_limit": 100})  # type: ignore
        print("=" * 60)
        print(f"Prompt: {prompt['name']}")
        print("Generated Query:")
        print(result["query"])  # type: ignore
        print()


if __name__ == "__main__":
    debug_sql_agent()