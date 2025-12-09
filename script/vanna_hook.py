from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from vanna.core.tool.models import ToolContext, ToolResult
from vanna.core.lifecycle import LifecycleHook
from vanna.core.storage import Conversation
from vanna.tools import RunSqlTool
from vanna.capabilities.sql_runner import RunSqlToolArgs


SQL_ID_REGISTRY: Dict[str, str] = {}


def _sanitize_identifier(value: str, default: str = "unknown") -> str:
    value = (value or "").strip()
    if not value:
        return default
    sanitized = re.sub(r"[^\w\-]+", "_", value)
    sanitized = sanitized.strip("_")
    return sanitized or default


def _extract_sql_id_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = re.search(r"(sql_\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def _resolve_sql_id(
    conversation_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    sql_text: Optional[str] = None,
) -> str:
    metadata = metadata or {}
    candidate_keys = (
        "sql_id",
        "conversation_sql_id",
        "task_id",
        "task_name",
        "identifier",
    )
    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_identifier(value)

    for source in (conversation_id, sql_text or ""):
        sql_id = _extract_sql_id_from_text(source)
        if sql_id:
            return _sanitize_identifier(sql_id)

    stem = Path(conversation_id).stem if conversation_id else ""
    if stem:
        return _sanitize_identifier(stem)

    return "unknown"


def _lookup_sql_id_from_conversation(conversation: Conversation) -> str:
    for message in reversed(conversation.messages):
        if message.metadata:
            for key in ("sql_id", "conversation_sql_id", "task_id", "task_name"):
                value = message.metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return _sanitize_identifier(value)
        sql_id = _extract_sql_id_from_text(message.content or "")
        if sql_id:
            return _sanitize_identifier(sql_id)
    return _resolve_sql_id(conversation.id)


class TGACRunSqlTool(RunSqlTool):
    """Extend the stock run_sql tool so we can retain the executed SQL string and results."""

    def __init__(
        self,
        output_path: Optional[str] = None,
        output_text_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize TGACRunSqlTool with optional output path for saving results.
        
        Args:
            output_path: Path to JSON file where SQL execution results will be saved.
                        If None, results will not be saved automatically.
            output_text_dir: Directory where plain-text summaries of results will be stored.
                        Each conversation will be saved as <sql_id>.txt. Optional.
        """
        super().__init__(**kwargs)
        self._output_path = Path(output_path).resolve() if output_path else None
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_text_dir = Path(output_text_dir).resolve() if output_text_dir else None
        if self._output_text_dir:
            self._output_text_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        result = await super().execute(context, args)
        
        # Ensure metadata exists
        if not hasattr(result, 'metadata') or result.metadata is None:
            result.metadata = {}
        
        # Persist the executed SQL for downstream consumers.
        try:
            result.metadata.setdefault("sql", args.sql)
        except AttributeError:
            # Fallback for unexpected args shape.
            result.metadata.setdefault("sql", "")
        
        # CRITICAL: Ensure results are stored in metadata
        # RunSqlTool should store results in metadata["results"], but we verify and ensure it
        if result.success and not result.error:
            # If results are missing, this is a problem - log it
            if "results" not in result.metadata or result.metadata.get("results") is None:
                # Try to find results in other possible locations
                # Check if result has a direct results attribute (some implementations)
                if hasattr(result, 'results') and result.results is not None:
                    result.metadata["results"] = result.results
                else:
                    # Log warning - this should not happen if RunSqlTool is working correctly
                    print(
                        f"[TGACRunSqlTool] Warning: SQL execution succeeded but no results in metadata. "
                        f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}"
                    )
        
        # NEW: Diagnose empty results and search for similar failed cases
        if result.success and not result.error:
            results = result.metadata.get("results")
            if results is None or (isinstance(results, list) and len(results) == 0):
                # Query returned 0 rows - perform diagnostics
                diagnostics = await self._diagnose_empty_result(args.sql, context)
                result.metadata["diagnostics"] = diagnostics
                
                # Build diagnostic message for LLM
                diagnostic_message = self._format_diagnostic_message(diagnostics)
                
                # Note: Failed cases search is now handled in prompt generation phase
                
                # Update result_for_llm to include diagnostic information
                original_result = result.result_for_llm or ""
                if diagnostic_message:
                    result.result_for_llm = (
                        f"{original_result}\n\n"
                        f"⚠️ 诊断信息：查询返回 0 行。\n{diagnostic_message}"
                    )
                
                # If diagnostics found issues, mark for retry
                if diagnostics.get("has_errors", False):
                    result.metadata["needs_retry"] = True
                    result.metadata["diagnostic_suggestions"] = diagnostics.get("suggestions", [])
        
        # Save results immediately after execution
        if self._output_path or self._output_text_dir:
            await self._save_result(context, args.sql, result)
        
        return result

    async def _save_result(self, context: ToolContext, sql_text: str, result: ToolResult) -> None:
        """Save SQL execution result to the output file and optional txt summary."""
        try:
            conversation_id = context.conversation_id
            
            # Extract results from metadata
            metadata = result.metadata or {}
            sql_id = _resolve_sql_id(conversation_id, metadata, sql_text)
            metadata["sql_id"] = sql_id
            SQL_ID_REGISTRY[conversation_id] = sql_id
            raw_results = metadata.get("results")
            
            # Try other possible locations
            if raw_results is None:
                for key in ["data", "rows", "query_results", "execution_results"]:
                    if key in metadata:
                        raw_results = metadata[key]
                        break
            
            # Clean results for JSON
            cleaned_results = self._clean_for_json(raw_results)
            if isinstance(cleaned_results, list) and len(cleaned_results) == 0:
                cleaned_results = None
            
            # Build entry
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
            
            text_content: Optional[str] = None
            if hasattr(result, "result_for_llm") and result.result_for_llm:
                text_content = str(result.result_for_llm)
            elif cleaned_results is not None:
                text_content = json.dumps(cleaned_results, ensure_ascii=False, indent=4)
            elif result.error:
                text_content = f"执行失败：{result.error}"

            # Save to file
            async with self._lock:
                if self._output_path:
                    data = self._load_entries()
                    updated = False
                    for idx, existing in enumerate(data):
                        if existing.get("sql_id") == sql_id:
                            data[idx] = entry
                            updated = True
                            break
                    if not updated:
                        data.append(entry)
                    self._write_entries(data)

                if self._output_text_dir and text_content is not None:
                    txt_path = self._output_text_dir / f"{sql_id}.txt"
                    existing = ""
                    if txt_path.exists():
                        existing = txt_path.read_text(encoding="utf-8").strip()
                    combined = text_content.strip()
                    if existing:
                        combined = f"{existing}\n\n=== 工具执行输出 ===\n{combined}"
                    txt_path.write_text(combined + "\n", encoding="utf-8")
                
            if result.success and cleaned_results is not None:
                print(f"[TGACRunSqlTool] Saved result for sql_id={sql_id} ({len(cleaned_results)} rows)")
            elif result.success:
                print(f"[TGACRunSqlTool] Warning: Saved entry for sql_id={sql_id} but result is None")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[TGACRunSqlTool] Error saving result: {exc}")
            import traceback
            traceback.print_exc()

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load existing entries from the output file."""
        if not self._output_path or not self._output_path.exists():
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
        """Write entries to the output file."""
        if not self._output_path:
            return
        with self._output_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=4)
            fh.write("\n")

    def _clean_for_json(self, value: Any) -> Any:
        """Clean value for JSON serialization."""
        if isinstance(value, dict):
            return {k: self._clean_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean_for_json(v) for v in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    async def _diagnose_empty_result(self, sql: str, context: ToolContext) -> Dict[str, Any]:
        """
        Diagnose why a query returned 0 rows.
        Checks for common issues like incorrect table/column names, date format mismatches, etc.
        Now includes actual database validation for tables/columns and intelligent date format detection.
        """
        diagnostics = {
            "has_errors": False,
            "warnings": [],
            "suggestions": [],
            "table_checks": {},
            "column_checks": {},
            "date_format_analysis": {},
        }
        
        try:
            # Extract table names from SQL
            table_pattern = r'\bFROM\s+(\w+)\b|\bJOIN\s+(\w+)\b'
            table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
            tables = [t[0] or t[1] for t in table_matches if t[0] or t[1]]
            
            # Extract column references (simplified - may miss some cases)
            column_pattern = r'\b(\w+)\.(\w+)\b'
            column_matches = re.findall(column_pattern, sql)
            
            # NEW: Validate table names against actual database
            if tables and hasattr(self, 'sql_runner') and self.sql_runner:
                table_validation = await self._validate_tables(tables)
                diagnostics["table_checks"] = table_validation
                if table_validation.get("invalid_tables"):
                    diagnostics["has_errors"] = True
                    diagnostics["warnings"].append(
                        f"以下表名在数据库中不存在: {', '.join(table_validation['invalid_tables'])}"
                    )
            
            # NEW: Validate column names against actual database
            if column_matches and hasattr(self, 'sql_runner') and self.sql_runner:
                column_validation = await self._validate_columns(column_matches)
                diagnostics["column_checks"] = column_validation
                if column_validation.get("invalid_columns"):
                    diagnostics["has_errors"] = True
                    diagnostics["warnings"].append(
                        f"以下列引用在数据库中不存在: {', '.join([f'{t}.{c}' for t, c in column_validation['invalid_columns']])}"
                    )
            
            # NEW: Intelligent date format detection
            date_format_analysis = await self._analyze_date_formats(sql, tables)
            if date_format_analysis:
                diagnostics["date_format_analysis"] = date_format_analysis
                if date_format_analysis.get("format_mismatch"):
                    diagnostics["has_errors"] = True
                    diagnostics["warnings"].append(
                        f"日期格式可能不匹配: 查询中使用 {date_format_analysis.get('query_format', '未知')}, "
                        f"但数据库中实际格式为 {date_format_analysis.get('actual_format', '未知')}"
                    )
            
            # Check for common date format issues (fallback if analysis failed)
            date_patterns = [
                r"DATE_FORMAT\s*\(",
                r"DATE_SUB\s*\(",
                r"STR_TO_DATE\s*\(",
                r"'\d{4}-\d{2}-\d{2}'",
                r"'\d{8}'",
            ]
            has_date_functions = any(re.search(pattern, sql, re.IGNORECASE) for pattern in date_patterns)
            if has_date_functions and not diagnostics["date_format_analysis"]:
                diagnostics["warnings"].append(
                    "Query contains date functions - verify date format matches database format"
                )
            
            # Check for very restrictive WHERE conditions
            where_pattern = r'\bWHERE\s+.*?(?:AND|OR)'
            where_matches = re.findall(where_pattern, sql, re.IGNORECASE)
            has_multiple_conditions = len(where_matches) > 2
            
            # Build diagnostic information (fallback if validation not performed)
            if tables and not diagnostics.get("table_checks"):
                diagnostics["table_checks"] = {
                    "tables_found": tables,
                    "note": "Verify these table names exist in the database schema"
                }
            
            if column_matches and not diagnostics.get("column_checks"):
                diagnostics["column_checks"] = {
                    "column_references": list(set(column_matches)),
                    "note": "Verify these table.column references are correct"
                }
            
            # Add warnings and suggestions
            if has_date_functions and not diagnostics["date_format_analysis"]:
                diagnostics["suggestions"].append(
                    "Check if date format in query matches the database date format (e.g., YYYYMMDD vs YYYY-MM-DD)"
                )
            
            if has_multiple_conditions:
                diagnostics["warnings"].append(
                    "Query has multiple WHERE conditions - may be too restrictive"
                )
                diagnostics["suggestions"].append(
                    "Consider relaxing WHERE conditions or checking if data exists for each condition separately"
                )
            
            # General suggestions
            if not diagnostics["suggestions"]:
                diagnostics["suggestions"].extend([
                    "Verify that the tables and columns in the query exist in the database schema",
                    "Check if the WHERE conditions are too restrictive",
                    "Verify date formats match the database format",
                    "Consider checking if data exists for each condition separately",
                ])
            
            # Mark as having potential errors if we found issues
            if diagnostics["warnings"] or (tables and len(tables) > 0):
                diagnostics["has_errors"] = True
            
        except Exception as exc:
            print(f"[TGACRunSqlTool] Error during diagnostics: {exc}")
            import traceback
            traceback.print_exc()
            diagnostics["error"] = str(exc)
        
        return diagnostics

    async def _validate_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """Validate table names against the actual database schema."""
        validation = {
            "tables_found": table_names,
            "valid_tables": [],
            "invalid_tables": [],
            "validation_performed": False,
        }
        
        try:
            if not hasattr(self, 'sql_runner') or not self.sql_runner:
                return validation
            
            # Query information_schema to check if tables exist
            # For MySQL/StarRocks, use information_schema.tables
            validation_query = """
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME IN ({})
            """.format(','.join([f"'{t}'" for t in table_names]))
            
            try:
                # Use the underlying database connection directly
                # MySQLRunner typically has a connection or engine attribute
                connection = None
                if hasattr(self.sql_runner, 'connection'):
                    connection = self.sql_runner.connection
                elif hasattr(self.sql_runner, 'engine'):
                    connection = self.sql_runner.engine.connect()
                elif hasattr(self.sql_runner, '_connection'):
                    connection = self.sql_runner._connection
                elif hasattr(self.sql_runner, '_engine'):
                    connection = self.sql_runner._engine.connect()
                
                if connection:
                    # Execute query using direct connection
                    def execute_query():
                        if hasattr(connection, 'execute'):
                            # SQLAlchemy connection - need to use text() for raw SQL
                            from sqlalchemy import text
                            result = connection.execute(text(validation_query))
                            return [row[0] for row in result]
                        elif hasattr(connection, 'cursor'):
                            # PyMySQL connection
                            cursor = connection.cursor()
                            cursor.execute(validation_query)
                            return [row[0] for row in cursor.fetchall()]
                        return None
                    
                    # Run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    existing_tables = await loop.run_in_executor(None, execute_query)
                    
                    if existing_tables:
                        existing_tables = [str(t).lower() for t in existing_tables]
                        table_names_lower = [t.lower() for t in table_names]
                        
                        validation["valid_tables"] = [t for t in table_names if t.lower() in existing_tables]
                        validation["invalid_tables"] = [t for t in table_names if t.lower() not in existing_tables]
                        validation["validation_performed"] = True
                    else:
                        # If query returned no results, assume all tables are invalid
                        validation["invalid_tables"] = table_names
                        validation["validation_performed"] = True
                else:
                    # Fallback: cannot access connection, skip validation
                    print("[TGACRunSqlTool] Cannot access database connection for table validation")
                    return validation
                    
            except Exception as query_exc:
                print(f"[TGACRunSqlTool] Error validating tables: {query_exc}")
                import traceback
                traceback.print_exc()
                validation["validation_error"] = str(query_exc)
                
        except Exception as exc:
            print(f"[TGACRunSqlTool] Error in _validate_tables: {exc}")
            validation["error"] = str(exc)
        
        return validation

    async def _validate_columns(self, column_references: List[tuple]) -> Dict[str, Any]:
        """Validate table.column references against the actual database schema."""
        validation = {
            "column_references": column_references,
            "valid_columns": [],
            "invalid_columns": [],
            "validation_performed": False,
        }
        
        try:
            if not hasattr(self, 'sql_runner') or not self.sql_runner:
                return validation
            
            # Group columns by table
            table_columns = {}
            for table, column in column_references:
                if table not in table_columns:
                    table_columns[table] = []
                table_columns[table].append(column)
            
            # Validate each table's columns
            for table, columns in table_columns.items():
                if not columns:
                    continue
                
                validation_query = """
                    SELECT COLUMN_NAME 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = '{}'
                    AND COLUMN_NAME IN ({})
                """.format(table, ','.join([f"'{c}'" for c in columns]))
                
                try:
                    # Use the underlying database connection directly
                    connection = None
                    if hasattr(self.sql_runner, 'connection'):
                        connection = self.sql_runner.connection
                    elif hasattr(self.sql_runner, 'engine'):
                        connection = self.sql_runner.engine.connect()
                    elif hasattr(self.sql_runner, '_connection'):
                        connection = self.sql_runner._connection
                    elif hasattr(self.sql_runner, '_engine'):
                        connection = self.sql_runner._engine.connect()
                    
                    if connection:
                        # Execute query using direct connection
                        def execute_query():
                            if hasattr(connection, 'execute'):
                                # SQLAlchemy connection - need to use text() for raw SQL
                                from sqlalchemy import text
                                result = connection.execute(text(validation_query))
                                return [row[0] for row in result]
                            elif hasattr(connection, 'cursor'):
                                # PyMySQL connection
                                cursor = connection.cursor()
                                cursor.execute(validation_query)
                                return [row[0] for row in cursor.fetchall()]
                            return None
                        
                        # Run in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        existing_columns = await loop.run_in_executor(None, execute_query)
                        
                        if existing_columns:
                            existing_columns = [str(c).lower() for c in existing_columns]
                            
                            for col in columns:
                                if col.lower() in existing_columns:
                                    validation["valid_columns"].append((table, col))
                                else:
                                    validation["invalid_columns"].append((table, col))
                        else:
                            # If query returned no results, assume all columns are invalid
                            for col in columns:
                                validation["invalid_columns"].append((table, col))
                    else:
                        # Cannot access connection, skip validation for this table
                        continue
                            
                except Exception as query_exc:
                    print(f"[TGACRunSqlTool] Error validating columns for table {table}: {query_exc}")
                    # If validation fails, mark columns as potentially invalid
                    for col in columns:
                        validation["invalid_columns"].append((table, col))
            
            validation["validation_performed"] = True
            
        except Exception as exc:
            print(f"[TGACRunSqlTool] Error in _validate_columns: {exc}")
            validation["error"] = str(exc)
        
        return validation

    async def _analyze_date_formats(self, sql: str, tables: List[str]) -> Dict[str, Any]:
        """Analyze date formats in the query and compare with actual database date formats."""
        analysis = {
            "query_format": None,
            "actual_format": None,
            "format_mismatch": False,
            "analysis_performed": False,
        }
        
        try:
            if not tables or not hasattr(self, 'sql_runner') or not self.sql_runner:
                return analysis
            
            # Extract date-related conditions from SQL
            date_patterns = {
                'YYYY-MM-DD': r"'\d{4}-\d{2}-\d{2}'",
                'YYYYMMDD': r"'\d{8}'",
                'DATE_FORMAT': r"DATE_FORMAT\s*\([^,]+,\s*'([^']+)'\)",
                'STR_TO_DATE': r"STR_TO_DATE\s*\([^,]+,\s*'([^']+)'\)",
            }
            
            query_format = None
            for format_name, pattern in date_patterns.items():
                if re.search(pattern, sql, re.IGNORECASE):
                    query_format = format_name
                    break
            
            if not query_format:
                return analysis
            
            analysis["query_format"] = query_format
            
            # Try to detect actual date format from database
            # Sample a date column from the first table
            for table in tables:
                try:
                    # Find a date/datetime column in the table
                    column_query = """
                        SELECT COLUMN_NAME, DATA_TYPE 
                        FROM information_schema.COLUMNS 
                        WHERE TABLE_SCHEMA = DATABASE()
                        AND TABLE_NAME = '{}'
                        AND DATA_TYPE IN ('date', 'datetime', 'timestamp', 'varchar', 'char', 'string')
                        LIMIT 1
                    """.format(table)
                    
                    # Use the underlying database connection directly
                    connection = None
                    if hasattr(self.sql_runner, 'connection'):
                        connection = self.sql_runner.connection
                    elif hasattr(self.sql_runner, 'engine'):
                        connection = self.sql_runner.engine.connect()
                    elif hasattr(self.sql_runner, '_connection'):
                        connection = self.sql_runner._connection
                    elif hasattr(self.sql_runner, '_engine'):
                        connection = self.sql_runner._engine.connect()
                    
                    if connection:
                        # Execute column query using direct connection
                        def execute_column_query():
                            if hasattr(connection, 'execute'):
                                # SQLAlchemy connection - need to use text() for raw SQL
                                from sqlalchemy import text
                                result = connection.execute(text(column_query))
                                rows = list(result)
                                return rows[0] if rows else None
                            elif hasattr(connection, 'cursor'):
                                # PyMySQL connection
                                cursor = connection.cursor()
                                cursor.execute(column_query)
                                return cursor.fetchone()
                            return None
                        
                        loop = asyncio.get_event_loop()
                        row = await loop.run_in_executor(None, execute_column_query)
                        
                        if row:
                            column_name = row[0] if isinstance(row, (list, tuple)) else row
                            data_type = row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else None
                            
                            # Sample actual date values from the table
                            sample_query = f"SELECT {column_name} FROM {table} WHERE {column_name} IS NOT NULL LIMIT 5"
                            
                            try:
                                # Execute sample query using direct connection
                                def execute_sample_query():
                                    if hasattr(connection, 'execute'):
                                        # SQLAlchemy connection - need to use text() for raw SQL
                                        from sqlalchemy import text
                                        result = connection.execute(text(sample_query))
                                        return [row[0] for row in result]
                                    elif hasattr(connection, 'cursor'):
                                        # PyMySQL connection
                                        cursor = connection.cursor()
                                        cursor.execute(sample_query)
                                        return [row[0] for row in cursor.fetchall()]
                                    return None
                                
                                sample_dates = await loop.run_in_executor(None, execute_sample_query)
                                
                                if sample_dates:
                                    # Analyze the format of sampled dates
                                    actual_format = self._detect_date_format_from_samples(sample_dates)
                                    
                                    if actual_format:
                                        analysis["actual_format"] = actual_format
                                        analysis["analysis_performed"] = True
                                        
                                        # Check for format mismatch
                                        if query_format == 'YYYY-MM-DD' and actual_format == 'YYYYMMDD':
                                            analysis["format_mismatch"] = True
                                        elif query_format == 'YYYYMMDD' and actual_format == 'YYYY-MM-DD':
                                            analysis["format_mismatch"] = True
                                        
                                        break
                            except Exception:
                                # If sampling fails, continue to next table
                                continue
                    else:
                        # Cannot access connection, continue to next table
                        continue
                            
                except Exception:
                    # If column detection fails, continue to next table
                    continue
            
            return analysis
            
        except Exception as exc:
            print(f"[TGACRunSqlTool] Error in _analyze_date_formats: {exc}")
            analysis["error"] = str(exc)
            return analysis

    def _detect_date_format_from_samples(self, samples: List[Any]) -> Optional[str]:
        """Detect date format from sample values."""
        if not samples:
            return None
        
        # Convert samples to strings
        sample_strs = [str(s) for s in samples if s is not None]
        if not sample_strs:
            return None
        
        # Check for common patterns
        for sample in sample_strs[:3]:  # Check first 3 samples
            sample = sample.strip()
            
            # YYYYMMDD format (8 digits)
            if re.match(r'^\d{8}$', sample):
                return 'YYYYMMDD'
            
            # YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}', sample):
                return 'YYYY-MM-DD'
            
            # YYYY/MM/DD format
            if re.match(r'^\d{4}/\d{2}/\d{2}', sample):
                return 'YYYY/MM/DD'
            
            # YYYYMMDDHHMMSS format (14 digits)
            if re.match(r'^\d{14}$', sample):
                return 'YYYYMMDDHHMMSS'
        
        return None

    def _format_diagnostic_message(self, diagnostics: Dict[str, Any]) -> str:
        """Format diagnostic information into a readable message for the LLM."""
        messages = []
        
        if diagnostics.get("warnings"):
            messages.append("警告：")
            for warning in diagnostics["warnings"]:
                messages.append(f"  - {warning}")
        
        # Enhanced table validation results
        table_checks = diagnostics.get("table_checks", {})
        if table_checks.get("validation_performed"):
            if table_checks.get("invalid_tables"):
                messages.append(f"\n❌ 无效的表名：{', '.join(table_checks['invalid_tables'])}")
                messages.append("  这些表在数据库中不存在，请检查表名是否正确")
            if table_checks.get("valid_tables"):
                messages.append(f"\n✅ 有效的表名：{', '.join(table_checks['valid_tables'])}")
        elif table_checks.get("tables_found"):
            tables = table_checks["tables_found"]
            messages.append(f"\n检测到的表：{', '.join(tables)}")
            messages.append("  建议：请验证这些表名在数据库 schema 中是否存在")
        
        # Enhanced column validation results
        column_checks = diagnostics.get("column_checks", {})
        if column_checks.get("validation_performed"):
            if column_checks.get("invalid_columns"):
                invalid_cols = [f"{t}.{c}" for t, c in column_checks["invalid_columns"][:5]]
                messages.append(f"\n❌ 无效的列引用：{', '.join(invalid_cols)}")
                if len(column_checks["invalid_columns"]) > 5:
                    messages.append(f"  ... 还有 {len(column_checks['invalid_columns']) - 5} 个无效列引用")
                messages.append("  这些列在数据库中不存在，请检查列名是否正确")
            if column_checks.get("valid_columns"):
                valid_cols = [f"{t}.{c}" for t, c in column_checks["valid_columns"][:3]]
                messages.append(f"\n✅ 有效的列引用：{', '.join(valid_cols)}")
                if len(column_checks["valid_columns"]) > 3:
                    messages.append(f"  ... 还有 {len(column_checks['valid_columns']) - 3} 个有效列引用")
        elif column_checks.get("column_references"):
            columns = column_checks["column_references"]
            messages.append(f"\n检测到的列引用：{', '.join([f'{t}.{c}' for t, c in columns[:5]])}")
            if len(columns) > 5:
                messages.append(f"  ... 还有 {len(columns) - 5} 个列引用")
            messages.append("  建议：请验证这些表.列引用是否正确")
        
        # Date format analysis results
        date_analysis = diagnostics.get("date_format_analysis", {})
        if date_analysis.get("analysis_performed"):
            query_format = date_analysis.get("query_format", "未知")
            actual_format = date_analysis.get("actual_format", "未知")
            if date_analysis.get("format_mismatch"):
                messages.append(f"\n⚠️ 日期格式不匹配：")
                messages.append(f"  查询中使用：{query_format}")
                messages.append(f"  数据库中实际格式：{actual_format}")
                messages.append("  请调整查询中的日期格式以匹配数据库格式")
            else:
                messages.append(f"\n✅ 日期格式检查：查询格式 {query_format} 与数据库格式 {actual_format} 匹配")
        
        if diagnostics.get("suggestions"):
            messages.append("\n建议：")
            for suggestion in diagnostics["suggestions"][:3]:  # Limit to top 3 suggestions
                messages.append(f"  - {suggestion}")
        
        return "\n".join(messages) if messages else "查询返回 0 行，请检查查询逻辑是否正确。"


class FinalResponseSaverHook(LifecycleHook):
    """在消息结束后保存 Agent 最终输出。"""

    def __init__(self, output_dir: Optional[str]):
        self._output_dir = Path(output_dir).resolve() if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def after_message(self, conversation: Conversation) -> None:  # type: ignore[override]
        if not self._output_dir:
            return

        if not conversation.messages:
            return

        # 找到最后一条助手回复
        last_assistant = None
        for message in reversed(conversation.messages):
            if message.role == "assistant" and message.content:
                last_assistant = message
                break

        if not last_assistant:
            return

        content = last_assistant.content.strip()
        if not content:
            return

        sql_id = SQL_ID_REGISTRY.get(conversation.id)
        if not sql_id:
            sql_id = _lookup_sql_id_from_conversation(conversation)
            SQL_ID_REGISTRY[conversation.id] = sql_id

        file_path = self._output_dir / f"{sql_id}.txt"

        async with self._lock:
            existing = ""
            if file_path.exists():
                existing = file_path.read_text(encoding="utf-8").strip()

            final_section = f"=== 最终总结 ===\n{content}"
            if existing:
                combined = f"{existing}\n\n{final_section}"
            else:
                combined = final_section

            file_path.write_text(combined + "\n", encoding="utf-8")
