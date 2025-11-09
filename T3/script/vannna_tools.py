import os
import json
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

import chromadb
from pydantic import BaseModel, Field, field_validator

from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.tool import Tool, ToolContext
from vanna.core.tool.models import ToolResult
from vanna.core.user import RequestContext, User, UserResolver
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.integrations.openai import OpenAILlmService

class UnifiedSearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to find database schema information or domain knowledge.")
    search_type: str = Field("both", description="Type of search: 'schema' for database schema only, 'domain' for domain knowledge only, 'both' to search both (default).")
    schema_results: int = Field(3, description="Maximum number of schema results to return.")
    domain_results: int = Field(3, description="Maximum number of domain knowledge results to return.")


class SearchFailedCasesToolInput(BaseModel):
    query: str = Field(..., description="The current user question or SQL problem description to find similar failed cases. Use the full question text or key phrases from the question.")
    n_results: int = Field(2, description="Maximum number of failed cases to return (default: 2).")


class UnifiedSearchTool(Tool):
    """
    A unified search tool that can search both database schema information and domain knowledge.
    This tool combines the functionality of schema search and domain knowledge search into a single interface.
    """
    name: str = "search"
    description: str = (
        "A unified search tool that searches both database schema information and domain knowledge. "
        "Use this tool to find information about database tables, columns, relationships, or business concepts. "
        "The tool automatically searches both schema and domain knowledge sources and returns the most relevant results. "
        "Examples: 'user login information', 'tables related to players', '付费用户 definition', "
        "'columns in the dws_jordass_device_login_di table', '竞品业务 meaning'."
    )
    
    SCHEMA_COLLECTION_NAME = "schema_knowledge"
    DOMAIN_COLLECTION_NAME = "domain_knowledge"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chroma_client = None
        self.schema_collection = None
        self.domain_collection = None
    
    def get_args_schema(self) -> BaseModel:
        return UnifiedSearchToolInput
    
    def _get_schema_collection(self):
        """Initializes and returns the ChromaDB schema_knowledge collection."""
        if self.schema_collection is None:
            try:
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")
                
                if self.chroma_client is None:
                    print(f"[UnifiedSearchTool] Connecting to ChromaDB for schema search at: {chroma_db_path}")
                    self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                
                try:
                    self.schema_collection = self.chroma_client.get_collection(name=self.SCHEMA_COLLECTION_NAME)
                    count = self.schema_collection.count()
                    if count > 0:
                        print(f"[UnifiedSearchTool] Connected to schema collection '{self.SCHEMA_COLLECTION_NAME}' (contains {count} documents)")
                    else:
                        print(
                            f"[UnifiedSearchTool] Warning: Schema collection '{self.SCHEMA_COLLECTION_NAME}' exists but is empty. "
                            f"Please run 'python T3/script/ingest_schema.py' to populate it."
                        )
                except Exception:
                    self.schema_collection = self.chroma_client.get_or_create_collection(name=self.SCHEMA_COLLECTION_NAME)
                    print(
                        f"[UnifiedSearchTool] Warning: Schema collection '{self.SCHEMA_COLLECTION_NAME}' not found. Created it but it is currently empty. "
                        f"Please run 'python T3/script/ingest_schema.py' to populate it."
                    )
            except Exception as e:
                print(f"[UnifiedSearchTool] Error connecting to schema collection: {e}")
                import traceback
                traceback.print_exc()
                return None
        return self.schema_collection
    
    def _get_domain_collection(self):
        """Initializes and returns the ChromaDB domain_knowledge collection."""
        if self.domain_collection is None:
            try:
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")
                
                if self.chroma_client is None:
                    print(f"[UnifiedSearchTool] Connecting to ChromaDB for domain knowledge search at: {chroma_db_path}")
                    self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                
                try:
                    self.domain_collection = self.chroma_client.get_collection(name=self.DOMAIN_COLLECTION_NAME)
                    count = self.domain_collection.count()
                    if count > 0:
                        print(f"[UnifiedSearchTool] Connected to domain collection '{self.DOMAIN_COLLECTION_NAME}' (contains {count} documents)")
                    else:
                        print(
                            f"[UnifiedSearchTool] Warning: Domain collection '{self.DOMAIN_COLLECTION_NAME}' exists but is empty. "
                            f"Please run 'python T3/script/ingest_common_knowledge.py' to populate it."
                        )
                except Exception:
                    self.domain_collection = self.chroma_client.get_or_create_collection(name=self.DOMAIN_COLLECTION_NAME)
                    print(
                        f"[UnifiedSearchTool] Warning: Domain collection '{self.DOMAIN_COLLECTION_NAME}' not found. Created it but it is currently empty. "
                        f"Please run 'python T3/script/ingest_common_knowledge.py' to populate it."
                    )
            except Exception as e:
                print(f"[UnifiedSearchTool] Error connecting to domain collection: {e}")
                import traceback
                traceback.print_exc()
                return None
        return self.domain_collection
    
    async def _search_schema(self, query: str, n_results: int) -> Dict[str, Any]:
        """Search schema information."""
        collection = self._get_schema_collection()
        if collection is None:
            return {
                "success": False,
                "message": "Could not connect to schema database. Please run 'python T3/script/ingest_schema.py' first.",
                "results": []
            }
        
        try:
            # 增加返回结果数量，以便后续按类型分组和筛选
            search_n_results = min(n_results * 3, 15)
            
            results = collection.query(
                query_texts=[query],
                n_results=search_n_results
            )
            
            documents_list = results.get("documents") or [[]]
            metadatas_list = results.get("metadatas") or [[]]
            distances_list = results.get("distances") or [[]]
            ids_list = results.get("ids") or [[]]
            
            documents = documents_list[0] if documents_list else []
            metadatas = metadatas_list[0] if metadatas_list else []
            distances = distances_list[0] if distances_list else []
            ids = ids_list[0] if ids_list else []
            
            if not documents:
                return {
                    "success": True,
                    "message": "No relevant schema information found.",
                    "results": []
                }
            
            # 按文档类型分组和排序
            table_results = []
            column_results = []
            toon_results = []
            relationship_results = []
            
            for idx, document in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                doc_id = ids[idx] if idx < len(ids) else None
                doc_type = meta.get("type", "unknown")
                
                result_item = {
                    "document": document,
                    "metadata": meta,
                    "distance": distance,
                    "id": doc_id,
                }
                
                if doc_type == "table":
                    table_results.append(result_item)
                elif doc_type == "column":
                    column_results.append(result_item)
                elif doc_type == "table_toon":
                    toon_results.append(result_item)
                elif doc_type == "relationship":
                    relationship_results.append(result_item)
            
            # 按距离排序
            table_results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
            column_results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
            relationship_results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
            
            # 格式化结果
            formatted_results = []
            result_count = 0
            
            # 优先返回表级结果
            for item in table_results[:n_results]:
                result_count += 1
                meta = item["metadata"]
                distance = item["distance"]
                table_name = meta.get("table_name", "")
                header = f"表信息 {result_count}:"
                similarity = f"相似度: {1 - distance:.3f}" if isinstance(distance, (float, int)) else ""
                
                formatted_results.append(
                    f"{header}\n"
                    f"表名: {table_name}\n"
                    f"{similarity}\n"
                    f"{item['document']}"
                )
            
            return {
                "success": True,
                "message": f"Found {len(documents)} schema documents.",
                "results": formatted_results
            }
        except Exception as e:
            print(f"[UnifiedSearchTool] Error during schema search: {e}")
            return {
                "success": False,
                "message": f"An error occurred during schema search: {e}",
                "results": []
            }
    
    async def _search_domain(self, query: str, n_results: int) -> Dict[str, Any]:
        """Search domain knowledge."""
        collection = self._get_domain_collection()
        if collection is None:
            return {
                "success": False,
                "message": "Could not connect to domain knowledge database. Please run 'python T3/script/ingest_common_knowledge.py' first.",
                "results": []
            }
        
        try:
            results = collection.query(query_texts=[query], n_results=n_results)
            
            documents_list = results.get("documents") or [[]]
            metadatas_list = results.get("metadatas") or [[]]
            distances_list = results.get("distances") or [[]]
            
            documents = documents_list[0]
            metadatas = metadatas_list[0] if metadatas_list else []
            distances = distances_list[0] if distances_list else []
            
            if not documents:
                return {
                    "success": True,
                    "message": "No matching domain knowledge found.",
                    "results": []
                }
            
            formatted_results = []
            for idx, document in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                annotations = []
                if meta:
                    annotations.append(f"metadata={meta}")
                if isinstance(distance, (float, int)):
                    annotations.append(f"distance={distance:.4f}")
                prefix = f"Result {idx + 1}:"
                if annotations:
                    formatted_results.append(f"{prefix}\n{' | '.join(annotations)}\n{document}")
                else:
                    formatted_results.append(f"{prefix}\n{document}")
            
            return {
                "success": True,
                "message": f"Found {len(documents)} domain knowledge entries.",
                "results": formatted_results
            }
        except Exception as e:
            print(f"[UnifiedSearchTool] Error during domain search: {e}")
            return {
                "success": False,
                "message": f"An error occurred during domain search: {e}",
                "results": []
            }
    
    async def execute(self, context: ToolContext, args: UnifiedSearchToolInput) -> ToolResult:
        """Executes unified search across schema and/or domain knowledge."""
        search_type = args.search_type.lower() if args.search_type else "both"
        
        print(f"[UnifiedSearchTool] Executing search with query: '{args.query}', type: '{search_type}'")
        
        schema_results = []
        domain_results = []
        schema_success = True
        domain_success = True
        
        # 搜索schema
        if search_type in ("schema", "both"):
            schema_data = await self._search_schema(args.query, args.schema_results)
            schema_results = schema_data.get("results", [])
            schema_success = schema_data.get("success", False)
            if not schema_success:
                schema_results = [f"[Schema Search Error] {schema_data.get('message', 'Unknown error')}"]
        
        # 搜索domain knowledge
        if search_type in ("domain", "both"):
            domain_data = await self._search_domain(args.query, args.domain_results)
            domain_results = domain_data.get("results", [])
            domain_success = domain_data.get("success", False)
            if not domain_success:
                domain_results = [f"[Domain Search Error] {domain_data.get('message', 'Unknown error')}"]
        
        # 合并结果
        formatted_parts = []
        
        if schema_results:
            formatted_parts.append("## [Schema] 数据库结构信息\n")
            formatted_parts.extend(schema_results)
        
        if domain_results:
            if formatted_parts:
                formatted_parts.append("\n---\n")
            formatted_parts.append("## [Domain Knowledge] 领域知识\n")
            formatted_parts.extend(domain_results)
        
        if not formatted_parts:
            result_text = "No relevant information found in either schema or domain knowledge."
        else:
            result_text = "\n\n".join(formatted_parts)
        
        # 确定整体成功状态
        overall_success = True
        if search_type == "schema":
            overall_success = schema_success
        elif search_type == "domain":
            overall_success = domain_success
        else:  # both
            overall_success = schema_success or domain_success  # 至少一个成功即可
        
        return ToolResult(
            success=overall_success,
            result_for_llm=result_text,
            metadata={
                "query": args.query,
                "search_type": search_type,
                "schema_success": schema_success,
                "domain_success": domain_success,
                "schema_results_count": len(schema_results),
                "domain_results_count": len(domain_results),
            },
        )


class SearchFailedCasesTool(Tool):
    """
    A tool to search for similar failed SQL cases (negative examples) from past executions.
    Use this tool to find relevant failure patterns that should be avoided when generating SQL.
    """
    name: str = "search_failed_cases"
    description: str = (
        "Searches for similar failed SQL cases from past executions to learn from past mistakes. "
        "Use this tool BEFORE generating SQL when the current question is similar to previous failed cases, "
        "or AFTER SQL execution fails to understand what went wrong. "
        "The tool returns failed cases with their questions, error SQL, and error reasons to help avoid similar errors. "
        "This is especially useful when you encounter questions about similar topics, metrics, or table combinations that have failed before."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collection = None

    def get_args_schema(self) -> BaseModel:
        return SearchFailedCasesToolInput

    def _get_collection(self):
        """Initializes and returns the ChromaDB collection for failed cases."""
        if self.collection is None:
            try:
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")
                print(f"[SearchFailedCasesTool] Connecting to ChromaDB at: {chroma_db_path}")
                client = chromadb.PersistentClient(path=chroma_db_path)
                
                # Try to get existing collection, or create if it doesn't exist
                try:
                    self.collection = client.get_collection(name="failed_cases")
                    count = self.collection.count()
                    if count > 0:
                        print(f"[SearchFailedCasesTool] Connected to collection 'failed_cases' (contains {count} documents)")
                    else:
                        print(
                            f"[SearchFailedCasesTool] Warning: Collection 'failed_cases' exists but is empty. "
                            f"Please run 'python T3/script/ingest_failed_cases.py' to populate it."
                        )
                except Exception as get_exc:
                    # Collection doesn't exist, create it
                    print(f"[SearchFailedCasesTool] Collection 'failed_cases' not found, creating it: {get_exc}")
                    self.collection = client.get_or_create_collection(name="failed_cases")
                    print(
                        f"[SearchFailedCasesTool] Warning: Collection 'failed_cases' was created but is empty. "
                        f"Please run 'python T3/script/ingest_failed_cases.py' to populate it."
                    )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[SearchFailedCasesTool] Error connecting to ChromaDB: {exc}")
                import traceback
                traceback.print_exc()
                return None
        return self.collection

    async def execute(self, context: ToolContext, args: SearchFailedCasesToolInput) -> ToolResult:
        """Executes a search query against the failed cases in ChromaDB."""
        collection = self._get_collection()
        if collection is None:
            message = "Error: Could not connect to the failed cases database. Please run 'python T3/script/ingest_failed_cases.py' first."
            return ToolResult(success=False, result_for_llm=message, error=message, metadata={"query": args.query})

        try:
            print(f"Executing failed cases search with query: '{args.query}'")
            results = collection.query(query_texts=[args.query], n_results=args.n_results)

            documents_list = results.get("documents") or [[]]
            metadatas_list = results.get("metadatas") or [[]]
            distances_list = results.get("distances") or [[]]
            ids_list = results.get("ids") or [[]]

            documents = documents_list[0] if documents_list else []
            metadatas = metadatas_list[0] if metadatas_list else []
            distances = distances_list[0] if distances_list else []
            ids = ids_list[0] if ids_list else []

            if not documents:
                message = "No similar failed cases found for your query."
                return ToolResult(
                    success=True,
                    result_for_llm=message,
                    metadata={"query": args.query, "documents": [], "metadatas": []},
                )

            print(f"Found {len(documents)} similar failed cases.")

            formatted_results = []
            for idx, document in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                doc_id = ids[idx] if idx < len(ids) else None
                
                # 从文档中提取信息
                lines = document.split('\n')
                question = ""
                sql = ""
                error = ""
                tables = ""
                for line in lines:
                    if line.startswith("问题:"):
                        question = line.replace("问题:", "").strip()
                    elif line.startswith("涉及表:"):
                        tables = line.replace("涉及表:", "").strip()
                    elif line.startswith("错误SQL:"):
                        sql = line.replace("错误SQL:", "").strip()
                    elif line.startswith("错误原因:"):
                        error = line.replace("错误原因:", "").strip()
                
                header = f"失败案例 {idx + 1}:"
                similarity_info = f"相似度: {1 - distance:.3f}" if isinstance(distance, (float, int)) else ""
                sql_id = meta.get("sql_id", "")
                
                info_parts = []
                if sql_id:
                    info_parts.append(f"SQL ID: {sql_id}")
                if similarity_info:
                    info_parts.append(similarity_info)
                if meta.get("error_type"):
                    info_parts.append(f"错误类型: {meta.get('error_type')}")
                
                info_str = " | ".join(info_parts)
                
                result_text = f"{header}"
                if info_str:
                    result_text += f"\n{info_str}"
                result_text += f"\n问题: {question}"
                if tables:
                    result_text += f"\n涉及表: {tables}"
                if sql:
                    # 截断过长的SQL
                    sql_display = sql[:300] + "..." if len(sql) > 300 else sql
                    result_text += f"\n错误SQL: {sql_display}"
                if error:
                    result_text += f"\n错误原因: {error}"
                
                formatted_results.append(result_text)

            result_text = "\n\n".join(formatted_results)

            return ToolResult(
                success=True,
                result_for_llm=result_text,
                metadata={
                    "query": args.query,
                    "documents": documents,
                    "metadatas": metadatas,
                    "distances": distances,
                    "ids": ids,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"An error occurred during failed cases search: {exc}")
            return ToolResult(
                success=False,
                result_for_llm=f"An error occurred during search: {exc}",
                error=str(exc),
                metadata={"query": args.query},
            )