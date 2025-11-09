import os
from uuid import uuid4
from typing import Any, Dict, List, Optional, Set

import chromadb
from pydantic import BaseModel, Field

from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.tool import Tool, ToolContext
from vanna.core.tool.models import ToolResult
from vanna.core.user import RequestContext, User, UserResolver
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.integrations.openai import OpenAILlmService

class SearchSchemaToolInput(BaseModel):
    query: str = Field(..., description="The natural language query describing the schema info to find.")
    n_results: int = Field(3, description="The maximum number of results to return.")


class SearchDomainKnowledgeToolInput(BaseModel):
    query: str = Field(..., description="Natural language description of the knowledge needed.")
    n_results: int = Field(3, description="Maximum number of knowledge snippets to return.")

class SearchSchemaTool(Tool):
    """
    A tool to search for database schema information (table structures, column descriptions, relationships)
    from a ChromaDB vector database. Use this tool to understand the database layout, find relevant tables,
    or get details about specific columns before generating a SQL query.
    """
    name: str = "search_schema"
    description: str = (
        "Searches for and retrieves information about database tables, columns, and their relationships. "
        "Input should be a query describing the information you are looking for, e.g., 'user login information', "
        "'tables related to players', or 'columns in the dws_jordass_device_login_di table'."
    )
    COLLECTION_CANDIDATES = ("schema_info", "schema_knowledge", "merged_schema_analysis")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chroma_client = None
        self.collection = None
        self.collection_name: Optional[str] = None

    def get_args_schema(self) -> BaseModel:
        return SearchSchemaToolInput

    def _get_collection(self):
        """Initializes and returns the ChromaDB collection."""
        if self.collection is None:
            try:
                # Go up two levels from T3/script to TGAC_2025
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")
                
                print(f"Connecting to ChromaDB for schema search at: {chroma_db_path}")
                self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                
                available_collections: List[str] = []
                try:
                    available_collections = [col.name for col in self.chroma_client.list_collections()]
                except Exception:  # pylint: disable=broad-except
                    available_collections = []

                chosen_collection: Optional[str] = None
                for candidate in self.COLLECTION_CANDIDATES:
                    try:
                        self.collection = self.chroma_client.get_collection(name=candidate)
                    except Exception:  # pylint: disable=broad-except
                        continue
                    chosen_collection = candidate
                    break

                if self.collection is None and available_collections:
                    fallback_name = available_collections[0]
                    self.collection = self.chroma_client.get_collection(name=fallback_name)
                    chosen_collection = fallback_name
                    print(
                        f"Falling back to existing ChromaDB collection '{fallback_name}' for schema search."
                    )

                if self.collection is None:
                    default_name = self.COLLECTION_CANDIDATES[0]
                    self.collection = self.chroma_client.get_or_create_collection(name=default_name)
                    chosen_collection = default_name
                    print(
                        f"Warning: No schema collection found. Created '{default_name}' but it is currently empty."
                    )

                self.collection_name = chosen_collection
                if self.collection_name:
                    print(f"Successfully connected to collection '{self.collection_name}'.")
                else:
                    print("Schema collection connection established, but collection name is unknown.")
            except Exception as e:
                available = []
                try:
                    available = [col.name for col in self.chroma_client.list_collections()]
                except Exception:  # pylint: disable=broad-except
                    available = []
                print(
                    f"Error connecting to ChromaDB for schema search: {e}. Available collections: {available}"
                )
                return None
        return self.collection

    async def execute(self, context: ToolContext, args: SearchSchemaToolInput) -> ToolResult:
        """Executes a search query against the schema information in ChromaDB."""
        collection = self._get_collection()
        if collection is None:
            message = "Error: Could not connect to the schema information database."
            return ToolResult(
                success=False,
                result_for_llm=message,
                error=message,
                metadata={"query": args.query},
            )

        try:
            print(f"Executing schema search with query: '{args.query}'")
            results = collection.query(
                query_texts=[args.query],
                n_results=args.n_results
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
                message = "No relevant schema information found for your query."
                return ToolResult(
                    success=True,
                    result_for_llm=message,
                    metadata={"query": args.query, "documents": [], "metadatas": []},
                )

            print(f"Found {len(documents)} relevant schema documents.")

            formatted_results = []
            for idx, document in enumerate(documents):
                header = f"Result {idx + 1}:"
                meta = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                identifier = ids[idx] if idx < len(ids) else None
                meta_info = [
                    f"id={identifier}" if identifier else None,
                    f"distance={distance:.4f}" if isinstance(distance, (float, int)) else None,
                    f"metadata={meta}" if meta else None,
                ]
                meta_info_str = " | ".join(filter(None, meta_info))
                if meta_info_str:
                    formatted_results.append(f"{header}\n{meta_info_str}\n{document}")
                else:
                    formatted_results.append(f"{header}\n{document}")

            result_text = "\n\n".join(formatted_results)

            # print(f"{args.query}\n{documents}\n{metadatas}\n{distances}\n{ids}")

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
        except Exception as e:
            print(f"An error occurred during schema search: {e}")
            message = f"An error occurred during search: {e}"
            return ToolResult(
                success=False,
                result_for_llm=message,
                error=str(e),
                metadata={"query": args.query},
            )


class SearchDomainKnowledgeTool(Tool):
    """Retrieve background domain knowledge to help interpret user questions."""

    name: str = "search_domain_knowledge"
    description: str = (
        "Searches the curated gameplay and data-warehouse knowledge base. Use before writing SQL when you need "
        "clarifications on terminology, metrics, or conventions mentioned by the user."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collection = None

    def get_args_schema(self) -> BaseModel:
        return SearchDomainKnowledgeToolInput

    def _get_collection(self):
        if self.collection is None:
            try:
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")
                print(f"Connecting to ChromaDB for domain knowledge at: {chroma_db_path}")
                client = chromadb.PersistentClient(path=chroma_db_path)
                self.collection = client.get_collection(name="domain_knowledge")
                print("Connected to collection 'domain_knowledge'.")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error connecting to ChromaDB collection 'domain_knowledge': {exc}")
                return None
        return self.collection

    async def execute(self, context: ToolContext, args: SearchDomainKnowledgeToolInput) -> ToolResult:
        collection = self._get_collection()
        if collection is None:
            message = "Error: Could not connect to the domain knowledge database."
            return ToolResult(success=False, result_for_llm=message, error=message, metadata={"query": args.query})

        try:
            print(f"Executing domain knowledge search with query: '{args.query}'")
            results = collection.query(query_texts=[args.query], n_results=args.n_results)

            documents_list = results.get("documents") or [[]]
            metadatas_list = results.get("metadatas") or [[]]
            distances_list = results.get("distances") or [[]]

            documents = documents_list[0]
            metadatas = metadatas_list[0] if metadatas_list else []
            distances = distances_list[0] if distances_list else []

            if not documents:
                message = "No matching domain knowledge found for your query."
                return ToolResult(success=True, result_for_llm=message, metadata={"query": args.query})

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

            result_text = "\n\n".join(formatted_results)

            return ToolResult(
                success=True,
                result_for_llm=result_text,
                metadata={
                    "query": args.query,
                    "documents": documents,
                    "metadatas": metadatas,
                    "distances": distances,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"An error occurred during domain knowledge search: {exc}")
            return ToolResult(
                success=False,
                result_for_llm=f"An error occurred during search: {exc}",
                error=str(exc),
                metadata={"query": args.query},
            )


class InvestigateConceptToolInput(BaseModel):
    concept: str = Field(..., description="The unfamiliar concept or term that needs clarification.")
    schema_results: int = Field(4, gt=0, le=10, description="Maximum schema snippets to retrieve.")
    domain_results: int = Field(4, gt=0, le=10, description="Maximum domain knowledge entries to retrieve.")
    memory_results: int = Field(5, gt=0, le=15, description="Maximum historical tool usages to surface.")
    include_agent_memory: bool = Field(True, description="Whether to search previously logged tool usage memories.")


class InvestigateConceptTool(Tool):
    name: str = "investigate_unknown_concept"
    description: str = (
        "Launches a focused Vanna sub-agent that controls its own context and tool planning to investigate"
        " unfamiliar concepts across the full knowledge base."
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def get_args_schema(self) -> BaseModel:
        return InvestigateConceptToolInput
    def _build_user_prompt(self, args: InvestigateConceptToolInput) -> str:
        return (
            "You are the TGAC concept-research sub-agent. Your task is to understand the concept "
            f"'{args.concept}'.\n"
            "Plan your own investigation, leverage the available tools, and deliver a crisp brief that helps the"
            " main analyst write accurate SQL. Follow these rules:\n"
            "1. Map the concept to relevant tables, columns, and metrics using `search_schema` (<= "
            f"{args.schema_results} results per call).\n"
            "2. Cross-check terminology and business definitions using `search_domain_knowledge` (<= "
            f"{args.domain_results} results).\n"
            "3. If past tool memories are available, identify prior successful reasoning paths.\n"
            "4. Iterate until you have enough evidence. Prefer multiple narrow queries over one broad request.\n"
            "5. Summarize with sections: Concept Interpretation, Relevant Schema Assets, Domain Insights,"
            " and Open Questions / Next Steps.\n"
            "6. Provide actionable guidance and call out uncertainties.\n"
            "Begin by outlining your plan before calling tools."
        )

    def _create_llm_service(self) -> OpenAILlmService:
        base_url = (
            os.getenv("SUB_AGENT_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = (
            os.getenv("SUB_AGENT_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "qwen3-coder-flash"
        )
        api_key = os.getenv("SUB_AGENT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        return OpenAILlmService(base_url=base_url, model=model, api_key=api_key)

    def _create_sub_agent(self) -> tuple[Agent, User]:
        llm_service = self._create_llm_service()
        access_groups = ["admin"]

        registry = ToolRegistry()
        registry.register_local_tool(SearchSchemaTool(), access_groups=access_groups)
        registry.register_local_tool(SearchDomainKnowledgeTool(), access_groups=access_groups)

        agent_memory = DemoAgentMemory(max_items=512)

        config = AgentConfig(
            max_tool_iterations=25,
            stream_responses=False,
            include_thinking_indicators=False,
            #temperature=0.1,
            #max_tokens=1600,
        )

        user = User(
            id=f"concept_subagent_{uuid4()}",
            email="admin@example.com",  # synthetic identity
            group_memberships=access_groups,
        )

        class _StaticResolver(UserResolver):
            def __init__(self, static_user: User) -> None:
                self._user = static_user

            async def resolve_user(self, request_context: RequestContext) -> User:  # type: ignore[override]
                return self._user

        resolver = _StaticResolver(user)

        agent = Agent(
            llm_service=llm_service,
            tool_registry=registry,
            user_resolver=resolver,
            agent_memory=agent_memory,
            config=config,
        )
        return agent, user

    @staticmethod
    def _truncate_text(value: str, limit: int = 1200) -> str:
        text = value.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + " … [truncated]"

    async def execute(self, context: ToolContext, args: InvestigateConceptToolInput) -> ToolResult:
        metadata_bundle: Dict[str, Any] = {"concept": args.concept}

        try:
            sub_agent, sub_agent_user = self._create_sub_agent()
        except Exception as exc:  # pylint: disable=broad-except
            message = f"Failed to initialize concept sub-agent: {exc}"
            return ToolResult(
                success=False,
                result_for_llm=message,
                error=str(exc),
                metadata=metadata_bundle,
            )

        conversation_id = f"concept-run-{uuid4()}"
        metadata_bundle["conversation_id"] = conversation_id
        metadata_bundle["sub_agent_user_id"] = sub_agent_user.id

        user_prompt = self._build_user_prompt(args)
        metadata_bundle["sub_agent_prompt"] = user_prompt

        request_context = RequestContext(
            metadata={
                "invoked_by_tool": self.name,
                "parent_conversation": context.conversation_id,
                "concept": args.concept,
            }
        )

        print(
            "[InvestigateConceptTool] Starting sub-agent run",
            f"conversation_id={conversation_id}",
            f"concept='{args.concept}'",
        )
        print("[InvestigateConceptTool] Prompt dispatched to sub-agent:\n", user_prompt)

        try:
            async for _component in sub_agent.send_message(
                request_context,
                user_prompt,
                conversation_id=conversation_id,
            ):
                try:
                    rich_type = getattr(_component.rich_component, "type", None)
                    rich_name = getattr(rich_type, "value", rich_type)
                    print(
                        "[InvestigateConceptTool] Sub-agent emitted component",
                        f"type={rich_name}",
                    )
                except Exception:  # pylint: disable=broad-except
                    print("[InvestigateConceptTool] Sub-agent emitted a UI component.")
                # Conversation transcript is persisted; nothing else required per component.
                continue
        except Exception as exc:  # pylint: disable=broad-except
            message = f"Concept sub-agent execution failed: {exc}"
            return ToolResult(
                success=False,
                result_for_llm=message,
                error=str(exc),
                metadata=metadata_bundle,
            )

        conversation = await sub_agent.conversation_store.get_conversation(
            conversation_id, sub_agent_user
        )

        if conversation is None:
            message = "Concept sub-agent completed without returning a transcript."
            return ToolResult(
                success=False,
                result_for_llm=message,
                error=message,
                metadata=metadata_bundle,
            )

        print(
            "[InvestigateConceptTool] Sub-agent conversation captured",
            f"messages={len(conversation.messages)}",
        )

        transcript_records: List[Dict[str, Any]] = []
        assistant_messages: List[str] = []
        tool_invocations: List[Dict[str, Any]] = []
        tool_outputs: List[str] = []

        for message in conversation.messages:
            transcript_records.append(
                {
                    "role": message.role,
                    "content": self._truncate_text(message.content, 600),
                    "tool_call_id": message.tool_call_id,
                    "timestamp": message.timestamp.isoformat(),
                }
            )

            if message.role == "assistant":
                if message.content:
                    assistant_messages.append(message.content)
                if message.tool_calls:
                    for call in message.tool_calls:
                        tool_invocations.append({"name": call.name, "arguments": call.arguments})
            elif message.role == "tool":
                tool_outputs.append(message.content)

        metadata_bundle["transcript"] = transcript_records
        metadata_bundle["assistant_messages"] = assistant_messages
        metadata_bundle["executed_tools"] = tool_invocations
        metadata_bundle["tool_outputs"] = [self._truncate_text(t, 1200) for t in tool_outputs]

        memory_highlights: List[str] = []
        if args.include_agent_memory and hasattr(context.agent_memory, "search_similar_usage"):
            try:
                memory_hits = await context.agent_memory.search_similar_usage(  # type: ignore[attr-defined]
                    args.concept,
                    context,
                    limit=args.memory_results,
                )
            except Exception as exc:  # pylint: disable=broad-except
                metadata_bundle["agent_memory_error"] = str(exc)
            else:
                formatted_hits: List[Dict[str, Any]] = []
                for hit in memory_hits:
                    memory = getattr(hit, "memory", None)
                    similarity = getattr(hit, "similarity_score", None)
                    if memory is None:
                        continue
                    record = {
                        "question": getattr(memory, "question", ""),
                        "tool_name": getattr(memory, "tool_name", ""),
                        "args": getattr(memory, "args", {}),
                        "success": getattr(memory, "success", True),
                        "similarity": similarity,
                    }
                    formatted_hits.append(record)
                    question_preview = self._truncate_text(record["question"] or "", 160)
                    tool_name = record["tool_name"] or "unknown_tool"
                    similarity_txt = f"{similarity:.2f}" if isinstance(similarity, (float, int)) else "n/a"
                    memory_highlights.append(
                        f"- {tool_name} (similarity {similarity_txt}) → {question_preview}"
                    )
                metadata_bundle["memory_hits"] = formatted_hits

        final_response = assistant_messages[-1].strip() if assistant_messages else ""
        if not final_response:
            final_response = (
                "No assistant summary was produced. Consult the tool evidence below for raw findings."
            )

        result_sections: List[str] = ["### Concept Investigation Summary", final_response]

        if tool_invocations:
            result_sections.append("### Executed Tools")
            for call in tool_invocations:
                result_sections.append(
                    f"- {call['name']} with args {call['arguments']}"
                )

        if memory_highlights:
            result_sections.append("### Historical Signals")
            result_sections.extend(memory_highlights)

        if tool_outputs:
            result_sections.append("### Tool Evidence")
            for idx, payload in enumerate(tool_outputs, start=1):
                result_sections.append(
                    f"{idx}. {self._truncate_text(payload, 800)}"
                )
        else:
            result_sections.append("### Tool Evidence")
            result_sections.append("No tool outputs were captured.")

        final_payload = "\n".join(result_sections).strip()

        print("[InvestigateConceptTool] Sub-agent run completed successfully.")
        if tool_invocations:
            print(
                "[InvestigateConceptTool] Tools invoked:",
                ", ".join(call["name"] for call in tool_invocations),
            )

        # Cleanup the temporary conversation to keep the sub-agent stateless per run.
        await sub_agent.conversation_store.delete_conversation(conversation_id, sub_agent_user)

        return ToolResult(
            success=True,
            result_for_llm=final_payload,
            metadata=metadata_bundle,
        )