import os

import chromadb
from pydantic import BaseModel, Field
from vanna.core.tool import Tool, ToolContext
from vanna.core.tool.models import ToolResult

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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chroma_client = None
        self.collection = None

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
                
                collection_name = "schema_info"
                self.collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Successfully connected to collection '{collection_name}'.")
            except Exception as e:
                print(f"Error connecting to ChromaDB collection 'schema_info': {e}")
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
            for idx, document in enumerate(iterable=documents):
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

            print(f"Domain knowledge search results:\n{result_text}")

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