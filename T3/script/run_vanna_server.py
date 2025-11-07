import os

from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.mysql import MySQLRunner
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.user import UserResolver, User, RequestContext

from vanna import Agent,AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.servers.fastapi import VannaFastAPIServer
from vanna_hook import SaveTGACResultHook, TGACRunSqlTool


# Set up OpenAI GPT as your LLM
llm = OpenAILlmService(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-coder-plus-2025-09-23",
    api_key=os.getenv("OPENAI_API_KEY")  # Or use os.getenv("OPENAI_API_KEY")
)

# Set up database connection
db_tool = TGACRunSqlTool(
    sql_runner=MySQLRunner(
        host="localhost",
        database="database_main",
        user="root",
        password="",
        port=9030
    )
)

# Create a simple user resolver
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

# Initialize the user resolver
user_resolver = SimpleUserResolver()

# Set up agent memory for learning from questions and SQL
# agent_memory = DemoAgentMemory(max_items=10000)
agent_memory = ChromaAgentMemory(
    collection_name="vanna_tool_memory",
    persist_directory="./chroma_db",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"
)

# Register tools
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

# Create your agent
agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    lifecycle_hooks=[
        SaveTGACResultHook(
            "T3/upload/dataset_exe_result.json",
        )
    ],
    config=AgentConfig(
        max_tool_iterations=25,
    ),
    agent_memory=agent_memory
)

server = VannaFastAPIServer(agent)
server.run()  # Access at http://localhost:8000