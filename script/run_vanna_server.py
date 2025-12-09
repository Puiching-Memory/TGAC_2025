import os
from typing import Optional

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.mysql import MySQLRunner
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.user import UserResolver, User, RequestContext

from vanna import Agent,AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.servers.fastapi import VannaFastAPIServer
from vanna_hook import TGACRunSqlTool, FinalResponseSaverHook
from vannna_tools import LightRAGQueryTool
from openai import OpenAI
from vanna.core.enhancer import DefaultLlmContextEnhancer
from vanna.core.system_prompt import DefaultSystemPromptBuilder

# Set up OpenAI GPT as your LLM
llm = OpenAILlmService(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus", # qwen3-coder-flash # qwen-plus # qwen3-coder-plus-2025-09-23,
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中读取

    # base_url="http://127.0.0.1:1234/v1",
    # model="qwen/qwen3-4b-2507",
    # api_key=None

    # base_url="http://127.0.0.1:8891/v1",
    # model="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    # api_key=None
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

DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma 自定义嵌入函数，使用阿里云 DashScope OpenAI 兼容接口。"""

    def __init__(
        self,
        *,
        model: str = DASHSCOPE_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL,
    ) -> None:
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY 环境变量，无法调用阿里云嵌入服务。")

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-DashScope-SSE": "disable"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(input),
        )
        return [list(item.embedding) for item in response.data]

    def get_config(self) -> dict[str, Optional[str]]:
        return {
            "model": self._model,
            "api_base": DASHSCOPE_COMPATIBLE_BASE_URL,
        }


agent_memory = ChromaAgentMemory(
    collection_name="vanna_tool_memory",
    persist_directory="chroma_db",
    embedding_function=DashScopeEmbeddingFunction(),
)

# Register tools
tools = ToolRegistry()

# Create db_tool
OUTPUT_TEXT_DIR = "upload/dataset_exe_result_txt"

db_tool = TGACRunSqlTool(
    sql_runner=MySQLRunner(
        host="localhost",
        database="database_main",
        user="root",
        password="",
        port=9030
    ),
    output_path="upload/dataset_exe_result.json",  # Save results immediately after SQL execution
    output_text_dir=OUTPUT_TEXT_DIR  # Save tool outputs
)

tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
# LightRAG 工具：通过 HTTP API 调用 LightRAG 服务器
tools.register_local_tool(LightRAGQueryTool(), access_groups=['admin', 'user'])
#tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
#tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
#tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])


class TGACSystemPromptBuilder(DefaultSystemPromptBuilder):
    async def build_system_prompt(self, user, tools):  # type: ignore[override]
        base_prompt = await super().build_system_prompt(user, tools)
        extra_guidance = (
            "\n\n=== 知识检索工具优先级 ===\n"
            "• lightrag_query 是图+向量混合检索子代理，擅长“查资料”：\n"
            "  - 精确定位业务概念、指标口径、schema 结构、字段关联、历史 SQL 案例等原始信息；\n"
            "  - 不具备强逻辑推理能力，请勿让其做复杂判断或综合决策。\n"
            "• 当需要补充事实性信息时先调用 lightrag_query，取得素材后由你负责分析与组合逻辑。\n"
            "• 提问时聚焦业务语义与数据内容，不要额外强调 StarRocks 或底层数据库，工具会自动选择数据源。\n"
            "• 在执行 SQL 或输出结论前，总结检索到的要点，并明确尚存的不确定性。\n"
            "• 不准确的提问会导致不准确的回答，当返回结果不符合预期时，请更正你的描述并重新提问。\n"
        )
        if base_prompt:
            return base_prompt + extra_guidance
        return extra_guidance

# Create your agent
agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    lifecycle_hooks=[FinalResponseSaverHook(OUTPUT_TEXT_DIR)],
    config=AgentConfig(
        max_tool_iterations=100,
    ),
    agent_memory=agent_memory,
    llm_context_enhancer=DefaultLlmContextEnhancer(agent_memory),
    system_prompt_builder=TGACSystemPromptBuilder(),
)

server = VannaFastAPIServer(agent)
server.run()  # Access at http://localhost:8000