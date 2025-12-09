import os
from typing import Any, Dict, List, Optional
from vanna.integrations.openai import OpenAILlmService
from vanna.tools import RunSqlTool
from vanna.integrations.mysql import MySQLRunner
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.registry import ToolRegistry
from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.tools import VisualizeDataTool
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.core.user import UserResolver, User, RequestContext
from vanna.core.agent import AgentConfig


class DeepSeekClientWrapper:
    """包装 OpenAI 客户端，在调用前修改 payload 以支持 DeepSeek Reasoner"""
    
    def __init__(self, original_client):
        self._original_client = original_client
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备消息列表，为 DeepSeek Reasoner 添加 reasoning_content 字段"""
        prepared_messages = []
        for msg in messages:
            prepared_msg = dict(msg)
            
            # 如果是 assistant 消息，需要检查是否需要添加 reasoning_content
            if prepared_msg.get("role") == "assistant":
                # 如果消息中有 tool_calls 但没有 reasoning_content，需要添加
                if "tool_calls" in prepared_msg and prepared_msg.get("tool_calls"):
                    if "reasoning_content" not in prepared_msg:
                        # 如果消息有 content，将其作为 reasoning_content；否则使用空字符串
                        if "content" in prepared_msg and prepared_msg["content"]:
                            prepared_msg["reasoning_content"] = prepared_msg["content"]
                        else:
                            prepared_msg["reasoning_content"] = ""
                # 即使没有 tool_calls，如果消息是空的，也添加 reasoning_content 以避免错误
                elif not prepared_msg.get("content") and "reasoning_content" not in prepared_msg:
                    prepared_msg["reasoning_content"] = ""
            
            prepared_messages.append(prepared_msg)
        
        return prepared_messages
    
    def _patch_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """修补 payload，确保消息格式符合 DeepSeek Reasoner 要求"""
        if "messages" in kwargs:
            kwargs["messages"] = self._prepare_messages(kwargs["messages"])
        return kwargs
    
    def __getattr__(self, name):
        """代理所有其他属性访问到原始客户端"""
        attr = getattr(self._original_client, name)
        
        # 如果是 chat.completions，需要包装它
        if name == "chat":
            return ChatWrapper(attr, self)
        
        return attr


class ChatWrapper:
    """包装 chat 对象"""
    
    def __init__(self, original_chat, client_wrapper):
        self._original_chat = original_chat
        self._client_wrapper = client_wrapper
    
    @property
    def completions(self):
        return CompletionsWrapper(self._original_chat.completions, self._client_wrapper)
    
    def __getattr__(self, name):
        return getattr(self._original_chat, name)


class CompletionsWrapper:
    """包装 completions 对象"""
    
    def __init__(self, original_completions, client_wrapper):
        self._original_completions = original_completions
        self._client_wrapper = client_wrapper
    
    def create(self, **kwargs):
        """在调用前修改 kwargs"""
        kwargs = self._client_wrapper._patch_payload(kwargs)
        return self._original_completions.create(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._original_completions, name)


class DeepSeekReasonerLlmService(OpenAILlmService):
    """自定义 LLM 服务，用于处理 DeepSeek Reasoner 模型的 reasoning_content 字段要求"""
    
    def __init__(self, *args, **kwargs):
        """初始化并包装客户端"""
        super().__init__(*args, **kwargs)
        # 延迟包装客户端，因为客户端可能在初始化后才创建
        self._wrapped_client = None
    
    def __getattribute__(self, name):
        """拦截 _client 属性访问，进行包装"""
        if name == '_client':
            # 获取原始客户端
            client = super().__getattribute__('_client')
            # 如果还没有包装，进行包装
            if self._wrapped_client is None and not isinstance(client, DeepSeekClientWrapper):
                self._wrapped_client = DeepSeekClientWrapper(client)
            # 如果已经包装，返回包装后的客户端
            if self._wrapped_client is not None:
                return self._wrapped_client
            return client
        return super().__getattribute__(name)


# Set up OpenAI GPT as your LLM
llm = DeepSeekReasonerLlmService(
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # model="qwen-plus", # qwen3-coder-flash # qwen-plus # qwen3-coder-plus-2025-09-23,
    # api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中读取

    # base_url="http://127.0.0.1:1234/v1",
    # model="qwen/qwen3-4b-2507",
    # api_key=None

    # base_url="http://127.0.0.1:8891/v1",
    # model="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    # api_key=None

    base_url="https://api.deepseek.com",
    model="deepseek-reasoner",
    api_key=os.getenv("OPENAI_API_KEY")
)

db_tool = RunSqlTool(
    sql_runner=MySQLRunner(
        host="localhost",
        database="database_main",
        user="root",
        password="",
        port=9030
    )
)

# Set up ChromaDB for persistent agent memory
agent_memory = ChromaAgentMemory(
    collection_name="vanna_memory",
    persist_directory="./chroma_db"
)

# Create a simple user resolver
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

# Register memory tools (they access agent_memory via ToolContext)
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])
tools.register_local_tool(VisualizeDataTool(), access_groups=['admin', 'user'])

# 配置 Agent，设置最大工具迭代次数（默认是 10）
agent_config = AgentConfig(
    max_tool_iterations=35  # 可以根据需要调整这个值，例如 20、30 等
)

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    agent_memory=agent_memory,
    config=agent_config
)

server = VannaFastAPIServer(agent)
server.run()  # Access at http://localhost:8000