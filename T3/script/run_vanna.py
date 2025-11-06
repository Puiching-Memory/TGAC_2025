import os

# Set up OpenAI GPT as your LLM
from vanna.integrations.openai import OpenAILlmService

llm = OpenAILlmService(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-coder-plus-2025-09-23",
    api_key=os.getenv("OPENAI_API_KEY")  # Or use os.getenv("OPENAI_API_KEY")
)

# Import MySQL tool
from vanna.tools import RunSqlTool
from vanna.integrations.mysql import MySQLRunner

# Set up database connection
db_tool = RunSqlTool(
    sql_runner=MySQLRunner(
        host="localhost",
        database="your_database",
        user="your_user",
        password="your_password",
        port=3306
    )
)

# Import agent memory tools
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool
from vanna.integrations.local.agent_memory import DemoAgentMemory

# Set up agent memory for learning from questions and SQL
agent_memory = DemoAgentMemory(max_items=1000)
save_memory_tool = SaveQuestionToolArgsTool(agent_memory)
search_memory_tool = SearchSavedCorrectToolUsesTool(agent_memory)