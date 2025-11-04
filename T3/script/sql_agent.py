from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    base_url="http://localhost:8800/v1",
    api_key=""
)

agent = create_agent(
    tools=[],
    model=model,
)

out = agent.invoke(
    {"messages": [{"role": "user", "content": "统计2025.07.24的手游全量用户且标签为其他，在竞品业务下2025.05.30-2025.07.24的在线时长。\n输出：suserid、sgamecode、ionlinetime\n\n"}]}
    )

print(out)