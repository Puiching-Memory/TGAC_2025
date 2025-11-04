from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools.tools4sql import run_sql_query,get_mschema
import json
from toon import encode
from transformers import AutoTokenizer

with open("T3/data/final_dataset.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

model = ChatOpenAI(
    #model="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    model="qwen3-coder-plus-2025-09-23",
    # base_url="http://localhost:8891/v1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-21f31afa708c4c6f9bf6b73585788e41",
    temperature=0.1,
    n=1,
    # max_completion_tokens=1024,
    # timeout=60,
    # model_kwargs={"tool_choice": "required"} # 强制工具调用会陷入死循环
)

agent = create_agent(
    tools=[run_sql_query],
    model=model,
)

prompt = f"""你是一名StarRocks mysql 4.0.0专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用StarRocks mysql 4.0.0知识生成sql语句回答【用户问题】。
[question]: {json_data[0]["question"]}  
[table_list]: {json_data[0]["table_list"]}  
[knowledge]: {json_data[0]["knowledge"]}  
[Schema]: {encode([get_mschema(i) for i in json_data[0]["table_list"]])}
"""

print(f"{prompt}")
print(f"Token count: {len(tokenizer.encode(prompt))}")

# print(f"{json_data[0]}")

out = agent.invoke(
    {"messages": [{"role": "user",
                        "content": f"{prompt}"}]}
    )

print(out)