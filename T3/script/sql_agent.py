from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools.tools4sql import run_sql_query
from pydantic import BaseModel, Field
import os
from langchain.messages import AIMessage, HumanMessage
import json

class text2SQLResult(BaseModel):
    sql: str = Field(..., description="生成的 SQL 查询语句")
    result: str | None = Field(None, description="查询结果")
    success: bool = Field(..., description="查询是否成功")
    error: str | None = Field(None, description="错误信息")

model = ChatOpenAI(
    # model="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    # base_url="http://localhost:8891/v1",
    # api_key="",
    
    model="qwen3-coder-plus-2025-09-23",
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
    response_format=text2SQLResult,
    # checkpointer=checkpointer
)

result = []
max_step = 5

# 获取提示文件目录
prompt_dir = os.path.join("T3", "script", "prompt", "input", "V1")
prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith('.txt')])

for prompt_file in prompt_files:
    with open(os.path.join(prompt_dir, prompt_file), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 初始化消息历史 - 使用 HumanMessage 对象
    messages = [HumanMessage(content=prompt)]
    
    try:
        out = agent.invoke({"messages": messages})
    except Exception as e:
        print(f"[文件: {prompt_file}] 失败: {str(e)}")
        result.append({
            "sql_id": prompt_file.replace(".txt", ""),
            "sql": None,
            "result": None,
            "success": False,
            "error": str(e),
            "retry_steps": 0
        })
        continue
    
    # 记录首次调用结果
    print(f"[文件: {prompt_file}] 首次尝试:")
    print(f"SQL: {out['structured_response'].sql}")
    print(f"成功: {out['structured_response'].success}")
    if out['structured_response'].error:
        print(f"错误: {out['structured_response'].error}")
    print("-" * 40)
    
    # 重试循环
    step = 0
    while step < max_step and not out["structured_response"].success:
        # 添加助手的响应到历史记录
        messages.append(AIMessage(content=f"SQL: {out['structured_response'].sql}\nResult: {out['structured_response'].result}\nError: {out['structured_response'].error}"))
        
        # 添加用户的反馈（错误信息）到历史记录
        messages.append(HumanMessage(content=f"上一个查询出错了:\n{out['structured_response'].error}\n请重新修正 SQL 查询。"))

        # 使用完整的消息历史进行重试
        try:
            out = agent.invoke({"messages": messages})
        except Exception:
            print(f"[文件: {prompt_file}] 重试时失败，停止重试")
            break
        
        step += 1
        print(f"[文件: {prompt_file}] 重试 {step}:")
        print(f"SQL: {out['structured_response'].sql}")
        print(f"成功: {out['structured_response'].success}")
        if out['structured_response'].error:
            print(f"错误: {out['structured_response'].error}")

    print("=" * 50)

    result.append({
        "sql_id": prompt_file.replace(".txt", ""),
        "sql": out["structured_response"].sql,
        "result": json.loads(out["structured_response"].result) if out["structured_response"].result else None,
        "success": out["structured_response"].success,
        "error": out["structured_response"].error,
        "retry_steps": step if not out["structured_response"].success else 0
    })
    
    # 每轮输出统计
    success_count = sum(1 for r in result if r["success"])
    success_rate = (success_count / len(result) * 100) if result else 0
    print(f"[统计] 已处理: {len(result)}, 成功: {success_count}, 成功率: {success_rate:.2f}%")

    # break

with open("T3/upload/dataset_exe_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=4)

# 最终统计
print("\n" + "=" * 50)
print("最终统计:")
total = len(result)
success_count = sum(1 for r in result if r["success"])
success_rate = (success_count / total * 100) if total > 0 else 0
print(f"总数: {total}")
print(f"成功: {success_count}")
print(f"失败: {total - success_count}")
print(f"成功率: {success_rate:.2f}%")
print("=" * 50)