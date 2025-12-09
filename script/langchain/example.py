"""
LangChain Text2SQL 系统使用示例
"""
from text2sql_agent import Text2SQLAgent


def example_basic():
    """基础使用示例"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)
    
    # 创建代理
    agent = Text2SQLAgent(
        model="openai:gpt-4o",  # 或使用其他模型
        temperature=0.0,
    )
    
    # 执行查询
    question = "统计2025年5月份活跃用户数"
    print(f"\n问题: {question}\n")
    
    response = agent.invoke(question, verbose=True)
    
    # 输出响应
    print("\n" + "=" * 60)
    print("响应:")
    print("=" * 60)
    if "messages" in response:
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            print(last_message.content)
        elif isinstance(last_message, dict):
            print(last_message.get("content", ""))


def example_with_memory():
    """多轮对话示例"""
    print("\n" + "=" * 60)
    print("示例 2: 多轮对话（带记忆）")
    print("=" * 60)
    
    agent = Text2SQLAgent(enable_memory=True)
    thread_id = "example_conversation"
    
    # 第一轮
    question1 = "统计2025年5月份活跃用户数"
    print(f"\n问题 1: {question1}\n")
    response1 = agent.invoke(question1, thread_id=thread_id, verbose=True)
    print("\n响应 1:")
    if "messages" in response1:
        last_msg = response1["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        print(content)
    
    # 第二轮（可以引用之前的对话）
    question2 = "那6月份的呢？"
    print(f"\n问题 2: {question2}\n")
    response2 = agent.invoke(question2, thread_id=thread_id, verbose=True)
    print("\n响应 2:")
    if "messages" in response2:
        last_msg = response2["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        print(content)


def example_stream():
    """流式输出示例"""
    print("\n" + "=" * 60)
    print("示例 3: 流式输出")
    print("=" * 60)
    
    agent = Text2SQLAgent()
    question = "统计2025年5月份活跃用户数"
    
    print(f"\n问题: {question}\n")
    print("流式响应:")
    print("-" * 60)
    
    for chunk in agent.stream(question):
        # 处理流式输出块
        if isinstance(chunk, dict):
            for key, value in chunk.items():
                print(f"{key}: {value}")
        else:
            print(chunk)
    
    print("-" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic()
        elif example_name == "memory":
            example_with_memory()
        elif example_name == "stream":
            example_stream()
        else:
            print(f"未知示例: {example_name}")
            print("可用示例: basic, memory, stream")
    else:
        # 运行所有示例
        try:
            example_basic()
        except Exception as e:
            print(f"示例 1 执行失败: {e}")
        
        try:
            example_with_memory()
        except Exception as e:
            print(f"示例 2 执行失败: {e}")
        
        try:
            example_stream()
        except Exception as e:
            print(f"示例 3 执行失败: {e}")


