"""
基于 LangChain 的 Text2SQL 系统主程序
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
from tools import (
    get_table_schema,
    get_tables_schema,
    execute_sql,
    get_related_examples,
    get_examples_by_table_names,
    get_common_knowledge,
    validate_sql_syntax,
)


class Text2SQLAgent:
    """Text2SQL 代理类"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        enable_memory: bool = True,
    ):
        """
        初始化 Text2SQL 代理
        
        Args:
            model: LLM 模型名称，默认使用 config 中的配置
            temperature: 模型温度参数
            system_prompt: 系统提示词
            enable_memory: 是否启用对话记忆
        """
        self.model_name = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.enable_memory = enable_memory
        
        # 初始化模型
        self.model = init_chat_model(
            self.model_name,
            temperature=self.temperature,
        )
        
        # 定义工具列表
        self.tools = [
            get_table_schema,
            get_tables_schema,
            execute_sql,
            get_related_examples,
            get_examples_by_table_names,
            get_common_knowledge,
            validate_sql_syntax,
        ]
        
        # 创建检查点（用于记忆）
        self.checkpointer = InMemorySaver() if enable_memory else None
        
        # 创建代理
        self.agent = create_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            checkpointer=self.checkpointer,
        )
    
    def invoke(
        self,
        question: str,
        thread_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        执行 Text2SQL 查询
        
        Args:
            question: 用户的问题
            thread_id: 对话线程 ID（用于多轮对话）
            verbose: 是否输出详细信息
        
        Returns:
            代理的响应结果
        """
        # 准备输入消息
        messages = [{"role": "user", "content": question}]
        
        # 准备配置（如果启用记忆）
        config = {}
        if self.enable_memory and thread_id:
            config = {"configurable": {"thread_id": thread_id}}
        
        if verbose:
            print(f"🤖 模型: {self.model_name}")
            print(f"❓ 问题: {question}")
            print(f"🔧 使用工具: {len(self.tools)} 个")
            if thread_id:
                print(f"💬 对话线程: {thread_id}")
            print("-" * 50)
        
        # 调用代理
        try:
            response = self.agent.invoke(
                {"messages": messages},
                config=config if config else None,
            )
            
            if verbose:
                print("\n✅ 响应完成")
            
            return response
        
        except Exception as e:
            error_msg = f"执行代理时出错: {str(e)}"
            if verbose:
                print(f"\n❌ {error_msg}")
            raise Exception(error_msg) from e
    
    def stream(
        self,
        question: str,
        thread_id: Optional[str] = None,
    ):
        """
        流式执行 Text2SQL 查询（实时输出）
        
        Args:
            question: 用户的问题
            thread_id: 对话线程 ID
        
        Yields:
            流式响应块
        """
        messages = [{"role": "user", "content": question}]
        
        config = {}
        if self.enable_memory and thread_id:
            config = {"configurable": {"thread_id": thread_id}}
        
        for chunk in self.agent.stream(
            {"messages": messages},
            config=config if config else None,
        ):
            yield chunk


def main():
    """主函数：命令行交互界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Text2SQL 系统")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 模型名称（例如: openai:gpt-4o, anthropic:claude-sonnet-4-5-20250929）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="模型温度参数（0.0-1.0）",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="禁用对话记忆",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default",
        help="对话线程 ID（用于多轮对话）",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="直接执行的问题（非交互模式）",
    )
    
    args = parser.parse_args()
    
    # 创建代理
    print("🚀 初始化 Text2SQL 代理...")
    agent = Text2SQLAgent(
        model=args.model,
        temperature=args.temperature,
        enable_memory=not args.no_memory,
    )
    print("✅ 代理初始化完成\n")
    
    # 如果提供了问题，直接执行
    if args.question:
        print(f"❓ 问题: {args.question}\n")
        response = agent.invoke(
            question=args.question,
            thread_id=args.thread_id,
            verbose=True,
        )
        
        # 输出响应
        print("\n" + "=" * 50)
        print("📋 响应:")
        print("=" * 50)
        if "messages" in response:
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                print(last_message.content)
            elif isinstance(last_message, dict):
                print(last_message.get("content", ""))
        else:
            print(response)
        return
    
    # 交互模式
    print("=" * 50)
    print("💬 进入交互模式（输入 'exit' 或 'quit' 退出）")
    print("=" * 50)
    print()
    
    while True:
        try:
            question = input("❓ 请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["exit", "quit", "退出"]:
                print("\n👋 再见！")
                break
            
            print("\n🔄 正在处理...\n")
            
            # 执行查询
            response = agent.invoke(
                question=question,
                thread_id=args.thread_id,
                verbose=True,
            )
            
            # 输出响应
            print("\n" + "=" * 50)
            print("📋 响应:")
            print("=" * 50)
            if "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, "content"):
                    print(last_message.content)
                elif isinstance(last_message, dict):
                    print(last_message.get("content", ""))
            else:
                print(response)
            
            print("\n" + "-" * 50 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")


if __name__ == "__main__":
    main()


