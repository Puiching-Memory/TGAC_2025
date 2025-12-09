from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field, field_validator

from vanna.core.tool import Tool, ToolContext
from vanna.core.tool.models import ToolResult

# LightRAG 默认服务器配置，集中管理便于修改
DEFAULT_LIGHTRAG_SCHEME = "http"
DEFAULT_LIGHTRAG_HOST = "localhost"
DEFAULT_LIGHTRAG_PORT = 18000
DEFAULT_LIGHTRAG_BASE_URL = f"{DEFAULT_LIGHTRAG_SCHEME}://{DEFAULT_LIGHTRAG_HOST}:{DEFAULT_LIGHTRAG_PORT}"


class LightRAGQueryArgs(BaseModel):
    """LightRAG 查询参数"""
    query: str = Field(..., description="要检索的问题或查询语句，必须为非空字符串。")
    server_url: Optional[str] = Field(default=None, description="可选，LightRAG 服务器地址。")
    api_key: Optional[str] = Field(
        default=None,
        description="可选，LightRAG API 认证密钥。",
    )
    max_output_chars: int = Field(
        default=4000,
        description="最大输出字符数，超过部分将被截断。",
    )
    mode: str = Field(
        default="mix",
        description="RAG 检索模式，mix 模式优先同时利用图谱与向量索引。",
    )
    include_references: bool = Field(
        default=True,
        description="是否在响应中包含引用来源信息。",
    )
    response_type: str = Field(
        default="要点",
        description="响应格式偏好，例如“要点”“列表”“多段落”等。",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="检索时返回的实体或关系数量上限。",
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="可选，对话历史上下文，用于保持语境一致性。",
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        description="整个响应的 token 预算限制。",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        sanitized = value.strip()
        if not sanitized:
            raise ValueError("query 不能为空")
        if len(sanitized) < 3:
            raise ValueError("query 长度至少为 3 个字符")
        return sanitized

    @field_validator("max_output_chars")
    @classmethod
    def validate_max_output_chars(cls, value: int) -> int:
        if value < 100:
            raise ValueError("max_output_chars 必须至少为 100")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("top_k 必须为正整数")
        return value

    @field_validator("max_total_tokens")
    @classmethod
    def validate_max_total_tokens(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("max_total_tokens 必须为正整数")
        return value

    class Config:
        extra = "forbid"


class LightRAGQueryTool(Tool[LightRAGQueryArgs]):
    """LightRAG 查询工具：通过 HTTP API 调用 LightRAG 服务器进行知识检索。"""

    def __init__(
        self,
        default_server_url: str = DEFAULT_LIGHTRAG_BASE_URL,
        default_api_key: Optional[str] = None,
        max_output_chars: int = 4000,
        timeout: int = 150,
    ) -> None:
        """
        初始化 LightRAG 查询工具。

        Args:
            default_server_url: 默认 LightRAG 服务器地址
            default_api_key: 默认 API 密钥（如果服务器需要认证）
            max_output_chars: 最大输出字符数
            timeout: HTTP 请求超时时间（秒）
        """
        self._default_server_url = default_server_url.rstrip("/")
        self._default_api_key = default_api_key
        self._max_output_chars = max_output_chars
        self._timeout = timeout

    def get_args_schema(self) -> type[LightRAGQueryArgs]:
        return LightRAGQueryArgs

    @property
    def name(self) -> str:
        return "lightrag_query"

    @property
    def description(self) -> str:
        return (
            "LightRAG 子代理检索工具：依托图结构与向量索引，可精准检索业务概念、字段释义、"
            "schema 结构、字段关联关系以及历史 SQL 案例等具体事实信息。"
            "请将该工具视作“资料查询助手”，仅用于获取细致、可引用的原始信息；"
            "在提问时聚焦业务问题本身，不必强调 StarRocks 或底层数据库，实现中会自动匹配数据源。"
            "复杂逻辑推演与多信息综合由你自行完成。"
        )

    def _format_output(self, raw_text: str) -> str:
        """格式化输出文本"""
        if not raw_text:
            raw_text = "LightRAG 查询已完成，但未返回任何内容。"
        if len(raw_text) > self._max_output_chars:
            raw_text = raw_text[: self._max_output_chars] + "\n\n...（输出过长，已截断）"
        header = "[LightRAG]"
        return f"{header}\n{raw_text.strip()}"

    @staticmethod
    def _truncate_for_log(value: str) -> str:
        return value

    async def _query_lightrag_api(
        self,
        payload: Dict[str, Any],
        server_url: str,
        api_key: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        调用 LightRAG API 进行查询。

        Args:
            payload: 查询请求体
            server_url: LightRAG 服务器地址
            api_key: API 密钥（可选）

        Returns:
            (response_text, metadata) 元组
        """
        # 构建请求 URL
        query_url = f"{server_url}/query"

        # 构建请求头
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 构建请求体
        request_payload = {key: value for key, value in payload.items() if value is not None}

        # 发送 HTTP 请求
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    query_url,
                    json=request_payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    try:
                        result = await response.json()
                    except aiohttp.ContentTypeError:
                        result = await response.text()

                    response_text = ""
                    if isinstance(result, dict):
                        response_text = (
                            result.get("response")
                            or result.get("answer")
                            or result.get("text")
                            or result.get("content")
                            or result.get("message")
                            or result.get("result")
                            or str(result)
                        )
                    elif isinstance(result, str):
                        response_text = result
                    else:
                        response_text = str(result)

                    metadata = {
                        "status_code": response.status,
                        "raw_response": result,
                        "query": request_payload.get("query", ""),
                        "server_url": server_url,
                        "request_url": query_url,
                        "request_payload": request_payload,
                    }

                    return response_text, metadata

            except aiohttp.ClientError as e:
                error_msg = f"LightRAG API 请求失败: {str(e)}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"LightRAG API 调用出错: {str(e)}"
                raise RuntimeError(error_msg) from e

    async def execute(
        self, context: ToolContext, args: LightRAGQueryArgs
    ) -> ToolResult:
        """执行 LightRAG 查询"""
        started_at = datetime.now(timezone.utc)

        # 确定服务器地址和 API 密钥
        server_url = args.server_url or self._default_server_url
        api_key = args.api_key or self._default_api_key
        max_output_chars = args.max_output_chars or self._max_output_chars

        print(
            "[LightRAGQueryTool] 调用开始",
            "| query:",
            self._truncate_for_log(args.query),
            "| server_url:",
            server_url,
        )

        try:
            # 调用 LightRAG API
            payload: Dict[str, Any] = {
                "query": args.query,
                "mode": args.mode,
                "include_references": args.include_references,
                "response_type": args.response_type,
                "top_k": args.top_k,
                "conversation_history": args.conversation_history,
                "max_total_tokens": args.max_total_tokens,
            }

            response_text, metadata = await self._query_lightrag_api(
                payload=payload,
                server_url=server_url,
                api_key=api_key,
            )

            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            metadata["elapsed_seconds"] = elapsed

            # 格式化输出
            formatted_output = self._format_output(response_text)

            print(
                "[LightRAGQueryTool] 调用成功",
                f"| elapsed={elapsed:.3f}s",
                "| 响应片段:",
                self._truncate_for_log(response_text),
            )

            return ToolResult(
                success=True,
                result_for_llm=formatted_output,
                metadata=metadata,
            )

        except Exception as exc:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            error_msg = f"LightRAG 查询执行出错: {exc}"

            print(
                "[LightRAGQueryTool] 调用失败",
                f"| elapsed={elapsed:.3f}s",
                "| error:",
                exc,
            )

            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(exc),
                metadata={
                    "query": args.query,
                    "server_url": server_url,
                    "elapsed_seconds": elapsed,
                },
            )
