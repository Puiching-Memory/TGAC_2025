"""
配置文件：统一管理提示词版本

使用方法：
1. 在 generate_prompts.py 和 run_datus.py 中导入此配置
2. 修改版本号时只需修改此文件

示例：
    from config import PROMPT_VERSION
"""

# 当前使用的提示词版本
# 修改此版本号会影响所有脚本
PROMPT_VERSION = "v1.0.0"

# 版本历史和说明
VERSION_HISTORY = {
    "v1.0.0": {
        "date": "2025-11-04",
        "description": "初始版本，基础提示词结构",
        "changes": [
            "包含数据库 schema",
            "包含公共知识和业务知识",
            "随机选择 1 条 few-shot 示例",
        ],
    },
    "v1.1.0": {
        "date": "2025-11-04",
        "description": "示例版本，演示版本管理",
        "changes": [
            "（这是一个示例，实际使用时根据需要修改）",
        ],
    },
}

# 其他配置常量
DATABASE_VERSION = "4.0.0"
DATABASE_TYPE = "StarRocks"

# API 配置
API_CONFIG = {
    "url": "http://localhost:6080/workflows/run",
    "token_url": "http://localhost:6080/auth/token",
    "client_id": "your_client_id",
    "client_secret": "client",
    "workflow_name": "reflection",
    "namespace": "game",
    "auth_timeout": 40,
    "workflow_timeout": 300,
}
