"""
LangChain Text2SQL 系统配置文件
"""
import os
from pathlib import Path
from typing import Optional

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# 数据路径
DATA_DIR = REPO_ROOT / "data"
FINAL_DATASET_PATH = DATA_DIR / "final_dataset.json"
COMMON_KNOWLEDGE_PATH = DATA_DIR / "common_knowledge.md"

# 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "9030")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_DATABASE", "database_main"),
    "charset": "utf8mb4",
}

# 数据库连接 URL（用于 SQLAlchemy）
DB_URL = os.getenv(
    "DB_URL",
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# LLM 模型配置
LLM_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# 系统提示词模板
SYSTEM_PROMPT = """你是一个专业的 SQL 生成助手，擅长根据自然语言问题生成准确的 SQL 查询语句。

你的任务：
1. 理解用户的问题和需求
2. 根据提供的表结构和业务知识生成正确的 SQL 查询
3. 确保 SQL 语法正确且符合数据库规范
4. 考虑日期格式、字段类型等细节

重要提示：
- 日期格式通常为 YYYYMMDD（如 20250724）
- 注意表的分区字段（通常是日期字段）
- 遵循数仓设计规范（DWD、DWS、DIM 层）
- 注意字段的数据类型和业务含义

当你需要更多信息时，可以使用提供的工具来：
- 查询表结构
- 查看示例 SQL
- 获取业务知识
"""


