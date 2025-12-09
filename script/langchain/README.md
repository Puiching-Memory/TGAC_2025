# LangChain Text2SQL 系统

基于 LangChain 框架构建的 Text2SQL 系统，可以将自然语言问题转换为 SQL 查询语句。

## 功能特性

- 🤖 **智能 SQL 生成**：基于大语言模型将自然语言问题转换为 SQL
- 🔧 **丰富的工具集**：提供表结构查询、SQL 执行、示例检索等工具
- 💾 **对话记忆**：支持多轮对话，保持上下文
- 📚 **知识库集成**：集成业务知识和示例库
- ✅ **SQL 验证**：自动验证 SQL 语法和结构

## 系统架构

```
script/langchain/
├── config.py              # 配置文件（数据库、模型等）
├── data_loader.py          # 数据加载工具
├── tools.py                # LangChain 工具定义
├── text2sql_agent.py       # 主程序
└── README.md               # 说明文档
```

## 安装依赖

确保已安装以下依赖：

```bash
# 基础依赖
pip install langchain langchain-community langchain-openai

# 数据库依赖
pip install pymysql sqlalchemy

# 可选：其他 LLM 提供商
pip install langchain-anthropic  # 用于 Claude
pip install langchain-google-vertexai  # 用于 Google Vertex AI
```

## 配置

### 环境变量

在运行前，请设置以下环境变量（或修改 `config.py`）：

```bash
# 数据库配置
export DB_HOST=127.0.0.1
export DB_PORT=9030
export DB_USER=root
export DB_PASSWORD=your_password
export DB_DATABASE=database_main

# LLM 配置
export LLM_MODEL=openai:gpt-4o
export LLM_TEMPERATURE=0.0

# API Keys（根据使用的模型提供商设置）
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

### 配置文件

也可以直接修改 `config.py` 中的配置：

```python
# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 9030,
    "user": "root",
    "password": "",
    "database": "database_main",
}

# LLM 模型
LLM_MODEL = "openai:gpt-4o"  # 或 "anthropic:claude-sonnet-4-5-20250929"
```

## 使用方法

### 1. 命令行交互模式

```bash
cd script/langchain
python text2sql_agent.py
```

进入交互模式后，输入问题即可：

```
❓ 请输入您的问题: 统计2025年5月份活跃用户数
```

### 2. 直接执行问题

```bash
python text2sql_agent.py --question "统计2025年5月份活跃用户数"
```

### 3. 指定模型和参数

```bash
python text2sql_agent.py \
    --model "anthropic:claude-sonnet-4-5-20250929" \
    --temperature 0.1 \
    --question "统计2025年5月份活跃用户数"
```

### 4. 多轮对话

使用 `--thread-id` 参数保持对话上下文：

```bash
python text2sql_agent.py --thread-id "conversation_1"
```

### 5. 禁用记忆

```bash
python text2sql_agent.py --no-memory
```

## 作为 Python 模块使用

```python
from text2sql_agent import Text2SQLAgent

# 创建代理
agent = Text2SQLAgent(
    model="openai:gpt-4o",
    temperature=0.0,
)

# 执行查询
response = agent.invoke(
    question="统计2025年5月份活跃用户数",
    thread_id="my_conversation",
    verbose=True,
)

# 获取响应
print(response)
```

## 可用工具

系统提供以下工具供代理使用：

1. **get_table_schema** - 获取表结构信息
2. **get_tables_schema** - 批量获取多个表的结构
3. **execute_sql** - 执行 SQL 查询（仅 SELECT）
4. **get_related_examples** - 获取相似问题示例
5. **get_examples_by_table_names** - 根据表名获取示例
6. **get_common_knowledge** - 获取通用业务知识
7. **validate_sql_syntax** - 验证 SQL 语法

## 数据文件

系统需要以下数据文件（位于 `data/` 目录）：

- `final_dataset.json` - 训练数据集（包含问题-SQL 对）
- `common_knowledge.md` - 通用业务知识库

## 支持的模型

系统支持所有 LangChain 兼容的模型，包括：

- OpenAI: `openai:gpt-4o`, `openai:gpt-4-turbo`, `openai:gpt-3.5-turbo`
- Anthropic: `anthropic:claude-sonnet-4-5-20250929`, `anthropic:claude-3-opus`
- Google: `google:gemini-pro`
- 其他 LangChain 支持的模型

## 示例

### 示例 1：简单查询

```python
agent = Text2SQLAgent()
response = agent.invoke("统计2025年5月份活跃用户数")
```

### 示例 2：复杂查询（使用工具）

代理会自动使用工具获取表结构、查找示例等：

```python
agent = Text2SQLAgent()
response = agent.invoke(
    "统计2025年5月份活跃但6月未活跃的qq号码包",
    thread_id="complex_query",
)
```

### 示例 3：流式输出

```python
agent = Text2SQLAgent()
for chunk in agent.stream("统计活跃用户数"):
    print(chunk)
```

## 注意事项

1. **数据库连接**：确保数据库可访问，且配置正确
2. **API Keys**：使用 OpenAI/Anthropic 等模型时需要设置相应的 API Key
3. **SQL 安全**：`execute_sql` 工具仅支持 SELECT 查询，不会修改数据
4. **数据文件**：确保 `data/final_dataset.json` 和 `data/common_knowledge.md` 存在

## 故障排除

### 1. 数据库连接失败

检查数据库配置和网络连接：

```python
from tools import get_db_engine
engine = get_db_engine()
# 测试连接
```

### 2. 模型 API 调用失败

检查 API Key 是否正确设置：

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### 3. 找不到数据文件

确保数据文件路径正确：

```python
from config import FINAL_DATASET_PATH
print(FINAL_DATASET_PATH)
```

## 扩展开发

### 添加新工具

在 `tools.py` 中添加新工具：

```python
@tool
def my_custom_tool(param: str) -> str:
    """工具描述"""
    # 实现逻辑
    return result
```

然后在 `text2sql_agent.py` 中将工具添加到 `self.tools` 列表。

### 自定义系统提示词

```python
agent = Text2SQLAgent(
    system_prompt="你的自定义提示词...",
)
```

## 许可证

本项目遵循项目主仓库的许可证。


