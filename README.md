<h1 align="center">TGAC 2025 Text-to-SQL Workspace</h1>

<p align="center">
  <strong>数智决策科学：面向游戏业务的自然语言查询与 SQL 智能体</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="License: GPL-3.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python: 3.12">
  <img src="https://img.shields.io/badge/Database-StarRocks-0A7B83.svg" alt="Database: StarRocks">
  <img src="https://img.shields.io/badge/Task-Text--to--SQL-brightgreen.svg" alt="Task: Text-to-SQL">
  <img src="https://img.shields.io/badge/Best%20Score-40.70-orange.svg" alt="Best recorded score: 40.70">
</p>

<p align="center">
  <a href="https://tgac.tencent.com/">Competition</a> ·
  <a href="#快速开始">Quick Start</a> ·
  <a href="upload/方案介绍.md">Solution</a> ·
  <a href="#成绩与版本记录">Results</a> ·
  <a href="#社区与相关工作">Community</a>
</p>

> [!NOTE]
> 这是 TGAC 2025 腾讯游戏算法大赛**赛道 3：数智决策科学**参赛队伍 **proximusA** 的代码与实验记录，不代表官方方案。
> 仓库原始记录：参赛队伍数 **> 140**，最终排名 **54**，最高记录得分 **40.70**。
> 本项目保留比赛期间的数据处理、Agent 探索与提交辅助脚本，供赛后研究和复盘使用。

## 比赛简介

数智决策科学赛道围绕游戏业务中的自然语言数据分析展开：将业务问题、表结构和补充知识转化为 SQL，在数据库中执行并返回查询结果。对于复杂问题，除了 SQL 语法，还需要处理业务口径、字段含义、多表关联、时间条件和返回格式。

本仓库以 **StarRocks + LLM Agent** 为基础，探索提示词工程、历史正确案例检索、业务知识检索与执行反馈。现有代码包含 Vanna 服务与批量客户端，以及独立的 LangChain 实验入口。比赛入口见 [TGAC 官网](https://tgac.tencent.com/)，比赛期间的环境与模型部署笔记见[方案介绍](upload/方案介绍.md)。

------

## 我们的工作

<table>
<tr>
<td width="50%" valign="top">

**SQL 生成与执行**

- **Vanna Agent 服务** — OpenAI 兼容接口接入，当前入口配置为 DeepSeek Reasoner，包含工具调用消息适配
- **StarRocks 查询** — 通过 MySQL 协议访问比赛数据库，配套建表、数据导入和 SQL 执行脚本
- **批量题目处理** — 按提示词文件逐题请求服务，支持复用已有评分中正确的缓存答案

</td>
<td width="50%" valign="top">

**上下文与知识**

- **任务提示词** — 结合问题、涉及表和补充知识，重新生成时保留人工填写的工程师信息
- **Schema 表达** — M-Schema 与 TOON 转换脚本，分别提供表级和字段级文本
- **记忆与检索** — ChromaDB 记忆导入工具，以及可供接入的 LightRAG HTTP 查询工具

</td>
</tr>
<tr>
<td valign="top">

**结果与复盘**

- **自定义 SQL Hook** — 保存执行 SQL、结果和文本摘要，并提供查询诊断辅助
- **版本对比** — 根据逐题评分比较两个本地版本新增的正确题目
- **错误分析** — 按题目复杂度及涉及表汇总未通过题目

</td>
<td valign="top">

**实验与资料**

- **LangChain 实验** — 独立的交互式与单问题查询入口，附配置和工具说明
- **本地模型笔记** — 保留 XiYanSQL 与 vLLM 的比赛期间部署记录
- **成绩演进** — 保留从 v2 到 V9 的全部 12 条得分记录

</td>
</tr>
</table>

### 代码入口与接入状态

| 模块 | 入口 | 当前状态 |
| --- | --- | --- |
| Vanna 服务 | [script/run_vanna_server.py](script/run_vanna_server.py) | 注册 SQL、记忆和可视化工具；模型与数据库参数在文件内配置 |
| 批量客户端 | [script/run_vanna_client.py](script/run_vanna_client.py) | 请求本地服务；缓存和提示词路径在文件顶部配置 |
| SQL 结果保存与诊断 | [script/vanna_hook.py](script/vanna_hook.py) | 提供 `TGACRunSqlTool` 等实现，默认服务入口尚未接入 |
| LightRAG 检索 | [script/vannna_tools.py](script/vannna_tools.py) | 提供 `LightRAGQueryTool`，默认服务入口尚未注册 |
| LangChain 实验 | [script/langchain](script/langchain) | 独立入口，需另装该目录说明中的依赖 |

> [!IMPORTANT]
> 默认 Vanna 服务可以提供交互式查询，但没有启用本仓库的自动结果保存 Hook。
> 若要生成批量提交文件，需要先完成下文的结果保存接入；LightRAG 也需自行启动、导入知识并注册工具。
> 仓库未提供锁定完整依赖的环境文件，下方命令依据现有源码整理，复现时需核对所安装 Vanna 等组件的 API 兼容性。

## 快速开始

以下命令从仓库根目录执行，Python 环境以 **Windows PowerShell + uv** 为例。运行前需要可访问的 StarRocks 实例和模型 API；原方案笔记使用 StarRocks 4.0.2，并建议数据库环境至少分配 5 GB 内存。

### 1. 获取代码与安装环境

```powershell
git clone https://github.com/Puiching-Memory/TGAC_2025.git
cd TGAC_2025

uv venv .venv --python 3.12 --seed
.\.venv\Scripts\Activate.ps1
uv pip install -r script/requirements.txt
uv pip install requests colorama
```

`script/requirements.txt` 从本地 `lib/toon-python` 安装 TOON 实现，因此请在根目录安装依赖。`requests` 与 `colorama` 是批量客户端直接使用、但未在该清单中显式列出的依赖。使用远程模型 API 的查询流程无需部署本地 vLLM；本地模型部署可参考[历史笔记](upload/方案介绍.md#vllm推理服务器)。

### 2. 初始化数据库

连接 StarRocks 的 MySQL 协议端口，以下使用 MySQL Client；主机、用户和密码按实际部署填写：

```bash
mysql -h 127.0.0.1 -P 9030 -u root -p
```

在客户端内创建并选择数据库，再加载建表脚本：

```sql
CREATE DATABASE IF NOT EXISTS database_main;
USE database_main;
SOURCE install/create_table.sql;
SHOW TABLES;
```

核对 [install/sql_exe.py](install/sql_exe.py) 底部的数据库配置后，再执行：

```powershell
python install/sql_exe.py
```

该脚本会依次导入 `data/insert_sql.json` 中的数据，并执行 `data/final_dataset.json` 中已有的 SQL；输出为 `data/insert_exe_result.json` 和 `data/dataset_exe_result.json`。请使用专门的实验数据库，避免重复导入。

### 3. 启动交互式 Agent

先修改 [script/run_vanna_server.py](script/run_vanna_server.py) 中的 `llm` 和 `MySQLRunner` 配置。当前模型为 `deepseek-reasoner`，API 地址为 `https://api.deepseek.com`，密钥从 `OPENAI_API_KEY` 读取；数据库默认配置为 `localhost:9030/database_main`。

```powershell
$env:OPENAI_API_KEY = "填写所选模型服务的 API Key"
python script/run_vanna_server.py
```

服务入口为 [http://localhost:8000](http://localhost:8000)。ChromaDB 记忆默认写入根目录下的 `chroma_db/`；历史案例与文本知识可按需使用 `import_training_data_to_memory.py`、`import_text_memory.py` 导入，运行前核对脚本配置。

### 4. 接入结果保存与批量推理

需要自动保存执行结果时，在 `script/run_vanna_server.py` 中导入自定义工具，并用以下配置替换原有 `db_tool = RunSqlTool(...)`；数据库连接参数按实际环境填写：

```python
from vanna_hook import TGACRunSqlTool

db_tool = TGACRunSqlTool(
    sql_runner=MySQLRunner(
        host="localhost",
        database="database_main",
        user="root",
        password="",
        port=9030,
    ),
    output_path="upload/dataset_exe_result.json",
    output_text_dir="ckpt/current/results",
)
```

保留原有 `tools.register_local_tool(db_tool, ...)` 注册语句并重启服务。先用一题确认 SQL ID、SQL 和执行结果已保存，再运行批量客户端：

```powershell
# 在另一个已激活同一环境的终端运行
python script/run_vanna_client.py
```

客户端读取 `prompt/input/V1/`，默认请求 `http://localhost:8000/api/vanna/v2/chat_poll`。运行前检查文件顶部的 `CKPT_DIR`、`SERVER_URL` 和认证配置；当前 `CKPT_DIR` 指向比赛期间的 `ckpt/V6.1_33.72_1109`，该目录未随仓库发布。没有评分缓存时，客户端会请求服务处理题目。

输出文件为 `upload/dataset_exe_result.json`，字段示例见 [upload/dataset_exe_result_example.json](upload/dataset_exe_result_example.json)。执行成功不等于业务答案正确，提交前仍需逐题核对结果与格式。

### 5. 其他实验与复盘

| 用途 | 命令 / 说明 | 前提 |
| --- | --- | --- |
| 生成任务提示词 | `python script/create_input_prompt.py` | 会下载 Qwen tokenizer，并更新 `prompt/input/V1/`；保留工程师信息段 |
| 生成表级 TOON Schema | `python script/generate_mschema_toon.py` | 核对脚本顶部的数据库和 tokenizer 配置 |
| 生成字段级 TOON Schema | `python script/export_toon_schemas.py` | 使用仓库中的 Schema JSON |
| 对比版本 | `python script/compare_versions.py V8.1_38.70_1114 V8.2_40.70_1116 --simple` | 需自行提供两个 `ckpt/` 版本目录及逐题评分 |
| 汇总未通过题目 | `python script/report_incorrect_tasks.py` | 先修改脚本中固定的版本目录，并提供 `score.csv` |
| LangChain 交互实验 | `python script/langchain/text2sql_agent.py` | 先按[独立说明](script/langchain/README.md)安装依赖和配置模型 |

------

## 数据与任务

| 资料 | 位置 | 内容 |
| --- | --- | --- |
| 任务数据 | [data/final_dataset.json](data/final_dataset.json) | 问题、SQL ID、涉及表、补充知识与部分已有 SQL |
| 数据库结构 | [data/schema.json](data/schema.json) · [data/mschema_database_main.json](data/mschema_database_main.json) | 原始 Schema 与 M-Schema 表达 |
| 业务知识 | [data/common_knowledge.md](data/common_knowledge.md) · [data/补充知识.md](data/补充知识.md) | 通用口径与补充说明 |
| 建表与数据导入 | [install/create_table.sql](install/create_table.sql) · [data/insert_sql.json](data/insert_sql.json) | StarRocks 表定义与插入语句 |
| 任务提示词 | [prompt/input/V1](prompt/input/V1) | 按题目组织的提示词与人工补充 |
| 提交格式示例 | [upload/dataset_exe_result_example.json](upload/dataset_exe_result_example.json) | SQL、SQL ID 与执行结果的字段示例 |

这些文件是仓库保留的比赛资料快照。`ckpt/` 历史版本、`chroma_db/` 记忆库、LightRAG 索引和完整提交结果需要在本地准备或重新生成；克隆仓库不会恢复比赛期间的全部实验状态。

## 成绩与版本记录

队伍：**proximusA** · 参赛队伍数：**> 140** · 最终排名：**54**

以下版本、得分和日期沿用原 README 的参赛记录，最高记录得分为 **40.70**。表中得分为历史提交记录，未据此推断每个版本的具体改动或消融收益。

| 版本 | 得分 | 日期 |
| --- | --- | --- |
| V9 | 40.70 | 2025-12-05 |
| V8.2 | 40.70 | 2025-11-16 |
| V8.1 | 38.70 | 2025-11-14 |
| V8 | 36.05 | 2025-11-13 |
| V7.1 | 34.88 | 2025-11-12 |
| V6.1 | 33.72 | 2025-11-09 |
| V7 | 22.09 | 2025-11-07 |
| V6 | 31.40 | 2025-11-05 |
| v5 | 20.93 | 2025-11-04 |
| v4 | 18.60 | 2025-11-03 |
| v3 | 17.44 | 2025-11-03 |
| v2 | 13.95 | 2025-11-02 |

## 仓库结构

```text
TGAC_2025/
├── data/                    # 任务、Schema、业务知识与导入数据
├── install/                 # StarRocks 建表和 SQL 执行脚本
├── lib/
│   ├── M-Schema/            # M-Schema 相关代码
│   └── toon-python/         # TOON Python 实现
├── prompt/input/V1/         # 逐题提示词
├── script/
│   ├── run_vanna_server.py  # Vanna 服务入口
│   ├── run_vanna_client.py  # 批量推理与正确答案缓存复用
│   ├── vanna_hook.py        # SQL 结果保存与诊断扩展
│   ├── vannna_tools.py      # LightRAG 查询工具（文件名为三个 n）
│   ├── langchain/          # 独立的 LangChain 实验
│   └── ...                 # Schema、记忆导入和复盘工具
├── upload/                 # 方案笔记与提交格式示例
└── LICENSE
```

## 社区与相关工作

资料核对日期：**2026-10-19**。以下按同赛道方案、技术组件与研究基准分类；外部方案的奖项以仓库自述为准，未与官方榜单独立核验。研究基准和通用工具的指标不与本仓库历史成绩直接比较。

### 赛事与同赛道公开方案

- [TGAC 腾讯游戏算法大赛](https://tgac.tencent.com/) — 官方赛事入口；本次核对时页面抓取超时，未据此补写官方榜单或规则。
- [terra901/TAGC-Data-Intelligence-Decision-Science](https://github.com/terra901/TAGC-Data-Intelligence-Decision-Science/blob/main/README.zh.md) — 仓库自述为 TGAC 2025 数智决策科学赛道二等奖方案；提供脱敏代码，覆盖增强 Schema、隐式 Join Graph、多策略候选生成、执行修复与结果裁决。
- [gstranded/tgac-2025-proof](https://github.com/gstranded/tgac-2025-proof) — 与上一项同为“帮帮我！肯德基爷爷”队伍的方案与获奖材料，附架构文档和脱敏源码快照，作为补充资料入口。
- [KailBug/TGAC2025](https://github.com/KailBug/TGAC2025) — 明确面向第三赛道的公开实现，包含 Schema 检索、双模型分级生成、SQL 执行与修正；未在此标注名次。

### Agent、检索与 Schema 工具

- [Vanna](https://github.com/vanna-ai/vanna) — 本仓库 Agent 服务使用的 Text-to-SQL 框架，可参考工具注册、数据库接入与记忆机制。
- [LightRAG](https://github.com/HKUDS/LightRAG) — 图结构与向量检索框架，对应本仓库预留的 HTTP 检索工具。
- [M-Schema](https://github.com/XGenerationLab/M-Schema) — 半结构化数据库 Schema 表达，对应 `lib/M-Schema` 与 Schema 转换工具。
- [TOON Python](https://github.com/toon-format/toon-python) — 面向 LLM 上下文的结构化文本序列化实现，对应 `lib/toon-python`。
- [Datus Agent](https://github.com/Datus-ai/Datus-agent) — 数据 Agent 与上下文工程参考；本仓库依赖清单仅保留了已注释的探索项，默认流程未使用。

### SQL 模型与研究基准

- [XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL) — 自然语言到 SQL 的多生成器方案，可参考候选生成与 Schema 表达。
- [XiYanSQL-QwenCoder-7B-2504](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-7B-2504) — 原部署笔记使用的 SQL 模型，模型卡提供 Transformers 与 vLLM 使用说明。
- [BIRD](https://bird-bench.github.io/) — Text-to-SQL 研究基准，可补充执行正确性与外部知识相关的评估视角。
- [Spider 2.0](https://github.com/xlang-ai/Spider2) — 面向真实企业工作流的 Text-to-SQL 基准，可参考复杂环境中的任务组织与 Agent 评估。

### 数据库与部署资料

- [StarRocks Docker 部署与 SQL 客户端连接](https://docs.starrocks.io/docs/quick_start/shared-nothing/) — 官方部署及连接说明，MySQL 协议端口与本仓库配置对应。
- [仓库内的比赛环境笔记](upload/方案介绍.md) — StarRocks 初始化和比赛期间的本地模型部署记录；历史兼容性描述需结合当前组件版本核对。

### 交流与问题反馈

- [本仓库 Issues](https://github.com/Puiching-Memory/TGAC_2025/issues) — 讨论复现问题、Schema 处理、提示词与实验结果。
- [Vanna Issues](https://github.com/vanna-ai/vanna/issues) · [LightRAG Issues](https://github.com/HKUDS/LightRAG/issues) · [XiYan-SQL Issues](https://github.com/XGenerationLab/XiYan-SQL/issues) — 查询上游 API 兼容、检索和 SQL 模型相关问题。

## 许可证

本仓库采用 [GNU GPL v3.0](LICENSE)。`lib/` 中的第三方组件保留各自许可证，使用时请同时查看相应目录的许可文件。
