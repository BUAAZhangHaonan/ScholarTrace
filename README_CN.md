# ScholarTrace

[English](README.md) | **中文**

> 基于主题文档的多源学术论文发现与全文获取系统

## 功能特性

- **多源检索**：支持 7 个学术数据库 —— OpenAlex、arXiv、Semantic Scholar、DBLP、OpenReview、Crossref、DeepXiv
- **主题文档解析**：理解完整的研究简报，而非仅关键词搜索
- **多键去重**：精确 ID 匹配（DOI、arXiv ID、S2 ID 等）+ 模糊标题匹配（阈值 0.85）
- **多目标排序**：相关性（TF-IDF）、时效性（指数衰减）、影响力（对数归一化引用）、期刊质量、开放获取加分、来源一致性
- **全文获取级联**：arXiv HTML → arXiv PDF（PyMuPDF）→ OA URL → 仅摘要回退
- **DeepXiv 集成**：混合 BM25 + 向量搜索、论文元数据与 TLDR、全文提取、基于 Agent 的智能筛选
- **双接口**：REST API（FastAPI，端口 8000）和 MCP 服务器（SSE 传输，端口 8001）
- **BigModel GLM 集成**：使用 `glm-5-turbo` 进行智能文献分析

## 快速开始

```bash
# 创建 conda 环境
conda create -n ScholarTrace python=3.13 -y
conda activate ScholarTrace

# 安装
cd ScholarTrace
pip install -e ".[dev]"

# 配置 API 密钥
cp .env.example .env
# 编辑 .env 填入你的 API 密钥

# 运行测试（117 个测试）
pytest tests/ -v

# 启动 REST API
scholartrace-api
# -> http://localhost:8000

# 启动 MCP 服务器（用于 LLM 客户端集成）
scholartrace-mcp
```

## 配置（.env）

所有配置使用 `SCHOLARTRACE_` 前缀。复制 `.env.example` 到 `.env` 并填入你的值。

| 变量 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `SCHOLARTRACE_SEMANTIC_SCHOLAR_API_KEY` | 否 | | Semantic Scholar API 密钥（提高速率限制） |
| `SCHOLARTRACE_OPENALEX_MAILTO` | 否 | | OpenAlex 礼貌池邮箱 |
| `SCHOLARTRACE_CROSSREF_MAILTO` | 否 | | Crossref 礼貌池邮箱 |
| `SCHOLARTRACE_API_HOST` | 否 | `127.0.0.1` | REST API 绑定地址 |
| `SCHOLARTRACE_API_PORT` | 否 | `8000` | REST API 端口 |
| `BIGMODEL_API_KEY` | 否 | | BigModel GLM API 密钥 |
| `BIGMODEL_BASE_URL` | 否 | | BigModel GLM API 端点 |
| `BIGMODEL_MODEL` | 否 | `glm-5-turbo` | GLM 模型名称 |

## REST API 端点

```
GET  /health                                    — 健康检查
POST /themes                                    — 从文本创建主题
POST /retrieval/jobs                            — 启动检索（后台）
GET  /retrieval/jobs/{job_id}                   — 作业状态
GET  /themes/{theme_id}/papers                  — 排序后的论文（分页）
GET  /papers/{paper_id}                         — 论文元数据
GET  /papers/{paper_id}/sections                — 章节级内容
GET  /papers/{paper_id}/fulltext                — 全文访问（触发级联）
GET  /themes/{theme_id}/export                  — 导出（JSON/Markdown）

# DeepXiv 端点
POST /deepxiv/search                            — 通过 DeepXiv 搜索 arXiv
GET  /deepxiv/papers/{arxiv_id}/summary         — 论文摘要和 TLDR
GET  /deepxiv/papers/{arxiv_id}/fulltext        — DeepXiv 全文
GET  /deepxiv/papers/{arxiv_id}/sections/{name} — 指定章节内容
POST /deepxiv/agent/filter                      — Agent 智能筛选
```

## MCP 服务器

MCP 服务器提供 12 个工具用于 LLM 代理集成：

### 原生 ScholarTrace 工具

| # | 工具 | 说明 |
|---|---|---|
| 1 | `search_papers_by_theme` | 完整流程：解析主题 → 检索 → 排序 → 返回前 10 |
| 2 | `get_ranked_papers` | 获取已存储主题的排序论文 |
| 3 | `get_paper_metadata` | 按 ID 获取完整论文元数据 |
| 4 | `get_paper_sections` | 章节级内容提取 |
| 5 | `get_paper_fulltext` | 全文（触发下载级联） |
| 6 | `get_related_papers` | 按共享期刊和年份查找相关论文 |
| 7 | `export_theme_report` | 导出完整报告（JSON 或 Markdown） |

### DeepXiv 工具

| # | 工具 | 说明 |
|---|---|---|
| 8 | `deepxiv_search` | 通过 DeepXiv 搜索 arXiv（混合 BM25 + 向量） |
| 9 | `deepxiv_paper_summary` | 论文元数据和 TLDR 摘要 |
| 10 | `deepxiv_paper_fulltext` | DeepXiv 全文（Markdown 格式） |
| 11 | `deepxiv_paper_section` | 获取论文指定章节 |
| 12 | `deepxiv_agent_filter` | GLM Agent 智能筛选：搜索 + 评分 + 过滤 |

### MCP 客户端配置

MCP 服务器使用 SSE 传输，支持局域网访问。在 `.env` 中设置 `SCHOLARTRACE_MCP_HOST=0.0.0.0` 即可接受来自其他机器的连接。

**Claude Desktop** — 添加到 `claude_desktop_config.json`（本机）：

```json
{
  "mcpServers": {
    "scholartrace": {
      "command": "conda",
      "args": ["run", "-n", "ScholarTrace", "scholartrace-mcp"]
    }
  }
}
```

**局域网 / 远程访问** — 通过 SSE URL 连接（网络中的任何机器）：

```json
{
  "mcpServers": {
    "scholartrace": {
      "url": "http://192.168.x.x:8001/sse"
    }
  }
}
```

## DeepXiv 集成

DeepXiv（data.rag.ac.cn）提供增强的 arXiv 访问能力：

- **混合搜索**：BM25 + 向量检索，比纯关键词搜索更精准
- **论文元数据**：标题、摘要、作者、章节 TLDR
- **全文提取**：完整的 Markdown 格式论文文本
- **Semantic Scholar 代理**：通过 DeepXiv 免费访问 Semantic Scholar
- **自动注册**：Token 池自动注册和轮换，无需手动配置
- **Agent 筛选**：使用 GLM 对论文进行相关性、新颖性、质量评分

```python
from scholartrace.deepxiv import DeepXivReader, DeepXivAgent

# 搜索论文
reader = DeepXivReader()
results = await reader.search("RLHF sycophancy", size=20, search_mode="hybrid")

# 获取全文
text = await reader.raw("2301.12345")

# Agent 筛选
agent = DeepXivAgent(api_key="your-key")
filtered = await agent.filter_papers(papers, "研究问题")
```

## 架构

```
主题文档 → 主题解析 → 多源并行检索 → 去重 → 排序 → 全文 → 存储
         │                                    │        │        │        │
    parsed_queries     ┌──────────────────┐  Union-Find  多目标    级联    SQLite
    parsed_topics      │ OpenAlex         │  + rapidfuzz  评分   BS4/PyMuPDF
    parsed_methods     │ arXiv            │
    parsed_datasets    │ Semantic Scholar  │
                       │ DBLP             │
                       │ OpenReview       │
                       │ Crossref         │
                       │ DeepXiv (BM25+向量)│
                       └──────────────────┘
                                                    REST API (FastAPI) + MCP Server
```

## 开机自启（systemd）

```bash
# 安装服务
sudo cp scripts/scholartrace-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable scholartrace-mcp
sudo systemctl start scholartrace-mcp

# 查看状态
sudo systemctl status scholartrace-mcp
```

## 许可证

MIT
