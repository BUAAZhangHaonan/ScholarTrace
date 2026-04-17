# ScholarTrace

[English](README.md) | **中文**

> 基于主题文档的学术检索、缓存证据访问与显式全文获取系统

## 概览

ScholarTrace 会把一份研究简报变成排序后的论文、可缓存的证据，以及可导出的报告。

- **统一检索**：默认接入 6 个核心学术源：OpenAlex、arXiv、Semantic Scholar、DBLP、OpenReview、Crossref。
- **DeepXiv 按配置加入统一检索**：只要设置了 `SCHOLARTRACE_DEEPXIV_TOKENS`，或显式开启自动注册并提供 `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET`，DeepXiv 就会作为正常检索源进入同一条 fan-out、去重、排序和存储路径。
- **只有一条排序路径**：所有候选论文都走同一套去重、来源合并和综合排序逻辑。DeepXiv 不是旁路排序系统。
- **缓存优先的全文模型**：`GET /papers/{paper_id}/fulltext` 和 `get_paper_fulltext` 只读取缓存状态。缺失全文只能通过显式 acquire 路径获取。
- **保留直接 DeepXiv 证据访问**：专门的 DeepXiv REST 端点和 MCP 工具仍然存在，但它们是直接读取 DeepXiv，不是 ScholarTrace 的缓存全文读取。
- **面向 LLM 的接口**：提供 REST API 和 13 个工具的 MCP 服务器。MCP 默认使用本地 `stdio`，SSE 是可选且需要令牌。
- **BigModel GLM 示例**：`examples/glm_scholar_search.py` 默认使用 `glm-5-turbo`，并且会把每次单独请求都限制在模型上下文窗口之内。

## 快速开始

```bash
conda create -n ScholarTrace python=3.13 -y
conda activate ScholarTrace

cd ScholarTrace
python -m pip install -r requirements-dev.txt

scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q

cp .env.example .env
# 按你的本机环境编辑 .env

scholartrace-api
# -> http://127.0.0.1:9000

scholartrace-mcp
# -> 本地 stdio MCP 服务器
```

当前本地校验会收集 **182 个测试**。

## 配置

所有运行时配置都使用 `SCHOLARTRACE_` 前缀。`.env` 只建议用于本地开发。部署服务时，请把密钥放在仓库外部。

| 变量 | 必需 | 默认值 | 用途 |
|---|---|---|---|
| `SCHOLARTRACE_API_HOST` | 否 | `127.0.0.1` | REST 绑定地址 |
| `SCHOLARTRACE_API_PORT` | 否 | `9000` | REST 端口 |
| `SCHOLARTRACE_MCP_HOST` | 否 | `127.0.0.1` | MCP SSE 绑定地址 |
| `SCHOLARTRACE_MCP_PORT` | 否 | `8001` | MCP SSE 端口 |
| `SCHOLARTRACE_MCP_TRANSPORT` | 否 | `stdio` | MCP 传输方式：`stdio` 或 `sse` |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | 否 | `false` | 非回环地址监听前必须显式开启 |
| `SCHOLARTRACE_ACCESS_TOKEN` | 远程时必需 | | REST 与 MCP SSE 共用 Bearer Token |
| `SCHOLARTRACE_SEMANTIC_SCHOLAR_API_KEY` | 否 | | Semantic Scholar 可选高配额密钥 |
| `SCHOLARTRACE_OPENALEX_MAILTO` | 否 | | OpenAlex polite-pool 邮箱 |
| `SCHOLARTRACE_CROSSREF_MAILTO` | 否 | | Crossref polite-pool 邮箱 |
| `SCHOLARTRACE_MAX_RESULTS_PER_SOURCE_PER_QUERY` | 否 | `200` | 每个来源的检索上限 |
| `SCHOLARTRACE_TARGET_CANDIDATE_POOL` | 否 | `500` | 合并后候选池目标大小 |
| `SCHOLARTRACE_MAX_FULLTEXT_DOWNLOADS` | 否 | `50` | 每次检索允许的全文下载上限 |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | 仅示例需要 | | GLM 示例脚本和 DeepXiv Agent 筛选使用的 BigModel 密钥 |
| `SCHOLARTRACE_BIGMODEL_BASE_URL` | 否 | `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` | BigModel 接口地址 |
| `SCHOLARTRACE_BIGMODEL_MODEL` | 否 | `glm-5-turbo` | 默认 GLM 模型 |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | 仅 DeepXiv 需要 | | 逗号分隔的 DeepXiv Token |
| `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER` | 否 | `false` | 显式开启自动注册 |
| `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` | 仅自动注册时需要 | | 自动注册开启时使用的 SDK Secret |

## 运行时模型

### 统一检索与 DeepXiv

主题检索只有一条主路径：

1. 把主题文档解析成查询
2. 把每个查询 fan-out 到已配置的连接器
3. 用稳定标识和模糊标题匹配合并重复论文
4. 对合并后的论文排序
5. 持久化 canonical work、关联、artifact 和 section

当 DeepXiv 已配置时，它会进入这条主路径。没有配置时，ScholarTrace 会保持原来的 6 源流程，并且干净地跳过 DeepXiv。

### 缓存读取与显式 acquire

ScholarTrace 现在只有一种清晰的全文模型：

1. **先读缓存状态**：`GET /papers/{paper_id}/fulltext` 或 `get_paper_fulltext`
2. **缺失时显式获取**：`POST /papers/{paper_id}/fulltext/acquire` 或 `acquire_paper_fulltext`
3. **再读一次缓存状态**：`GET /papers/{paper_id}/fulltext` 或 `get_paper_fulltext`

关键点：

- cache-only 读取不会触发网络获取
- 只有显式 acquire 才会触发网络工作
- 会先尝试公开来源的全文路径
- 对 arXiv 论文，如果 DeepXiv 已配置，显式 acquire 时可以把 DeepXiv markdown 作为后备路径
- 昂贵操作有预算和限流；缓存读取是轻量操作

### 直接 DeepXiv 读取

专门的 DeepXiv REST 端点和 MCP 工具仍然很有用，但它们和 ScholarTrace 缓存读取不是一回事：

- `GET /deepxiv/papers/{arxiv_id}/fulltext` 和 `deepxiv_paper_fulltext` 返回的是直接 DeepXiv markdown
- 它们不能替代 `GET /papers/{paper_id}/fulltext`
- 它们适合做直接 arXiv 证据访问、摘要、章节和 Agent 筛选

## REST API

### 核心 REST 端点

```text
GET  /health
POST /themes
POST /retrieval/jobs
GET  /retrieval/jobs/{job_id}
GET  /themes/{theme_id}/papers
GET  /papers/{paper_id}
GET  /papers/{paper_id}/sections
GET  /papers/{paper_id}/fulltext
POST /papers/{paper_id}/fulltext/acquire
GET  /themes/{theme_id}/export
```

### 直接 DeepXiv REST 端点

```text
POST /deepxiv/search
GET  /deepxiv/papers/{arxiv_id}/summary
GET  /deepxiv/papers/{arxiv_id}/fulltext
GET  /deepxiv/papers/{arxiv_id}/sections/{section_name}
POST /deepxiv/agent/filter
```

### 端到端 REST 工作流

客户端或脚本的标准 REST 流程如下：

```bash
# 1. 创建主题
curl -s -X POST http://127.0.0.1:9000/themes \
  -F 'text=RLHF sycophancy and affective hallucination in language models'

# 2. 启动检索
curl -s -X POST http://127.0.0.1:9000/retrieval/jobs \
  -F 'theme_id=<theme-id>'

# 3. 轮询作业状态
curl -s http://127.0.0.1:9000/retrieval/jobs/<job-id>

# 4. 读取排序后的论文
curl -s 'http://127.0.0.1:9000/themes/<theme-id>/papers?limit=20'

# 5. 读取缓存全文状态
curl -s http://127.0.0.1:9000/papers/<paper-id>/fulltext

# 6. 需要时显式获取全文
curl -s -X POST http://127.0.0.1:9000/papers/<paper-id>/fulltext/acquire

# 7. 再次读取缓存状态
curl -s http://127.0.0.1:9000/papers/<paper-id>/fulltext
```

全文返回结果会告诉你论文是否已缓存、是否仍然需要获取，以及上一次失败是否进入了 negative cache 窗口。

## MCP 服务器

ScholarTrace 通过 MCP 暴露 **13 个工具**。本地默认是 `stdio`。只有显式开启并提供访问令牌时，才会启用 SSE。

| # | 工具 | 用途 |
|---|---|---|
| 1 | `search_papers_by_theme` | 解析主题文档，运行统一检索，排序，并返回前几篇论文 |
| 2 | `get_ranked_papers` | 读取已存储主题的排序论文 |
| 3 | `get_paper_metadata` | 读取单篇论文的脱敏公开元数据 |
| 4 | `get_paper_sections` | 读取已缓存的章节内容 |
| 5 | `get_paper_fulltext` | 只读取缓存全文状态 |
| 6 | `acquire_paper_fulltext` | 显式获取全文，然后返回刷新后的缓存状态 |
| 7 | `get_related_papers` | 按期刊和年份读取相关论文 |
| 8 | `export_theme_report` | 导出 JSON 或 Markdown 报告 |
| 9 | `deepxiv_search` | 通过 DeepXiv 搜索 arXiv |
| 10 | `deepxiv_paper_summary` | 读取直接 DeepXiv 元数据和 TLDR |
| 11 | `deepxiv_paper_fulltext` | 读取直接 DeepXiv markdown 全文 |
| 12 | `deepxiv_paper_section` | 读取单个直接 DeepXiv 章节 |
| 13 | `deepxiv_agent_filter` | 先用 DeepXiv 搜索，再用 GLM Agent 做筛选 |

### 本地 `stdio` 与可选 SSE

推荐的本地模式：

```bash
scholartrace-mcp
```

可选的网络模式：

```bash
SCHOLARTRACE_MCP_TRANSPORT=sse \
SCHOLARTRACE_MCP_HOST=0.0.0.0 \
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true \
SCHOLARTRACE_ACCESS_TOKEN=change-me \
scholartrace-mcp
```

只有在你确实需要网络端点时才使用 SSE。没有显式开启远程访问或没有令牌时，远程启动会被拒绝。

### ChatBox 风格的 MCP 工作流

对 ChatBox 或任何其他 MCP 客户端，标准流程都是：

1. 用完整研究简报调用 `search_papers_by_theme`
2. 用返回的 `theme_id` 调用 `get_ranked_papers`
3. 对目标论文调用 `get_paper_fulltext`，先检查缓存状态
4. 如果 `needs_acquisition` 为 `true`，调用 `acquire_paper_fulltext`
5. 再次调用 `get_paper_fulltext`，读取更新后的缓存

这就是主 MCP 工作流。只有在你明确需要直接 DeepXiv 搜索、摘要、markdown 全文或章节访问时，才使用专门的 DeepXiv 工具。

## 示例脚本

`examples/glm_scholar_search.py` 现在完全跟随 API 的运行模型：

1. 创建主题
2. 启动统一检索
3. 读取排序后的论文
4. 读取缓存全文状态
5. 显式获取缺失全文
6. 再次读取缓存状态
7. 用 BigModel GLM 总结论文

说明：

- 默认模型仍然是 `glm-5-turbo`
- 必须设置 `SCHOLARTRACE_BIGMODEL_API_KEY`
- 不再使用仓库内置默认密钥
- prompt 是按单次请求做边界控制，不是全局硬限制
- 脚本会裁剪消息历史并打包论文批次，确保一次调用不会超出模型上下文窗口

示例脚本中的交互命令：

- `papers`：显示当前排序列表
- `fulltext N`：读取第 `N` 篇论文的缓存状态
- `acquire N`：显式获取第 `N` 篇论文的全文，然后再次读取缓存
- `chat`：进入带边界控制的 GLM 交互模式

## 架构摘要

```text
主题文档
    -> 解析后的查询
    -> 已配置来源上的统一 fan-out
    -> 去重 + 来源合并
    -> 综合排序
    -> canonical 存储
    -> 已缓存章节 / 已缓存全文状态

默认来源：
    OpenAlex、arXiv、Semantic Scholar、DBLP、OpenReview、Crossref

按配置加入的来源：
    DeepXiv

显式证据路径：
    先读缓存
    -> 显式 acquire
    -> 再读缓存
```

## 验证

常用本地检查：

```bash
scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q
python -m compileall scholartrace examples/glm_scholar_search.py
```

## 许可证

MIT
