# ScholarTrace

[English](README.md) | **中文**

> 面向 ChatBox 的干净 LAN SSE 学术 MCP，只有两个公开工具，并且对全文状态保持诚实。

## 概览

ScholarTrace 会接收一份主题文档，做多源论文检索和二阶段重排，然后只在需要时让 MCP 客户端继续深读。

- **2 个公开 MCP 工具**：`query` 和 `read`
- **主部署模式**：面向团队共享的 LAN SSE
- **本地调试专用**：`stdio`
- **默认二阶段模型**：`glm-5-turbo`
- **默认二阶段候选数**：`agent_candidate_limit=100`
- **默认最终返回数**：`final_limit=20`

这一轮只收敛 MCP 产品表面。REST 现在仍然保持 broad。

## LAN SSE 快速开始

仓库根目录的 tmux 脚本就是主操作路径：

- 如果仓库根目录存在 `.env`，脚本会自动加载。
- 旧 `.env` 里如果还是 `BIGMODEL_API_KEY`、`BIGMODEL_BASE_URL`、`BIGMODEL_MODEL`，脚本会在启动时自动规范化成运行时使用的 `SCHOLARTRACE_*` 变量名。
- 如果没有显式设置，脚本会默认使用 `SCHOLARTRACE_MCP_TRANSPORT=sse`、`SCHOLARTRACE_MCP_HOST=0.0.0.0`、`SCHOLARTRACE_MCP_PORT=8001`、`SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true`、`SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`。
- 如果 `.env` 加载后仍然没有 `SCHOLARTRACE_BIGMODEL_API_KEY`，脚本会直接失败并给出清楚提示。
- 如果 `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true`，但没有 `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET`，脚本也会直接失败并给出清楚提示。
- 如果 DeepXiv 本身没有配置，脚本不会阻止 ScholarTrace 启动，只会提示 DeepXiv 检索、直接证据和 markdown fallback 不可用。

示例 `.env`：

```bash
SCHOLARTRACE_BIGMODEL_API_KEY=<your-bigmodel-key>
SCHOLARTRACE_ACCESS_TOKEN=g203-mcp

# 可选：脚本本身已经默认这些 LAN SSE 值
SCHOLARTRACE_MCP_TRANSPORT=sse
SCHOLARTRACE_MCP_HOST=0.0.0.0
SCHOLARTRACE_MCP_PORT=8001
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true

# 可选：DeepXiv tokens
# SCHOLARTRACE_DEEPXIV_TOKENS=token-a,token-b

# 可选：DeepXiv auto-register
# SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true
# SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET=<real-sdk-secret-from-deepxiv>
```

```bash
./run_scholartrace_mcp_sse.sh
./status_scholartrace_mcp_sse.sh
./stop_scholartrace_mcp_sse.sh
```

默认 tmux session 名字是 `scholartrace_mcp_sse`。

常用检查命令：

```bash
tmux attach -t scholartrace_mcp_sse
tmux capture-pane -pt scholartrace_mcp_sse
ss -ltnp | grep ':8001'
```

局域网客户端使用这个地址：
- `http://172.17.194.210:8001/sse`

这个 token 是用户自己定义的，不是自动生成的。

MCP 客户端必须发送：

- `Authorization: Bearer g203-mcp`

实际示例里，请设置 `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`。

`SCHOLARTRACE_BIGMODEL_API_KEY` 来自 `.env` 或环境变量。MCP 客户端不会在请求参数里传这个 key。

## ChatBox JSON

把下面这段直接粘贴或导入 ChatBox：

```json
{
  "mcpServers": {
    "scholartrace": {
      "url": "http://172.17.194.210:8001/sse",
      "headers": {
        "Authorization": "Bearer g203-mcp"
      }
    }
  }
}
```

如果 ChatBox 仍然提示“未从剪贴板解析到MCP服务器”，可以改用一键安装链接导入：

```text
chatbox://mcp/install?server=eyJtY3BTZXJ2ZXJzIjp7InNjaG9sYXJ0cmFjZSI6eyJ1cmwiOiJodHRwOi8vMTcyLjE3LjE5NC4yMTA6ODAwMS9zc2UiLCJoZWFkZXJzIjp7IkF1dGhvcml6YXRpb24iOiJCZWFyZXIgZzIwMy1tY3AifX19fQ==
```

## DeepXiv 行为说明

DeepXiv 是可选的。

如果 DeepXiv 已经配置：

- 统一检索会加入 DeepXiv 源
- 对有 arXiv 背景的论文，`read` 的 `direct_evidence` 可以返回 DeepXiv metadata 和 brief
- 显式全文获取在公共 URL 路径失败后，可以继续尝试 DeepXiv markdown fallback

如果 DeepXiv 没有配置：

- ScholarTrace 仍然可以启动
- `query` 仍然可以工作
- 内置 `glm-5-turbo` 重排仍然可以工作
- DeepXiv 的检索补充会被跳过
- `direct_evidence` 可能返回 `available=false`
- 显式全文获取里的 DeepXiv markdown fallback 可能不可用

如果你要用 `auto-register`：

- ScholarTrace 可以自动生成用户名和邮箱
- `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` 必须已经来自 DeepXiv 服务侧或之前的部署配置
- 代码自己找不到这个 SDK secret

## 公共 MCP 表面

ScholarTrace 现在只暴露两个公开 MCP 工具。

### `query`

推荐调用：

```json
{
  "theme_document": "你的主题文档文本",
  "final_limit": 20,
  "agent_candidate_limit": 100,
  "coarse_pool_limit": 500,
  "include_rationale": true
}
```

`query` 的默认流程：

1. 解析主题文档
2. 在已配置学术源上做统一检索
3. 对原始候选去重
4. 做第一阶段综合排序
5. 保留 coarse candidate pool
6. 用内置 DeepXiv Agent 和 `glm-5-turbo` 做第二阶段重排
7. 返回最终选中的论文

关键点：

- DeepXiv 是可选的。如果 DeepXiv 没有配置，ScholarTrace 仍然可以启动，`query` 仍然可以工作，内置 `glm-5-turbo` 重排仍然可以工作，但 DeepXiv 的检索补充、direct evidence 和 markdown fallback 可能不可用。
- DeepXiv Agent 不再是正常流程里单独手动调用的 MCP 步骤
- 如果不传 `final_limit`，`query` 默认返回 20 篇
- 如果客户端要更多结果，ScholarTrace 会在可能时返回更多
- `agent_candidate_limit` 默认是 100
- `coarse_pool_limit` 是可选参数

`query` 的返回会包含：

- `theme_id`
- `total_retrieved`
- `total_after_dedup`
- `total_after_first_stage`
- `total_agent_candidates`
- `total_final`
- `papers`

每篇论文摘要至少包含：

- `paper_id`
- `title`
- `authors`
- `year`
- `venue`
- `abstract`
- `composite_score`
- `agent_score`
- `agent_rank`
- `rationale`
- `fulltext_status`

### `read`

`read` 是唯一的分层读取工具。

支持的深度：

- `summary`
- `sections`
- `fulltext_status`
- `fulltext`
- `direct_evidence`

标准 MCP 流程：

1. 先调用 `query`
2. 从返回结果里选一篇论文
3. 再调用 `read`
4. 如果 `fulltext_status` 说明全文还没缓存，就再次调用 `read`，并设置 `allow_acquire=true`

例子：

```json
{
  "paper_id": "paper-id-from-query",
  "depth": "fulltext",
  "allow_acquire": true
}
```

每个深度的含义：

- `summary`：元数据、摘要、排序状态、agent 状态、紧凑全文状态
- `sections`：只返回缓存章节
- `fulltext_status`：返回真实缓存状态和获取状态
- `fulltext`：如果有缓存解析文本就返回；如果没有且 `allow_acquire=true`，就触发显式获取并返回结果状态
- `direct_evidence`：对有 arXiv 背景的论文，返回直接 DeepXiv 元数据和 brief，但仍然留在同一个 `read` 工具里

## 全文能力说明

ScholarTrace 的显式获取路径是真实可用的，不是空壳。

现在显式获取路径按这个顺序尝试：

1. arXiv HTML
2. arXiv PDF
3. 元数据里的 `pdf_url`
4. 元数据里的 `oa_url` 或 `html_url`
5. DeepXiv markdown fallback when DeepXiv is configured

目前确认真实可用的路径：

- `arXiv HTML` 抓取，加上基于标题的章节切分
- `arXiv PDF` 抓取，加上纯文本提取
- 元数据 PDF 和 HTML 抓取
- DeepXiv markdown fallback with heading-based section parsing when configured
- 获取失败时的显式 negative cache 状态

当前仍然存在的限制：

- 新论文全文获取并不保证成功
- PDF 解析现在主要还是纯文本提取，不是强结构恢复
- 没有 OCR，扫描版或图片版 PDF 不行
- HTML 解析是简单的标题切分
- markdown fallback 解析也是简单的标题切分
- 检索流程本身不会自动下载全文

这也是为什么 `read` 要保持诚实：

- 有缓存全文就返回
- 只有章节就返回章节
- 只有摘要和元数据就明确说明
- 显式获取失败就清楚说明失败

## REST API

这一轮 REST 仍然保持 broad。

核心 REST 端点：

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

显式全文 REST 流程仍然是：

1. `GET /papers/{paper_id}/fulltext`
2. `POST /papers/{paper_id}/fulltext/acquire`
3. `GET /papers/{paper_id}/fulltext`

## 配置说明

关键运行时变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `SCHOLARTRACE_MCP_TRANSPORT` | `stdio` | 局域网部署时改成 `sse` |
| `SCHOLARTRACE_MCP_HOST` | `127.0.0.1` | 局域网部署时改成 `0.0.0.0` |
| `SCHOLARTRACE_MCP_PORT` | `8001` | MCP SSE 端口 |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | `false` | 非回环地址监听时必须设为 `true` |
| `SCHOLARTRACE_ACCESS_TOKEN` | | 网络 MCP 的共享 bearer token |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | | 来自 `.env` 或环境变量；MCP 客户端不会在请求参数里传这个 key |
| `SCHOLARTRACE_BIGMODEL_MODEL` | `glm-5-turbo` | 默认重排模型 |
| `SCHOLARTRACE_AGENT_CANDIDATE_LIMIT` | `100` | 默认第二阶段候选数 |
| `SCHOLARTRACE_FINAL_LIMIT` | `20` | 默认最终返回数 |
| `SCHOLARTRACE_TARGET_CANDIDATE_POOL` | `500` | 重排前默认 coarse pool |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | | 可选的 DeepXiv tokens，用于检索、直接证据和 markdown fallback |
| `SCHOLARTRACE_DEEPXIV_AUTO_REGISTER` | `false` | 可选的 DeepXiv auto-register 开关 |
| `SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET` | | DeepXiv auto-register 需要；代码自己找不到这个 secret |

## `stdio` 调试模式

`stdio` 仍然保留，但只作为本地调试或开发模式。

```bash
SCHOLARTRACE_MCP_TRANSPORT=stdio scholartrace-mcp
```

不要把 `stdio` 当成团队共享部署的主故事。主故事就是 LAN SSE。

## systemd 管理示例

仓库里的 `scripts/scholartrace-mcp.service` 现在是一个次级 managed-deployment 示例。

日常启动和停止请用上面的 tmux 脚本。需要受管服务时再用 systemd。

把运行时密钥放在 systemd 的 `EnvironmentFile` 里：

- `/etc/scholartrace/scholartrace.env`

关键 SSE 值是：

- `SCHOLARTRACE_MCP_TRANSPORT=sse`
- `SCHOLARTRACE_MCP_HOST=0.0.0.0`
- `SCHOLARTRACE_MCP_PORT=8001`
- `SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true`
- `SCHOLARTRACE_ACCESS_TOKEN=g203-mcp`
- `SCHOLARTRACE_BIGMODEL_API_KEY=<your-bigmodel-key>`

## 验证

常用本地检查：

```bash
scholartrace-check-env --include-dev --pytest-collect
pytest tests/ -q
python -m compileall scholartrace examples/glm_scholar_search.py
```

## 许可证

MIT
