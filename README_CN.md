# ScholarTrace

[English](README.md) | **中文**

> 多源学术论文检索 + LLM 多模型重排，通过 MCP 协议服务于 ChatBox 等 AI 客户端。

## 它做什么

ScholarTrace 接收一份主题文档，从 6+ 学术源检索论文，用多模型 LLM 池排序，最终暴露两个 MCP 工具。

**两个 MCP 工具**：`query`（检索排序）和 `read`（分层论文阅读）。

**Pipeline**：主题解析 → 多源检索 → 去重 → 综合评分 → ModelPool LLM 重排 → 最终论文。

**模型池**：glm-5-turbo(5) → glm-4.7(20) → glm-4.6(20) → deepseek(10) → qwen(10)，自动故障转移和冷却。

**数据源**：OpenAlex、Semantic Scholar、arXiv、DBLP、CrossRef、OpenReview、DeepXiv（可选）。

## 快速开始

### 1. 配置 `.env`

```bash
SCHOLARTRACE_BIGMODEL_API_KEY=<your-bigmodel-key>
SCHOLARTRACE_ACCESS_TOKEN=g203-mcp

# LAN SSE 默认值（脚本自动设置）
SCHOLARTRACE_MCP_TRANSPORT=sse
SCHOLARTRACE_MCP_HOST=0.0.0.0
SCHOLARTRACE_MCP_PORT=8001
SCHOLARTRACE_REMOTE_ACCESS_ENABLED=true

# 可选：DeepXiv
# SCHOLARTRACE_DEEPXIV_TOKENS=token-a,token-b
```

### 2. 启动服务

```bash
./run_scholartrace_mcp_sse.sh        # 启动
./status_scholartrace_mcp_sse.sh     # 查看状态
./stop_scholartrace_mcp_sse.sh       # 停止
```

### 3. ChatBox 连接

把 `<server-ip>` 替换成你的局域网 IP：

```json
{
  "mcpServers": {
    "scholartrace": {
      "url": "http://<server-ip>:8001/sse",
      "headers": {
        "Authorization": "Bearer g203-mcp"
      }
    }
  }
}
```

## MCP 工具

### `query` — 检索与排序

```json
{
  "theme_document": "你的研究主题描述",
  "final_limit": 25,
  "agent_candidate_limit": 200,
  "include_rationale": true
}
```

返回带分数、理由和全文状态的排序论文。

### `read` — 分层论文阅读

```json
{
  "paper_id": "theme-id:paper-id",
  "depth": "fulltext",
  "allow_acquire": true
}
```

深度：`summary` → `sections` → `fulltext` → `direct_evidence`。每个深度诚实返回可用状态。

## 架构

```
主题文档
    │
    ▼
parse_theme（提取查询 + 压缩摘要）
    │
    ▼
fan-out 检索（6+ 数据源，每源 45s 超时，429/5xx 自动重试）
    │
    ▼
去重 → 综合评分（相关性 + 时效性 + 影响力 + 期刊 + ...）
    │
    ▼
ModelPool LLM 重排（多模型故障转移，总超时 180s）
    │
    ▼
最终论文（top 25）
```

流程图：[`docs/architecture/pipeline_flow.md`](docs/architecture/pipeline_flow.md)

设计文档：[`docs/plans/`](docs/plans/)

## 可靠性

- **连接器重试**：6 个数据源全部支持 429/5xx/超时指数退避重试
- **每源超时**：每个数据源 45s 上限，不会互相阻塞
- **查询重试**：所有源返回空时自动重试
- **检索总超时**：300s 整体检索阶段上限
- **模型池故障转移**：LLM 出错自动冷却 + 切换下一模型
- **确定性兜底**：所有 LLM 模型都失败时，综合评分仍然返回论文

## 配置

| 变量 | 默认值 | 用途 |
|---|---|---|
| `SCHOLARTRACE_MCP_TRANSPORT` | `stdio` | 局域网用 `sse` |
| `SCHOLARTRACE_MCP_HOST` | `127.0.0.1` | 局域网用 `0.0.0.0` |
| `SCHOLARTRACE_MCP_PORT` | `8001` | SSE 端口 |
| `SCHOLARTRACE_REMOTE_ACCESS_ENABLED` | `false` | 局域网必须 `true` |
| `SCHOLARTRACE_ACCESS_TOKEN` | | Bearer token |
| `SCHOLARTRACE_BIGMODEL_API_KEY` | | 主 LLM API key |
| `SCHOLARTRACE_AGENT_CANDIDATE_LIMIT` | `200` | 送入 LLM 重排的论文数 |
| `SCHOLARTRACE_FINAL_LIMIT` | `25` | 最终返回论文数 |
| `SCHOLARTRACE_RETRIEVAL_CONNECTOR_TIMEOUT_SECONDS` | `45` | 每源超时 |
| `SCHOLARTRACE_RETRIEVAL_TOTAL_TIMEOUT_SECONDS` | `300` | 检索总超时 |
| `SCHOLARTRACE_AGENT_TOTAL_TIMEOUT_SECONDS` | `180` | LLM 重排超时 |
| `SCHOLARTRACE_DEEPXIV_TOKENS` | | 可选 DeepXiv tokens |

## 部署

### tmux（推荐）

根目录脚本自动处理 `.env` 加载、传输默认值和错误检查。

### systemd

参考 `scripts/scholartrace-mcp.service`。密钥放在 `/etc/scholartrace/scholartrace.env`。

### stdio（仅调试）

```bash
SCHOLARTRACE_MCP_TRANSPORT=stdio scholartrace-mcp
```

## 许可证

MIT
