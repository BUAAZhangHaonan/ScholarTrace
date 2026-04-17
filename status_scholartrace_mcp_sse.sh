#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCHOLARTRACE_ENV_FILE:-${ROOT_DIR}/.env}"
readonly LAN_IP="${SCHOLARTRACE_LAN_IP:-172.17.194.210}"

load_repo_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    echo "Loaded repo .env from ${ENV_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi

  export SCHOLARTRACE_MCP_TRANSPORT="${SCHOLARTRACE_MCP_TRANSPORT:-sse}"
  export SCHOLARTRACE_MCP_HOST="${SCHOLARTRACE_MCP_HOST:-0.0.0.0}"
  export SCHOLARTRACE_MCP_PORT="${SCHOLARTRACE_MCP_PORT:-8001}"
  export SCHOLARTRACE_REMOTE_ACCESS_ENABLED="${SCHOLARTRACE_REMOTE_ACCESS_ENABLED:-true}"
  export SCHOLARTRACE_ACCESS_TOKEN="${SCHOLARTRACE_ACCESS_TOKEN:-g203-mcp}"
  export SCHOLARTRACE_MCP_SSE_SESSION_NAME="${SCHOLARTRACE_MCP_SSE_SESSION_NAME:-scholartrace_mcp_sse}"
}

apply_runtime_defaults() {
  SESSION_NAME="${SCHOLARTRACE_MCP_SSE_SESSION_NAME}"
  LAN_URL="http://${LAN_IP}:${SCHOLARTRACE_MCP_PORT}/sse"
  EXPECTED_AUTH_HEADER="Authorization: Bearer ${SCHOLARTRACE_ACCESS_TOKEN}"
}

print_common() {
  local status_text="$1"
  local chatbox_config
  chatbox_config="$(cat <<JSON
{"mcpServers":{"scholartrace":{"url":"${LAN_URL}","headers":{"Authorization":"Bearer ${SCHOLARTRACE_ACCESS_TOKEN}"}}}}
JSON
)"
  local chatbox_encoded
  chatbox_encoded="$(printf '%s' "${chatbox_config}" | base64 | tr -d '\n\r')"

  cat <<EOF
tmux session: ${SESSION_NAME}
LAN URL: ${LAN_URL}
Authorization header: ${EXPECTED_AUTH_HEADER}
ChatBox clipboard JSON:
${chatbox_config}
ChatBox one-click link:
chatbox://mcp/install?server=${chatbox_encoded}
status: ${status_text}
Verification commands:
  ./status_scholartrace_mcp_sse.sh
  tmux has-session -t ${SESSION_NAME}
  tmux attach -t ${SESSION_NAME}
  tmux capture-pane -pt ${SESSION_NAME}
  ss -ltnp | grep ':${SCHOLARTRACE_MCP_PORT}'
  curl -H '${EXPECTED_AUTH_HEADER}' '${LAN_URL}'
EOF
}

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required to inspect the ScholarTrace SSE server." >&2
    exit 1
  }

  load_repo_env
  apply_runtime_defaults

  if tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1; then
    print_common "running"
  else
    print_common "stopped"
  fi
}

main "$@"
