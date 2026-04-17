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

print_banner() {
  cat <<EOF
tmux session: ${SESSION_NAME}
LAN URL: ${LAN_URL}
Authorization header: ${EXPECTED_AUTH_HEADER}
EOF
}

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required to manage the ScholarTrace SSE server." >&2
    exit 1
  }

  load_repo_env
  apply_runtime_defaults
  print_banner

  if tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session ${SESSION_NAME}."
  else
    echo "tmux session ${SESSION_NAME} is not running."
  fi
}

main "$@"
