#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCHOLARTRACE_ENV_FILE:-${ROOT_DIR}/.env}"
readonly LAN_IP="${SCHOLARTRACE_LAN_IP:-172.17.194.210}"
readonly DEBUG_SESSION="scholartrace_mcp_debug"
readonly DEBUG_PORT=8002

normalize_legacy_env() {
  local target_name="$1"
  local legacy_name="$2"
  local target_value="${!target_name:-}"
  local legacy_value="${!legacy_name:-}"

  if [[ -z "${target_value}" && -n "${legacy_value}" ]]; then
    export "${target_name}=${legacy_value}"
  fi
}

load_repo_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    echo "Loaded repo .env from ${ENV_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    echo "No repo .env found at ${ENV_FILE}"
  fi

  normalize_legacy_env SCHOLARTRACE_BIGMODEL_API_KEY BIGMODEL_API_KEY
  normalize_legacy_env SCHOLARTRACE_BIGMODEL_BASE_URL BIGMODEL_BASE_URL
  normalize_legacy_env SCHOLARTRACE_BIGMODEL_MODEL BIGMODEL_MODEL
  export SCHOLARTRACE_BIGMODEL_BASE_URL="${SCHOLARTRACE_BIGMODEL_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4/chat/completions}"
  export SCHOLARTRACE_BIGMODEL_MODEL="${SCHOLARTRACE_BIGMODEL_MODEL:-glm-5-turbo}"

  # Override for debug instance
  export SCHOLARTRACE_MCP_TRANSPORT="sse"
  export SCHOLARTRACE_MCP_HOST="0.0.0.0"
  export SCHOLARTRACE_MCP_PORT="${DEBUG_PORT}"
  export SCHOLARTRACE_REMOTE_ACCESS_ENABLED="true"
  export SCHOLARTRACE_ACCESS_TOKEN="${SCHOLARTRACE_ACCESS_TOKEN:-g203-mcp}"
  export SCHOLARTRACE_LOG_LEVEL="DEBUG"
}

seed_tmux_environment() {
  local name
  tmux set-environment -g PATH "${PATH}"
  while IFS= read -r name; do
    [[ "${name}" == SCHOLARTRACE_* ]] || continue
    tmux set-environment -g "${name}" "${!name}"
  done < <(compgen -v)
}

print_banner() {
  local lan_url="http://${LAN_IP}:${DEBUG_PORT}/sse"
  local auth_header="Authorization: Bearer ${SCHOLARTRACE_ACCESS_TOKEN}"
  cat <<EOF
=== ScholarTrace DEBUG Server ===
tmux session: ${DEBUG_SESSION}
LAN URL: ${lan_url}
Authorization header: ${auth_header}
Log level: DEBUG
NOTE: This is a debug instance. Production remains on port 8001.
Verification:
  ./status_scholartrace_mcp_debug.sh
  tmux attach -t ${DEBUG_SESSION}
  ss -ltnp | grep ':${DEBUG_PORT}'
EOF
}

require_bigmodel_key() {
  if [[ -z "${SCHOLARTRACE_BIGMODEL_API_KEY:-}" ]]; then
    echo "Error: SCHOLARTRACE_BIGMODEL_API_KEY is required after loading ${ENV_FILE}." >&2
    exit 1
  fi
}

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required." >&2
    exit 1
  }

  load_repo_env
  require_bigmodel_key
  print_banner

  if tmux has-session -t "${DEBUG_SESSION}" >/dev/null 2>&1; then
    echo "tmux session ${DEBUG_SESSION} is already running."
    exit 0
  fi

  local mcp_bin
  mcp_bin="$(command -v scholartrace-mcp)" || {
    echo "Error: scholartrace-mcp is not on PATH." >&2
    exit 1
  }

  seed_tmux_environment

  local launch_command
  launch_command="$(printf 'cd %q && exec %q' "${ROOT_DIR}" "${mcp_bin}")"
  tmux new-session -d -s "${DEBUG_SESSION}" bash -lc "${launch_command}"
  echo "Started debug tmux session ${DEBUG_SESSION} on port ${DEBUG_PORT}."
}

main "$@"
