#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCHOLARTRACE_ENV_FILE:-${ROOT_DIR}/.env}"
readonly LAN_IP="${SCHOLARTRACE_LAN_IP:-127.0.0.1}"

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

  # Normalize legacy BigModel variable names from existing .env files.
  normalize_legacy_env SCHOLARTRACE_BIGMODEL_API_KEY BIGMODEL_API_KEY
  normalize_legacy_env SCHOLARTRACE_BIGMODEL_BASE_URL BIGMODEL_BASE_URL
  normalize_legacy_env SCHOLARTRACE_BIGMODEL_MODEL BIGMODEL_MODEL
  export SCHOLARTRACE_BIGMODEL_BASE_URL="${SCHOLARTRACE_BIGMODEL_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4/chat/completions}"
  export SCHOLARTRACE_BIGMODEL_MODEL="${SCHOLARTRACE_BIGMODEL_MODEL:-glm-5-turbo}"

  export SCHOLARTRACE_MCP_TRANSPORT="${SCHOLARTRACE_MCP_TRANSPORT:-sse}"
  export SCHOLARTRACE_MCP_HOST="${SCHOLARTRACE_MCP_HOST:-0.0.0.0}"
  export SCHOLARTRACE_MCP_PORT="${SCHOLARTRACE_MCP_PORT:-8001}"
  export SCHOLARTRACE_REMOTE_ACCESS_ENABLED="${SCHOLARTRACE_REMOTE_ACCESS_ENABLED:-true}"
  export SCHOLARTRACE_ACCESS_TOKEN="${SCHOLARTRACE_ACCESS_TOKEN:-}"
  export SCHOLARTRACE_MCP_SSE_SESSION_NAME="${SCHOLARTRACE_MCP_SSE_SESSION_NAME:-scholartrace_mcp_sse}"
}

apply_runtime_defaults() {
  SESSION_NAME="${SCHOLARTRACE_MCP_SSE_SESSION_NAME}"
  LAN_URL="http://${LAN_IP}:${SCHOLARTRACE_MCP_PORT}/sse"
  EXPECTED_AUTH_HEADER="Authorization: Bearer ${SCHOLARTRACE_ACCESS_TOKEN}"
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
Verification commands:
  ./status_scholartrace_mcp_sse.sh
  tmux has-session -t ${SESSION_NAME}
  tmux attach -t ${SESSION_NAME}
  tmux capture-pane -pt ${SESSION_NAME}
  ss -ltnp | grep ':${SCHOLARTRACE_MCP_PORT}'
  curl -H '${EXPECTED_AUTH_HEADER}' '${LAN_URL}'
EOF
}

check_deepxiv() {
  local deepxiv_tokens="${SCHOLARTRACE_DEEPXIV_TOKENS:-}"
  local deepxiv_auto_register="${SCHOLARTRACE_DEEPXIV_AUTO_REGISTER:-false}"
  local deepxiv_secret="${SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET:-}"
  local has_tokens="false"
  local has_auto_register="false"

  if [[ -n "${deepxiv_tokens//[[:space:]]/}" ]]; then
    has_tokens="true"
  fi
  if [[ "${deepxiv_auto_register,,}" == "true" ]]; then
    has_auto_register="true"
  fi

  if [[ "${has_auto_register}" == "true" && -z "${deepxiv_secret}" ]]; then
    echo "Error: SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET is required when SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true." >&2
    exit 1
  fi

  if [[ "${has_tokens}" == "true" || ( "${has_auto_register}" == "true" && -n "${deepxiv_secret}" ) ]]; then
    echo "DeepXiv retrieval and evidence markdown fallback are enabled."
  else
    echo "DeepXiv retrieval/evidence markdown fallback will be skipped/unavailable because DeepXiv is not configured."
  fi
}

require_bigmodel_key() {
  if [[ -z "${SCHOLARTRACE_BIGMODEL_API_KEY:-}" ]]; then
    echo "Error: SCHOLARTRACE_BIGMODEL_API_KEY is required after loading ${ENV_FILE}." >&2
    exit 1
  fi
}

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required to keep the ScholarTrace SSE server alive after disconnect." >&2
    exit 1
  }

  load_repo_env
  apply_runtime_defaults
  require_bigmodel_key
  check_deepxiv

  print_banner

  if tmux has-session -t "${SESSION_NAME}" >/dev/null 2>&1; then
    echo "tmux session ${SESSION_NAME} is already running."
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
  tmux new-session -d -s "${SESSION_NAME}" bash -lc "${launch_command}"
  echo "Started tmux session ${SESSION_NAME}."
}

main "$@"
