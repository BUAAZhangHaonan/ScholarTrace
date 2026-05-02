#!/usr/bin/env bash
set -euo pipefail

readonly DEBUG_SESSION="scholartrace_mcp_debug"
readonly DEBUG_PORT=8002
readonly LAN_IP="${SCHOLARTRACE_LAN_IP:-172.17.194.210}"

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required." >&2
    exit 1
  }

  local lan_url="http://${LAN_IP}:${DEBUG_PORT}/sse"

  echo "=== ScholarTrace DEBUG Server Status ==="
  echo "tmux session: ${DEBUG_SESSION}"
  echo "LAN URL: ${lan_url}"
  echo "Port ${DEBUG_PORT}:"

  if ss -ltnp | grep -q ":${DEBUG_PORT}"; then
    echo "  LISTENING"
    ss -ltnp | grep ":${DEBUG_PORT}"
  else
    echo "  NOT LISTENING"
  fi

  if tmux has-session -t "${DEBUG_SESSION}" >/dev/null 2>&1; then
    echo "tmux session: RUNNING"
    echo ""
    echo "--- Last 20 lines of output ---"
    tmux capture-pane -pt "${DEBUG_SESSION}" -S -20 2>/dev/null || echo "(no output captured)"
  else
    echo "tmux session: STOPPED"
  fi
}

main "$@"
