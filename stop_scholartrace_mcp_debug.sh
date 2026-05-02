#!/usr/bin/env bash
set -euo pipefail

readonly DEBUG_SESSION="scholartrace_mcp_debug"

main() {
  command -v tmux >/dev/null 2>&1 || {
    echo "Error: tmux is required." >&2
    exit 1
  }

  if tmux has-session -t "${DEBUG_SESSION}" >/dev/null 2>&1; then
    tmux kill-session -t "${DEBUG_SESSION}"
    echo "Stopped debug tmux session ${DEBUG_SESSION}."
  else
    echo "Debug tmux session ${DEBUG_SESSION} is not running."
  fi
}

main "$@"
