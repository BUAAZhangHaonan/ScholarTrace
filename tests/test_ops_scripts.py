from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "run_scholartrace_mcp_sse.sh"
STOP_SCRIPT = ROOT / "stop_scholartrace_mcp_sse.sh"
STATUS_SCRIPT = ROOT / "status_scholartrace_mcp_sse.sh"
SESSION_NAME = "scholartrace_mcp_sse"
LAN_URL = "http://172.17.194.210:8001/sse"
AUTH_HEADER = "Authorization: Bearer g203-mcp"


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    _make_executable(path)


def _prepare_fake_workspace(
    tmp_path: Path,
    *,
    env_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    tmux_active: bool = False,
) -> tuple[Path, dict[str, str], Path, Path]:
    fake_root = tmp_path / "workspace"
    fake_root.mkdir()

    for source in (RUN_SCRIPT, STOP_SCRIPT, STATUS_SCRIPT):
        assert source.exists(), f"expected script to exist: {source.name}"
        target = fake_root / source.name
        shutil.copy2(source, target)
        _make_executable(target)

    if env_text is not None:
        (fake_root / ".env").write_text(env_text, encoding="utf-8")

    bin_dir = fake_root / "bin"
    bin_dir.mkdir()

    tmux_log = fake_root / "tmux.log"
    tmux_state = fake_root / "tmux.state"
    if tmux_active:
        tmux_state.write_text(SESSION_NAME, encoding="utf-8")

    _write_file(
        bin_dir / "tmux",
        f"""#!/usr/bin/env bash
set -euo pipefail
log_file="${{TMUX_LOG_FILE:?}}"
state_file="${{TMUX_STATE_FILE:?}}"
printf '%s\\n' "tmux $*" >> "$log_file"
case "${{1:-}}" in
  has-session)
    if [[ -f "$state_file" ]]; then
      exit 0
    fi
    exit 1
    ;;
  new-session)
    printf '%s\\n' "${{2:-}}" >> "$log_file"
    printf '%s\\n' "{SESSION_NAME}" > "$state_file"
    exit 0
    ;;
  kill-session)
    rm -f "$state_file"
    exit 0
    ;;
  display-message|list-sessions|attach)
    if [[ -f "$state_file" ]]; then
      exit 0
    fi
    exit 1
    ;;
  *)
    exit 0
    ;;
esac
""",
    )
    _write_file(
        bin_dir / "scholartrace-mcp",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["TMUX_LOG_FILE"] = str(tmux_log)
    env["TMUX_STATE_FILE"] = str(tmux_state)
    env.pop("SCHOLARTRACE_BIGMODEL_API_KEY", None)
    env.pop("BIGMODEL_API_KEY", None)
    env.pop("SCHOLARTRACE_DEEPXIV_TOKENS", None)
    env.pop("SCHOLARTRACE_DEEPXIV_AUTO_REGISTER", None)
    env.pop("SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET", None)
    if extra_env:
        env.update(extra_env)

    return fake_root, env, tmux_log, tmux_state


def _run_script(
    script: Path,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_loads_repo_env_and_reports_it(tmp_path: Path) -> None:
    fake_root, env, tmux_log, _ = _prepare_fake_workspace(
        tmp_path,
        env_text=(
            "SCHOLARTRACE_BIGMODEL_API_KEY=from-env\n"
            "SCHOLARTRACE_MCP_PORT=8123\n"
            "SCHOLARTRACE_ACCESS_TOKEN=env-token\n"
            "SCHOLARTRACE_MCP_SSE_SESSION_NAME=env-session\n"
        ),
    )

    result = _run_script(fake_root / RUN_SCRIPT.name, fake_root, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loaded repo .env" in result.stdout
    assert "env-session" in result.stdout
    assert "http://172.17.194.210:8123/sse" in result.stdout
    assert "Authorization: Bearer env-token" in result.stdout
    assert "tmux has-session -t env-session" in result.stdout
    assert "tmux attach -t env-session" in result.stdout
    assert tmux_log.read_text(encoding="utf-8")


def test_run_normalizes_legacy_bigmodel_env_names(tmp_path: Path) -> None:
    fake_root, env, _, tmux_state = _prepare_fake_workspace(
        tmp_path,
        env_text=(
            "BIGMODEL_API_KEY=legacy-key\n"
            "BIGMODEL_BASE_URL=https://example.invalid/glm\n"
        ),
    )

    result = _run_script(fake_root / RUN_SCRIPT.name, fake_root, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert tmux_state.read_text(encoding="utf-8").strip() == SESSION_NAME


def test_run_fails_cleanly_when_bigmodel_key_is_missing(tmp_path: Path) -> None:
    fake_root, env, _, _ = _prepare_fake_workspace(tmp_path)

    result = _run_script(fake_root / RUN_SCRIPT.name, fake_root, env)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "SCHOLARTRACE_BIGMODEL_API_KEY is required" in combined


def test_run_allows_missing_deepxiv_config(tmp_path: Path) -> None:
    fake_root, env, _, _ = _prepare_fake_workspace(
        tmp_path,
        env_text="SCHOLARTRACE_BIGMODEL_API_KEY=from-env\n",
    )

    result = _run_script(fake_root / RUN_SCRIPT.name, fake_root, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DeepXiv retrieval/evidence markdown fallback will be skipped/unavailable" in result.stdout


def test_run_fails_when_auto_register_secret_is_missing(tmp_path: Path) -> None:
    fake_root, env, _, _ = _prepare_fake_workspace(
        tmp_path,
        env_text="SCHOLARTRACE_BIGMODEL_API_KEY=from-env\n",
        extra_env={"SCHOLARTRACE_DEEPXIV_AUTO_REGISTER": "true"},
    )

    result = _run_script(fake_root / RUN_SCRIPT.name, fake_root, env)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET is required" in combined


def test_status_reports_session_url_and_header(tmp_path: Path) -> None:
    fake_root, env, _, _ = _prepare_fake_workspace(
        tmp_path,
        env_text=(
            "SCHOLARTRACE_BIGMODEL_API_KEY=from-env\n"
            "SCHOLARTRACE_MCP_PORT=8123\n"
            "SCHOLARTRACE_ACCESS_TOKEN=env-token\n"
            "SCHOLARTRACE_MCP_SSE_SESSION_NAME=env-session\n"
        ),
        tmux_active=True,
    )

    result = _run_script(fake_root / STATUS_SCRIPT.name, fake_root, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "env-session" in result.stdout
    assert "http://172.17.194.210:8123/sse" in result.stdout
    assert "Authorization: Bearer env-token" in result.stdout
