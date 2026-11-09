#!/usr/bin/env bash
# Wire the per-container state that cannot be baked into an image, then exec the command.
#
# Ported from k8s-ai-platform. Everything is best-effort: the read-only targets (`review`, `test`)
# mount the workspace :ro and cannot write .git/config, and failing to arm a hook must never stop
# the command.
set -uo pipefail

repo=/workspace/quant-playground

# Pi may rewrite its credential store during refresh, so hack/agent.sh mounts the host copy at a
# neutral read-only path and it is copied into the writable home here.
if [[ -r /run/agent-creds/pi-auth.json ]]; then
  mkdir -p /home/agent/.pi/agent
  cp /run/agent-creds/pi-auth.json /home/agent/.pi/agent/auth.json
  chmod 0600 /home/agent/.pi/agent/auth.json 2>/dev/null || true
fi

if [[ -r /run/agent-host/opencode-model.json ]]; then
  mkdir -p /home/agent/.local/state/opencode
  cp /run/agent-host/opencode-model.json /home/agent/.local/state/opencode/model.json
fi

# The container shares the host network; use the OpenCode port hack/agent.sh allocated.
opencode_port=${AGENT_OPENCODE_PORT:-49376}
if [[ $opencode_port =~ ^[1-9][0-9]*$ ]] && ((opencode_port <= 65535)); then
  mkdir -p /home/agent/.config/opencode
  printf '{\n  "port": %s\n}\n' "$opencode_port" >/home/agent/.config/opencode/service.json
fi

if [[ -r /run/agent-creds/opencode-auth.json ]]; then
  if ! opencode auth import /run/agent-creds/opencode-auth.json >/dev/null 2>&1; then
    printf 'agent-entrypoint: could not import OpenCode credentials\n' >&2
  fi
fi

if [[ -d $repo/.git ]]; then
  if ! git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$repo"; then
    git config --global --add safe.directory "$repo" 2>/dev/null || true
  fi

  # Write a repository config key only when it does not already carry the value we want: the test
  # is on the outcome, not on write permission, because under :ro it is usually already correct.
  ensure_git_config() {
    local key=$1 want=$2
    [[ $(git -C "$repo" config --get "$key" 2>/dev/null) == "$want" ]] && return 0
    git -C "$repo" config "$key" "$want" 2>/dev/null
  }

  ensure_git_config user.name "${AGENT_GIT_USER_NAME:-Quant Playground Agent}" || true
  ensure_git_config user.email "${AGENT_GIT_USER_EMAIL:-agent@quant-playground.local}" || true

  if [[ -d $repo/hack/git-hooks ]] && ! ensure_git_config core.hooksPath hack/git-hooks; then
    printf 'agent-entrypoint: core.hooksPath is not hack/git-hooks and .git/config could not be\n' >&2
    printf '  written; the pre-commit hook is disarmed in this container.\n' >&2
  fi
fi

exec "$@"
