#!/usr/bin/env bash
# AI-agent development container: build the image, and run this repository's work inside it.
#
# Ported from k8s-ai-platform's hack/agent.sh. The Go-specific parts (module proxy, LOCALBIN) are
# gone; the Python environment is baked into the image at /opt/venv, and uv's download cache sits on
# a named volume so `uv add` inside the container is cheap after the first time.
#
# Usage: hack/agent.sh [-v] <command> [--creds [LIST]] [--mount SPEC ...] [-- COMMAND ...]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

prog=agent.sh
# shellcheck source=lib/registry.sh
source "$repo_root/hack/lib/registry.sh"

image_repo=${AGENT_IMAGE_REPOSITORY:-quant-playground-agent}
image=${AGENT_IMAGE:-$image_repo:latest}
remote_image_repo=${AGENT_IMAGE_REMOTE_REPOSITORY:-chihayaanon0908/quant-playground-agent}
name=${AGENT_NAME:-quant-agent}
workspace=/workspace/quant-playground

# OpenCode's managed service would otherwise take the host's port under host networking. A range
# distinct from k8s-ai-platform's (49376-49400), so agents from both repositories can coexist.
opencode_port_range=${AGENT_OPENCODE_PORT_RANGE:-49401-49425}

# Git identity for commits made inside the container, in git's own precedence order.
git_user_name=${AGENT_GIT_USER_NAME:-$(git config --local user.name 2>/dev/null || git config --global user.name 2>/dev/null || printf 'Quant Playground Agent')}
git_user_email=${AGENT_GIT_USER_EMAIL:-$(git config --local user.email 2>/dev/null || git config --global user.email 2>/dev/null || printf 'agent@quant-playground.local')}

# Named volumes for everything expensive to rebuild and pointless to keep in the image.
uv_volume=(-v quant-uv-cache:/home/agent/.cache/uv)
# LazyVim's ~500MB plugin tree; `docker volume rm quant-nvim` forces a clean reinstall.
nvim_volume=(-v quant-nvim:/home/agent/.local/share/nvim)
# `claude /login` once inside the container and the credential survives --rm here.
claude_volume=(-v quant-claude:/home/agent/.claude)
opencode_volume=(-v quant-opencode:/home/agent/.local/share/opencode)

# Shell configuration from the repository rather than the baked copy, so edits take effect in the
# next shell. Read-only: nothing in the container writes its own configuration.
home_mounts=(
  -v "$repo_root/hack/agent-home/.zshenv:/home/agent/.zshenv:ro"
  -v "$repo_root/hack/agent-home/.zprofile:/home/agent/.zprofile:ro"
  -v "$repo_root/hack/agent-home/.zshrc:/home/agent/.zshrc:ro"
  -v "$repo_root/hack/agent-home/.zsh_aliases:/home/agent/.zsh_aliases:ro"
  -v "$repo_root/hack/agent-home/.config/starship.toml:/home/agent/.config/starship.toml:ro"
  -v "$repo_root/hack/agent-home/.config/opencode/cli.json:/home/agent/.config/opencode/cli.json:ro"
)

opencode_config_dir=${XDG_CONFIG_HOME:-$HOME/.config}/opencode
opencode_state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/opencode
opencode_config_mounts=()
opencode_state_mounts=()
for config_name in opencode.json opencode.jsonc; do
  if [[ -f $opencode_config_dir/$config_name ]]; then
    opencode_config_mounts+=(-v "$opencode_config_dir/$config_name:/home/agent/.config/opencode/$config_name:ro")
  fi
done
if [[ -f $opencode_state_dir/model.json ]]; then
  opencode_state_mounts=(--mount "type=bind,src=$opencode_state_dir/model.json,dst=/run/agent-host/opencode-model.json,readonly")
fi

# Credential files the agent CLIs read, as name:host:container. Opt-in through --creds only.
# Claude Code is absent on purpose: log in once inside the container and the quant-claude volume
# keeps it. OpenCode's accounts live in SQLite and are exported, not mounted.
credentials=(
  "pi:${HOME}/.pi/agent/auth.json:/run/agent-creds/pi-auth.json"
  "codex:${CODEX_HOME:-$HOME/.codex}/auth.json:/home/agent/.codex/auth.json"
)

sandbox=(--cap-drop ALL --security-opt no-new-privileges)

verbose=0
pull_policy=${AGENT_PULL:-missing}
build_cache_dir=${AGENT_BUILD_CACHE-${XDG_CACHE_HOME:-$HOME/.cache}/quant-playground/buildkit}
force_latest=0
session_mounts=()
extra_mounts=()
cred_mounts=()
opencode_cred_file=
creds=0
creds_select=
command_override=()
# Ports published under bridge networking; with host networking (the default) they are no-ops.
publish_ports=()

port_in_use() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ss -Hltn 2>/dev/null | awk -v p="$port" '$4 ~ ":" p "$" { found = 1 } END { exit found ? 0 : 1 }'
  else
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
  fi
}

choose_opencode_port() {
  local range=$1 start end port
  if [[ ! $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
    die "AGENT_OPENCODE_PORT_RANGE must be START-END, got: $range"
  fi
  start=$((10#${BASH_REMATCH[1]}))
  end=$((10#${BASH_REMATCH[2]}))
  if ((start < 1 || end > 65535 || start > end)); then
    die "AGENT_OPENCODE_PORT_RANGE is outside 1-65535 or reversed: $range"
  fi
  for ((port = start; port <= end; port++)); do
    if ! port_in_use "$port"; then
      printf '%s\n' "$port"
      return 0
    fi
  done
  die "no free port in AGENT_OPENCODE_PORT_RANGE=$range"
}

parse_options() {
  while [[ $# -gt 0 ]]; do
    case $1 in
    -v | --verbose)
      verbose=1
      shift
      ;;
    --force-latest)
      force_latest=1
      shift
      ;;
    --mount)
      [[ $# -ge 2 ]] || die "--mount needs a specification"
      extra_mounts+=(--mount "$2")
      shift 2
      ;;
    --mount=*)
      extra_mounts+=(--mount "${1#--mount=}")
      shift
      ;;
    --creds)
      creds=1
      if [[ -n ${2:-} && ${2:-} != -* ]]; then
        creds_select=$2
        shift 2
      else
        shift
      fi
      ;;
    --creds=*)
      creds=1
      creds_select=${1#--creds=}
      shift
      ;;
    --)
      shift
      command_override=("$@")
      return
      ;;
    -*) die "unknown option: $1 (see hack/agent.sh help)" ;;
    *) die "unexpected argument: $1 (pass a command after --)" ;;
    esac
  done
}

resolve_sessions() {
  local pi_sessions codex_sessions
  pi_sessions=${PI_SESSIONS_DIR:-$HOME/.pi/agent/sessions}
  codex_sessions=${CODEX_SESSIONS_DIR:-${CODEX_HOME:-$HOME/.codex}/sessions}
  mkdir -p "$pi_sessions" "$codex_sessions"
  session_mounts=(
    --mount "type=bind,src=$pi_sessions,dst=/home/agent/.pi/agent/sessions"
    --mount "type=bind,src=$codex_sessions,dst=/home/agent/.codex/sessions"
  )
}

export_opencode_credentials() {
  local json
  command -v opencode >/dev/null 2>&1 || return 1
  json=$(opencode auth export --standalone 2>/dev/null) || return 1
  [[ -n $json && $json != "[]" ]] || return 1
  opencode_cred_file=$(mktemp "${TMPDIR:-/tmp}/opencode-auth.XXXXXX") || return 1
  chmod 0600 "$opencode_cred_file"
  printf '%s\n' "$json" >"$opencode_cred_file"
  cred_mounts+=(--mount "type=bind,src=$opencode_cred_file,dst=/run/agent-creds/opencode-auth.json,readonly")
  printf 'agent.sh: mounting opencode credential, read-only: host auth export\n' >&2
}

# A named credential that is absent is an error; a bare --creds takes whatever exists and says so.
resolve_credentials() {
  ((creds)) || return 0

  local wanted=() entry name host container known_names=(pi codex opencode)
  [[ -n $creds_select ]] && IFS=, read -ra wanted <<<"$creds_select"

  for name in ${wanted[@]+"${wanted[@]}"}; do
    local known=0
    for entry in "${known_names[@]}"; do
      [[ $entry == "$name" ]] && known=1
    done
    ((known)) || die "unknown credential: $name (known: ${known_names[*]})"
  done

  for entry in "${credentials[@]}"; do
    name=${entry%%:*}
    host=${entry#*:}
    host=${host%:*}
    container=${entry##*:}
    if ((${#wanted[@]})) && ! printf '%s\n' "${wanted[@]}" | grep -qxF "$name"; then
      continue
    fi
    if [[ ! -f $host ]]; then
      ((${#wanted[@]})) && die "no $name credential at $host"
      continue
    fi
    cred_mounts+=(--mount "type=bind,src=$host,dst=$container,readonly")
    printf 'agent.sh: mounting %s credential, read-only: %s\n' "$name" "$host" >&2
  done

  local want_opencode=0
  if ((${#wanted[@]})); then
    for name in "${wanted[@]}"; do
      [[ $name == opencode ]] && want_opencode=1
    done
  else
    want_opencode=1
  fi
  if ((want_opencode)); then
    if ! export_opencode_credentials; then
      ((${#wanted[@]})) && die "no saved OpenCode credentials on this host"
    fi
  fi
}

build_image() {
  local git_sha commit_image
  git_sha=$(current_git_sha)
  commit_image="$image_repo:$git_sha"

  local python_image node_image uv_image
  python_image=${AGENT_PYTHON_IMAGE:-$(dockerfile_arg_default agent.Dockerfile PYTHON_IMAGE)}
  node_image=${AGENT_NODE_IMAGE:-$(dockerfile_arg_default agent.Dockerfile NODE_IMAGE)}
  uv_image=${AGENT_UV_IMAGE:-$(dockerfile_arg_default agent.Dockerfile UV_IMAGE)}

  # Resolve moving tags to immutable digests before the build, from the local image when there is
  # one, so the build is pinned without a pull.
  pull_base_image "$python_image"
  pull_base_image "$node_image"
  pull_base_image "$uv_image"

  local build_flags=(--network "${AGENT_BUILD_NETWORK:-host}")
  append_build_cache_flags build_flags "$build_cache_dir"
  local build_args=(
    --build-arg PYTHON_IMAGE="$(base_image_digest "$python_image")"
    --build-arg NODE_IMAGE="$(base_image_digest "$node_image")"
    --build-arg UV_IMAGE="$(base_image_digest "$uv_image")"
    --build-arg USER_UID="$(id -u)"
    --build-arg USER_GID="$(id -g)"
    --build-arg GIT_SHA="$git_sha"
  )
  local mirror_var
  for mirror_var in NPM_REGISTRY APT_MIRROR GITHUB_MIRROR NVIM_RELEASE_BASE PYPI_INDEX; do
    if [[ -n ${!mirror_var:-} ]]; then
      build_args+=(--build-arg "$mirror_var=${!mirror_var}")
    fi
  done
  local proxy_var
  for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    if [[ -n ${!proxy_var:-} ]]; then
      build_args+=(--build-arg "$proxy_var=${!proxy_var}")
    fi
  done

  ((verbose)) || printf 'agent.sh: building %s and %s (-v to follow)\n' "$commit_image" "$image" >&2
  run_step docker build \
    "${build_flags[@]}" \
    -f agent.Dockerfile \
    "${build_args[@]}" \
    -t "$commit_image" \
    -t "$image" .
}

ensure_image() {
  if docker image inspect "$image" >/dev/null 2>&1; then
    return 0
  fi
  die "no agent image found: $image (run ./hack/agent.sh image first)"
}

# Named volumes are created root-owned by the daemon; the agent user has to be able to write them.
prepare_volumes() {
  ensure_image
  run_step docker run --rm --user root --entrypoint sh \
    "${uv_volume[@]}" \
    "${nvim_volume[@]}" \
    "${claude_volume[@]}" \
    "${opencode_volume[@]}" \
    "$image" \
    -c 'mkdir -p /home/agent/.cache/uv /home/agent/.claude /home/agent/.local/share/opencode /home/agent/.local/share/nvim \
      && chown -R agent:agent /home/agent/.cache /home/agent/.claude /home/agent/.local/share/opencode /home/agent/.local/share/nvim'
}

#   $1  rw|ro   how to mount the checkout
#   $2  tty|batch
#   rest        the command, when not overridden by --
agent_run() {
  local access=$1 mode=$2
  shift 2

  local opencode_port
  opencode_port=$(choose_opencode_port "$opencode_port_range")

  local mount_flags=(-v "$repo_root:$workspace")
  [[ $access == ro ]] && mount_flags=(-v "$repo_root:$workspace:ro")

  # Neovim's configuration follows the session's access mode: LazyVim rewrites lazy-lock.json when
  # it installs plugins, and a review session must not write to the tree.
  local nvim_flags=(-v "$repo_root/hack/agent-home/.config/nvim:/home/agent/.config/nvim")
  [[ $access == ro ]] && nvim_flags=(-v "$repo_root/hack/agent-home/.config/nvim:/home/agent/.config/nvim:ro")

  local mode_flags=(--rm)
  if [[ $mode == tty ]]; then
    mode_flags+=(--name "$name" --pids-limit 1024)
    [[ -t 0 ]] && mode_flags+=(-it)
  fi

  local proxy_flags=() proxy_var
  for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    if [[ -n ${!proxy_var:-} ]]; then
      proxy_flags+=(-e "$proxy_var=${!proxy_var}")
    fi
  done

  local cmd=("$@")
  [[ ${#command_override[@]} -gt 0 ]] && cmd=("${command_override[@]}")

  run docker run \
    "${mode_flags[@]}" \
    --network "${AGENT_RUN_NETWORK:-host}" \
    "${sandbox[@]}" \
    -e AGENT_OPENCODE_PORT="$opencode_port" \
    -e AGENT_GIT_USER_NAME="$git_user_name" \
    -e AGENT_GIT_USER_EMAIL="$git_user_email" \
    ${proxy_flags[@]+"${proxy_flags[@]}"} \
    ${publish_ports[@]+"${publish_ports[@]}"} \
    "${mount_flags[@]}" \
    "${nvim_flags[@]}" \
    "${uv_volume[@]}" \
    "${nvim_volume[@]}" \
    "${claude_volume[@]}" \
    "${opencode_volume[@]}" \
    "${session_mounts[@]}" \
    "${home_mounts[@]}" \
    ${opencode_config_mounts[@]+"${opencode_config_mounts[@]}"} \
    ${opencode_state_mounts[@]+"${opencode_state_mounts[@]}"} \
    ${cred_mounts[@]+"${cred_mounts[@]}"} \
    ${extra_mounts[@]+"${extra_mounts[@]}"} \
    -w "$workspace" \
    "$image" \
    "${cmd[@]}"
}

usage() {
  cat <<'EOF'
Usage: hack/agent.sh [-v] <command> [--creds [LIST]] [--mount SPEC ...] [-- COMMAND ...]

Commands:
  image     build the agent container image locally, tagged with the commit sha and latest
  sync      align the local and published images for this commit: pull, build, or push as needed
  prepare   make the named volumes writable using the latest image
  test      run every lab's tests against solutions/ from a read-only checkout
  lint      ruff check, from a read-only checkout
  fmt       ruff format + ruff check --fix on the tree
  jupyter   start JupyterLab on port 8888 (host network: open http://localhost:8888)
  shell     open a development shell; the checkout is writable
  review    open a review shell; the checkout is read-only
  help      this text

Options:
  -v, --verbose  print the docker commands and the setup output that is otherwise captured
  --force-latest move the remote latest tag even when sync cannot prove this checkout is newer
  --creds [LIST] mount agent CLI credentials read-only: pi,codex,opencode (default: whichever
                 exist). Claude Code needs none: `claude /login` once inside, the quant-claude
                 volume keeps it.
  --mount SPEC   passed to docker as --mount, repeatable
  -- COMMAND     run COMMAND instead of the default, e.g.
                   hack/agent.sh shell -- pytest labs/lab03_backtest_statistics

Environment:
  AGENT_IMAGE, AGENT_IMAGE_REPOSITORY, AGENT_IMAGE_REMOTE_REPOSITORY, AGENT_NAME
  AGENT_BUILD_NETWORK, AGENT_RUN_NETWORK   docker network mode (default: host)
  AGENT_PULL                               base image refresh: missing (default) | always
  AGENT_BUILD_CACHE                        BuildKit cache export dir (empty disables)
  AGENT_PYTHON_IMAGE, AGENT_NODE_IMAGE, AGENT_UV_IMAGE
  APT_MIRROR, GITHUB_MIRROR, NVIM_RELEASE_BASE, NPM_REGISTRY, PYPI_INDEX
  AGENT_GIT_USER_NAME, AGENT_GIT_USER_EMAIL
  AGENT_OPENCODE_PORT_RANGE                (default: 49401-49425)
  CODEX_HOME, PI_SESSIONS_DIR, CODEX_SESSIONS_DIR
EOF
}

main() {
  cleanup() {
    if [[ -n ${opencode_cred_file:-} ]]; then
      rm -f -- "$opencode_cred_file"
    fi
  }
  trap cleanup EXIT HUP INT TERM

  if [[ ${1:-} == -v || ${1:-} == --verbose ]]; then
    verbose=1
    shift
  fi

  local command=${1:-help}
  [[ $# -gt 0 ]] && shift
  parse_options "$@"

  case $command in
  image) build_image ;;
  sync) sync_image ;;
  prepare) prepare_volumes ;;
  help | -h | --help) usage ;;
  test | lint | fmt | jupyter | shell | review)
    resolve_sessions
    resolve_credentials
    prepare_volumes
    case $command in
    # -p no:cacheprovider: pytest's cache would otherwise try to write .pytest_cache into the :ro tree.
    test) agent_run ro batch env QUANT_SOLUTIONS=1 pytest -p no:cacheprovider labs ;;
    lint) agent_run ro batch ruff check --no-cache . ;;
    fmt) agent_run rw batch bash -c 'ruff format . && ruff check --fix .' ;;
    jupyter)
      # Under host networking 127.0.0.1 is the host's loopback already; under bridge it has to
      # listen on all interfaces for the published port to reach it. Its own container name, so it
      # can run next to a `shell`.
      local ip=127.0.0.1
      if [[ ${AGENT_RUN_NETWORK:-host} != host ]]; then
        publish_ports=(-p 127.0.0.1:8888:8888)
        ip=0.0.0.0
      fi
      name="$name-jupyter"
      agent_run rw tty jupyter lab --no-browser --ip="$ip" --port=8888 --ServerApp.root_dir="$workspace"
      ;;
    shell) agent_run rw tty ;;
    review) agent_run ro tty ;;
    esac
    ;;
  *)
    printf 'agent.sh: unknown command: %s\n\n' "$command" >&2
    usage >&2
    exit 1
    ;;
  esac
}

main "$@"
