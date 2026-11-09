#!/usr/bin/env bash
# VS Code devcontainer image: build it, and keep it aligned with the published copy.
#
# Ported from k8s-ai-platform's hack/devcontainer.sh. The local and remote repository are the same
# name, because devcontainer.json's "image" field is the single name both VS Code and this script
# have to agree on, and on a fresh clone that name has to be pullable (or buildable: see `ensure`).
#
# Usage: hack/devcontainer.sh [-v] <command>
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

prog=devcontainer.sh
# shellcheck source=lib/registry.sh
source "$repo_root/hack/lib/registry.sh"

# Must match the "image" field in .devcontainer/devcontainer.json; check_json_image enforces that.
image_repo=${DEVCONTAINER_IMAGE_REPOSITORY:-chihayaanon0908/quant-playground-dev}
remote_image_repo=${DEVCONTAINER_IMAGE_REMOTE_REPOSITORY:-$image_repo}
devcontainer_json=.devcontainer/devcontainer.json
dockerfile=.devcontainer/Dockerfile

verbose=0
force_latest=0
pull_policy=${DEVCONTAINER_PULL:-missing}

# BuildKit's own cache is pruned when Docker's disk fills, and the layer worth keeping -- the
# venv -- is exactly what a prune takes. A local export survives that. Empty disables it; the `-`
# rather than `:-` is deliberate so DEVCONTAINER_BUILD_CACHE= means "off".
build_cache_dir=${DEVCONTAINER_BUILD_CACHE-${XDG_CACHE_HOME:-$HOME/.cache}/quant-playground/devcontainer-buildkit}

# A build tagged with anything other than what devcontainer.json names produces an image VS Code
# will never open. devcontainer.json is JSONC, so grep rather than jq.
check_json_image() {
  local declared
  declared=$(grep -oE '^[[:space:]]*"image"[[:space:]]*:[[:space:]]*"[^"]+"' "$devcontainer_json" |
    head -1 | sed -E 's/.*"image"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  if [[ -z $declared ]]; then
    die "no \"image\" field found in $devcontainer_json"
  fi
  if [[ $declared != "$image_repo:latest" ]]; then
    die "$devcontainer_json opens $declared, but this script builds $image_repo:latest -- align them, or set DEVCONTAINER_IMAGE_REPOSITORY"
  fi
}

build_image() {
  check_json_image

  local git_sha commit_image latest_image
  git_sha=$(current_git_sha)
  commit_image="$image_repo:$git_sha"
  latest_image="$image_repo:latest"

  local python_image uv_image python_digest uv_digest
  python_image=${DEVCONTAINER_PYTHON_IMAGE:-$(dockerfile_arg_default "$dockerfile" PYTHON_IMAGE)}
  uv_image=${DEVCONTAINER_UV_IMAGE:-$(dockerfile_arg_default "$dockerfile" UV_IMAGE)}
  pull_base_image "$python_image"
  pull_base_image "$uv_image"
  python_digest=$(base_image_digest "$python_image")
  uv_digest=$(base_image_digest "$uv_image")

  # Host network by default, because a proxy bound to the host's loopback is unreachable from the
  # bridge network a build normally gets.
  local build_flags=(--network "${DEVCONTAINER_BUILD_NETWORK:-host}")
  append_build_cache_flags build_flags "$build_cache_dir"
  local build_args=(
    --build-arg PYTHON_IMAGE="$python_digest"
    --build-arg UV_IMAGE="$uv_digest"
    --build-arg GIT_SHA="$git_sha"
  )
  # Mirrors keep their defaults in the Dockerfile and are passed only when the environment sets them.
  local mirror_var
  for mirror_var in APT_MIRROR GITHUB_MIRROR PYPI_INDEX; do
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

  ((verbose)) || printf '%s: building %s and %s (-v to follow)\n' "$prog" "$commit_image" "$latest_image" >&2
  run_step docker build \
    "${build_flags[@]}" \
    -f "$dockerfile" \
    "${build_args[@]}" \
    -t "$commit_image" \
    -t "$latest_image" .

  printf '%s: built. Run "Dev Containers: Rebuild Container" in VS Code to pick it up.\n' "$prog" >&2
}

# For devcontainer.json's initializeCommand: build only when there is no image at all. Never
# rebuilds an existing one -- that stays an explicit `image`, so opening the folder is never a
# surprise multi-minute build.
ensure_image() {
  check_json_image
  if docker image inspect "$image_repo:latest" >/dev/null 2>&1; then
    return 0
  fi
  printf '%s: %s not found locally; building it once (this takes a few minutes)\n' "$prog" "$image_repo:latest" >&2
  verbose=1
  build_image
}

usage() {
  cat <<'EOF'
Usage: hack/devcontainer.sh [-v] <command>

Commands:
  image     build the devcontainer image locally, tagged with the commit sha and latest
  ensure    build it only if no local image exists (what devcontainer.json runs on open)
  sync      align the local and published images for this commit: pull, build, or push as needed
  help      this text

A build only puts the image on disk. VS Code picks it up on the next
"Dev Containers: Rebuild Container"; reopening an existing container does not.

Options:
  -v, --verbose  print the docker commands and the build output that is otherwise captured.
  --force-latest move the remote latest tag to this commit even when sync cannot prove this
                 checkout is newer.

Environment:
  DEVCONTAINER_IMAGE_REPOSITORY   must match "image" in devcontainer.json
                                  (default: chihayaanon0908/quant-playground-dev)
  DEVCONTAINER_IMAGE_REMOTE_REPOSITORY
                                  repository `sync` publishes to (default: the same)
  DEVCONTAINER_PYTHON_IMAGE, DEVCONTAINER_UV_IMAGE
                                  base images (default: the ARG defaults in the Dockerfile)
  DEVCONTAINER_PULL               missing (default) | always
  DEVCONTAINER_BUILD_CACHE        BuildKit cache export dir (default:
                                  ~/.cache/quant-playground/devcontainer-buildkit; empty disables)
  DEVCONTAINER_BUILD_NETWORK      docker build network (default: host)
  APT_MIRROR                      Debian mirror (default: mirrors.aliyun.com)
  GITHUB_MIRROR                   GitHub release prefix (default: https://github.com)
  PYPI_INDEX                      index for `uv add` inside the container (default: Tsinghua);
                                  uv.lock pins the URLs `uv sync` downloads from
  http_proxy, https_proxy, all_proxy, no_proxy (and uppercase)
                                  passed through to the build when set
EOF
}

main() {
  while [[ ${1:-} == -* ]]; do
    case $1 in
    -v | --verbose) verbose=1 ;;
    --force-latest) force_latest=1 ;;
    -h | --help)
      usage
      return 0
      ;;
    *) die "unknown option: $1" ;;
    esac
    shift
  done

  local command=${1:-help}
  [[ $# -gt 0 ]] && shift
  while [[ $# -gt 0 ]]; do
    case $1 in
    -v | --verbose) verbose=1 ;;
    --force-latest) force_latest=1 ;;
    *) die "unknown argument: $1" ;;
    esac
    shift
  done

  case $command in
  image) build_image ;;
  ensure) ensure_image ;;
  sync)
    check_json_image
    sync_image
    ;;
  help | -h | --help) usage ;;
  *)
    printf '%s: unknown command: %s\n\n' "$prog" "$command" >&2
    usage >&2
    exit 1
    ;;
  esac
}

main "$@"
