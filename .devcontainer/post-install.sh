#!/usr/bin/env bash
set -euo pipefail

# Everything heavy -- the Python venv, uv, starship, shellcheck, the zsh plugins -- is baked into
# the image by .devcontainer/Dockerfile. What is left is only what has to run against the live
# container: volume ownership, linking the baked zsh plugins into $HOME, and a check that the venv
# still matches uv.lock.

# Every external command this script runs, checked before any of them does work, so a missing
# binary is reported by name rather than as a bare "exit code 127" in the Dev Containers log.
missing=()
for cmd in sudo chsh zsh git uv python starship shellcheck; do
  command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "post-install.sh: not on PATH: ${missing[*]}" >&2
  echo "  sudo and chsh come from the common-utils feature; everything else is baked by" >&2
  echo "  .devcontainer/Dockerfile -- run hack/devcontainer.sh image and Rebuild Container." >&2
  exit 1
fi

# Docker creates a named volume's mount point, and every missing parent of a mount, owned by root.
for dir in "$HOME/.cache" "$HOME/.cache/uv" "$HOME/.config"; do
  if [ -d "$dir" ]; then sudo chown "$(id -u):$(id -g)" "$dir"; fi
done
# updateRemoteUserUID may have moved the user off 1000, which is the uid the venv was chowned to
# at build time. Only then is the (slow, recursive) chown worth doing.
if [ "$(stat -c %u /opt/venv)" != "$(id -u)" ]; then
  sudo chown -R "$(id -u):$(id -g)" /opt/venv
fi

# The plugins are baked under /opt/zsh-plugins because the user does not exist at image build time;
# link them to the path hack/agent-home/.zshrc reads.
zsh_plugins="$HOME/.local/share/zsh/plugins"
mkdir -p "$zsh_plugins"
for name in zsh-autosuggestions zsh-syntax-highlighting; do
  rm -rf "${zsh_plugins:?}/${name}"
  ln -sfn "/opt/zsh-plugins/${name}" "${zsh_plugins}/${name}"
done

sudo chsh -s /bin/zsh "$(id -un)"

# Arm the repository's pre-commit hook. core.hooksPath lives in the untracked .git/config, so a
# fresh clone starts with it disarmed.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "$repo_root/.git" ]; then
  git config --global --add safe.directory "$repo_root" 2>/dev/null || true
  git -C "$repo_root" config core.hooksPath hack/git-hooks || true
fi

# A no-op when the image was built from the current uv.lock. When the lock moved after the image
# was built, this installs the difference into /opt/venv instead of failing the first import --
# and says so, because the durable fix is rebuilding the image.
if ! uv sync --frozen --all-extras --all-groups --check >/dev/null 2>&1; then
  echo "post-install.sh: uv.lock is newer than the image's venv; syncing (rebuild the image to bake it in)" >&2
  uv sync --frozen --all-extras --all-groups
fi

python --version
uv --version
python -c "import numpy, pandas, scipy; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__)"
zsh --version
starship --version | head -1
