# Development sandbox for AI coding agents working on quant-playground.
#
# Modelled on k8s-ai-platform's agent.Dockerfile. It is separate from .devcontainer/Dockerfile on
# purpose: that one is the lighter editor image; this one adds the AI CLIs (claude, codex, pi,
# opencode), Neovim and the rest of the agent sandbox. hack/agent.sh builds and runs it.
#
# Stage graph -- every leaf starts from an upstream image, not from a shared apt layer, so BuildKit
# runs them concurrently and a pin change in one rebuilds only that one:
#
#   uv ────────────┐
#   python-deps ───┤  (pyproject.toml + uv.lock only -> /opt/venv)
#   editor-tools ──┤  (nvim, starship, shellcheck, zsh plugins -> /opt, /usr/local/bin)
#   agent-cli ─────┤  (node image; npm -g into /opt/agent-cli, /opt/opencode)
#   pi-extensions ─┤
#   base ──────────┴─> final  (python image + apt + node runtime; COPY of everything above)
#
# Defaults are human-readable tags; hack/agent.sh resolves them to immutable digests before
# building so a moving Docker Hub tag does not silently change the sandbox.
ARG PYTHON_IMAGE=python:3.13-bookworm
ARG NODE_IMAGE=node:24-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

FROM ${UV_IMAGE} AS uv
FROM ${NODE_IMAGE} AS node

# ---------------------------------------------------------------------------------------------
# The Python environment. Identical in shape to the python-deps stage in .devcontainer/Dockerfile
# (and to its cache key, when the base digest and lock match), so the two images can share the
# layer. It sees only the three dependency files, so editing labs never invalidates it.
FROM ${PYTHON_IMAGE} AS python-deps
ARG http_proxy
ARG https_proxy
ARG no_proxy
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=120
WORKDIR /src
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    uv sync --frozen --all-extras --all-groups --no-install-project; \
    /opt/venv/bin/python -c "import numpy, pandas, scipy, statsmodels, sklearn, arch, cvxpy, lightgbm, matplotlib; print('python-deps ok')"

# ---------------------------------------------------------------------------------------------
# The editor side of the sandbox. Pinned to the versions the host runs, so a session in the
# container behaves like a session outside it.
FROM ${PYTHON_IMAGE} AS editor-tools
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG NVIM_VERSION=0.12.6
ARG STARSHIP_VERSION=1.25.0
ARG SHELLCHECK_VERSION=v0.11.0
ARG ZSH_AUTOSUGGESTIONS_COMMIT=85919cd1ffa7d2d5412f6d3fe437ebdbeeec4fc5
ARG ZSH_SYNTAX_HIGHLIGHTING_COMMIT=e6683f20e5c38d3e5447f45845d04ddda75d0f23
# A mirror with a different layout -- e.g. the NJU GitHub-release mirror -- is given as the whole
# directory: --build-arg NVIM_RELEASE_BASE=https://mirror.nju.edu.cn/github-release/neovim/neovim/Nvim%200.12.6
ARG NVIM_RELEASE_BASE=
ARG GITHUB_MIRROR=https://github.com
ARG TARGETARCH
# The stall guard (--speed-limit/--speed-time) catches a proxy that accepts the connection and then
# sends nothing; without it curl waits on that socket forever.
RUN set -eu; \
    curl_opts="--http1.1 --connect-timeout 15 --retry 5 --retry-delay 2 --retry-all-errors --speed-limit 1024 --speed-time 60"; \
    case "${TARGETARCH:-amd64}" in \
      amd64) nvim_arch=linux-x86_64; star_arch=x86_64-unknown-linux-gnu; sc_arch=x86_64 ;; \
      arm64) nvim_arch=linux-arm64;  star_arch=aarch64-unknown-linux-musl; sc_arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    mkdir -p /opt/editor-tools/bin; \
    nvim_base=${NVIM_RELEASE_BASE:-${GITHUB_MIRROR}/neovim/neovim/releases/download/v${NVIM_VERSION}}; \
    curl $curl_opts -fsSL "${nvim_base}/nvim-${nvim_arch}.tar.gz" | tar -C /opt -xz; \
    mv "/opt/nvim-${nvim_arch}" /opt/nvim; \
    curl $curl_opts -fsSL \
      "${GITHUB_MIRROR}/starship/starship/releases/download/v${STARSHIP_VERSION}/starship-${star_arch}.tar.gz" \
      | tar -C /opt/editor-tools/bin -xz starship; \
    curl $curl_opts -fsSL \
      "${GITHUB_MIRROR}/koalaman/shellcheck/releases/download/${SHELLCHECK_VERSION}/shellcheck-${SHELLCHECK_VERSION}.linux.${sc_arch}.tar.xz" \
      | tar -C /tmp -xJ; \
    mv "/tmp/shellcheck-${SHELLCHECK_VERSION}/shellcheck" /opt/editor-tools/bin/shellcheck; \
    for plugin in "zsh-autosuggestions:${ZSH_AUTOSUGGESTIONS_COMMIT}" \
                  "zsh-syntax-highlighting:${ZSH_SYNTAX_HIGHLIGHTING_COMMIT}"; do \
      name="${plugin%%:*}"; commit="${plugin#*:}"; \
      mkdir -p "/opt/zsh-plugins/${name}"; \
      curl $curl_opts -fsSL "${GITHUB_MIRROR}/zsh-users/${name}/archive/${commit}.tar.gz" \
        | tar -C "/opt/zsh-plugins/${name}" --strip-components=1 -xz; \
      test -r "/opt/zsh-plugins/${name}/${name}.zsh"; \
    done; \
    rm -rf /tmp/shellcheck-*; \
    /opt/nvim/bin/nvim --version | head -1; \
    /opt/editor-tools/bin/starship --version | head -1; \
    /opt/editor-tools/bin/shellcheck --version | grep version:

# ---------------------------------------------------------------------------------------------
# The agent CLIs. Off the node image directly: npm is all this needs, so it starts at once instead
# of waiting for the final stage's apt.
#
# One ARG and one RUN per CLI, in increasing order of how often the pin moves. BuildKit keys a
# layer on its expanded command string, so a single RUN naming every version would reinstall all
# of them on any bump; split, a bump invalidates only its own layer and those after it -- which is
# why Claude Code, the pin that moves most, is last.
#
# Separate commands are also load-bearing: npm drops the per-platform optionalDependencies of codex
# and claude-code when several roots are installed in one global command, and both then ship as
# stubs. The cache mount is what makes an interrupted ~600MB download survivable.
FROM ${NODE_IMAGE} AS agent-cli
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG NPM_REGISTRY=https://registry.npmjs.org

ARG MARKDOWNLINT_VERSION=0.23.2
ARG PRETTIER_VERSION=3.9.6
RUN --mount=type=cache,target=/root/.npm set -eu; \
    npm install -g --prefix /opt/agent-cli --registry="${NPM_REGISTRY}" \
        "markdownlint-cli2@${MARKDOWNLINT_VERSION}" \
        "prettier@${PRETTIER_VERSION}"

# OpenCode 2 is @opencode/cli. Its wrapper package pulls every platform build (~900MB) to run one
# 230MB binary, so name the architecture-specific package directly.
ARG OPENCODE_VERSION=2.1.4
ARG TARGETARCH
RUN --mount=type=cache,target=/root/.npm set -eu; \
    case "${TARGETARCH:-amd64}" in \
      amd64) opencode_pkg=@opencode/cli-linux-x64 ;; \
      arm64) opencode_pkg=@opencode/cli-linux-arm64 ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    if [ "$opencode_pkg" = "@opencode/cli-linux-x64" ] && ! grep -qwi avx2 /proc/cpuinfo 2>/dev/null; then \
      opencode_pkg=@opencode/cli-linux-x64-baseline; \
    fi; \
    npm install --prefix /opt/opencode --no-save --no-package-lock --ignore-scripts \
        --registry="${NPM_REGISTRY}" "${opencode_pkg}@${OPENCODE_VERSION}"; \
    mkdir -p /opt/opencode/bin; \
    ln -sf "../node_modules/${opencode_pkg}/bin/opencode" /opt/opencode/bin/opencode

ARG PI_VERSION=1.4.0
RUN --mount=type=cache,target=/root/.npm set -eu; \
    npm install -g --prefix /opt/agent-cli --include=optional --registry="${NPM_REGISTRY}" \
        --ignore-scripts \
        "@earendil-works/pi-coding-agent@${PI_VERSION}"

ARG CODEX_VERSION=0.154.0
RUN --mount=type=cache,target=/root/.npm set -eu; \
    npm install -g --prefix /opt/agent-cli --include=optional --registry="${NPM_REGISTRY}" \
        "@openai/codex@${CODEX_VERSION}"

# --allow-scripts is not optional: npm 11.19+ skips dependency install scripts unless named, and
# only *warns*. Claude Code copies its native binary into place from its own postinstall, so
# without this the build succeeds and the image ships a stub.
ARG CLAUDE_VERSION=2.1.278
RUN --mount=type=cache,target=/root/.npm set -eu; \
    npm install -g --prefix /opt/agent-cli --include=optional --registry="${NPM_REGISTRY}" \
        --allow-scripts=@anthropic-ai/claude-code \
        "@anthropic-ai/claude-code@${CLAUDE_VERSION}"

# Both failure modes above are silent, so run every CLI before the image may exist.
RUN set -eu; \
    export PATH="/opt/agent-cli/bin:/opt/opencode/bin:${PATH}"; \
    for cli in claude codex pi opencode; do \
      if ! "$cli" --version; then \
        echo "agent.Dockerfile: '$cli --version' failed: a skipped postinstall (--allow-scripts)," >&2; \
        echo "  a dropped per-platform optionalDependency, or NPM_REGISTRY=${NPM_REGISTRY} not serving it." >&2; \
        exit 1; \
      fi; \
    done

FROM ${NODE_IMAGE} AS pi-extensions
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PI_READER_VERSION=0.3.3
ARG PI_BACKGROUND_TASKS_VERSION=2.6.10
ARG NPM_REGISTRY=https://registry.npmjs.org
RUN --mount=type=cache,target=/root/.npm set -eu; \
    npm install --prefix /home/agent/.pi/agent/npm --registry="${NPM_REGISTRY}" \
        "@inobit/pi-reader@${PI_READER_VERSION}" \
        "pi-background-tasks@${PI_BACKGROUND_TASKS_VERSION}"; \
    printf '%s\n' \
        '{' \
        '  "packages": [' \
        "    \"npm:@inobit/pi-reader@${PI_READER_VERSION}\"," \
        "    \"npm:pi-background-tasks@${PI_BACKGROUND_TASKS_VERSION}\"" \
        '  ]' \
        '}' \
        > /home/agent/.pi/agent/settings.json

# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS base
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG APT_MIRROR=mirrors.aliyun.com
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eu; \
    for f in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
        if [ -f "$f" ]; then sed -i "s|deb.debian.org|${APT_MIRROR}|g" "$f"; fi; \
    done; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        make \
        procps \
        ripgrep \
        tree \
        zsh \
        fd-find \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v fdfind)" /usr/local/bin/fd

# node/npm/npx for the CLIs. A whole-prefix copy merges into /usr/local without touching Python.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version

FROM base

ARG USERNAME=agent
ARG USER_UID=1000
ARG USER_GID=1000

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=agent-cli /opt/agent-cli /opt/agent-cli
COPY --from=agent-cli /opt/opencode /opt/opencode
COPY --from=editor-tools /opt/nvim /opt/nvim
COPY --from=editor-tools /opt/editor-tools/bin/ /usr/local/bin/
RUN set -eu; \
    ln -sf /opt/nvim/bin/nvim /usr/local/bin/nvim; \
    for cli in claude codex pi markdownlint-cli2 prettier; do \
      ln -sf "/opt/agent-cli/bin/${cli}" "/usr/local/bin/${cli}"; \
    done; \
    ln -sf /opt/opencode/bin/opencode /usr/local/bin/opencode

# CLAUDE_CONFIG_DIR: every piece of Claude Code state in one directory, backed by one named volume.
# DISABLE_AUTOUPDATER: the CLIs live in a root-owned prefix and are pinned above; bump and rebuild.
# PATH puts the bypass wrappers (hack/agent-bin) ahead of the real binaries, and /opt/venv/bin
# ahead of the system Python, so `python`, `pytest`, `ruff` and `jupyter` are the locked ones.
# PYTHONPYCACHEPREFIX keeps __pycache__ out of the bind-mounted checkout (and works under :ro).
# PYTHONPATH makes `import quantlab` work without installing the project (package = false).
ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ENV CLAUDE_CONFIG_DIR=/home/agent/.claude \
    DISABLE_AUTOUPDATER=1 \
    LANG=C.UTF-8 \
    SHELL=/bin/zsh \
    TERM=xterm-256color \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/agent/.cache/uv \
    UV_DEFAULT_INDEX=${PYPI_INDEX} \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTHONPATH=/workspace/quant-playground/src:/workspace/quant-playground \
    PATH=/usr/local/libexec/agent-bin:/opt/agent-cli/bin:/opt/venv/bin:${PATH}

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/zsh "${USERNAME}" \
    && mkdir -p \
        /workspace/quant-playground \
        /home/agent/.cache/uv \
        /home/agent/.pi/agent/npm \
        /home/agent/.codex \
        /home/agent/.claude \
        /home/agent/.local/share/opencode \
        /home/agent/.local/state \
        /home/agent/.cache/opencode \
    && chown -R "${USERNAME}:${USERNAME}" /workspace /home/agent

COPY --from=pi-extensions --chown=${USER_UID}:${USER_GID} /home/agent/.pi/ /home/agent/.pi/

# Shell and editor configuration, tracked in hack/agent-home/ so the container's environment is
# reviewed in the same diff as the code.
COPY --chown=${USER_UID}:${USER_GID} hack/agent-home/ /home/agent/
COPY --from=editor-tools --chown=${USER_UID}:${USER_GID} /opt/zsh-plugins/ /home/agent/.local/share/zsh/plugins/

COPY hack/agent-entrypoint.sh /usr/local/bin/agent-entrypoint
# Root-owned and 0755, so the agent user cannot edit its own leash. See hack/agent-bin/claude.
COPY hack/agent-bin/ /usr/local/libexec/agent-bin/
RUN chmod 0755 /usr/local/bin/agent-entrypoint /usr/local/libexec/agent-bin/*

# The venv last among the big layers, so a dependency change re-writes only this layer and the
# small ones after it. --link: independent of the layers above, so an apt or CLI change does not
# re-write the venv either. Owned by the agent user so `uv add` works without root.
COPY --link --chown=${USER_UID}:${USER_GID} --from=python-deps /opt/venv /opt/venv

USER ${USERNAME}
WORKDIR /workspace/quant-playground

ENTRYPOINT ["/usr/local/bin/agent-entrypoint"]
CMD ["/bin/zsh", "-l"]

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
