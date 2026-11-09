# VS Code devcontainer image for quant-playground.
#
# Shape borrowed from k8s-ai-platform's .devcontainer/Dockerfile: every download sits in a stage of
# its own, off the base image rather than off a shared apt layer, so BuildKit runs them
# concurrently, and a version bump in one of them invalidates one stage instead of every layer
# below it. Each stage leaves its output under /opt, and the final stage is apt + COPY and little
# else.
#
# The stage that matters most for caching is python-deps. It sees *only* pyproject.toml, uv.lock
# and .python-version, so editing a lab, a README or this repository's scripts never touches it;
# only a dependency change (uv add / uv lock) rebuilds the venv, and even then the uv cache mount
# means wheels already downloaded are not downloaded again.
#
# The user creation and sudo are still handled by the common-utils feature in devcontainer.json;
# this file only owns what belongs in a layer.
#
# Built by hack/devcontainer.sh, not by the devcontainer CLI: the script pins the base to a digest,
# passes the host's proxy and mirror settings through, and stamps the revision label that `sync`
# reads back from the registry.

ARG PYTHON_IMAGE=python:3.13-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

# uv is a single static binary; its official image is the cheapest way to get a pinned one.
FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------------------------
# The Python environment: every third-party package the labs use, resolved from uv.lock.
#
# UV_PROJECT_ENVIRONMENT puts the venv at /opt/venv instead of the project's .venv: the workspace is
# a bind mount, so a .venv inside it would be shadowed by (or leak onto) the host checkout. The
# final images point UV_PROJECT_ENVIRONMENT at the same path, so `uv sync` / `uv add` inside the
# container update this venv in place.
#
# --link-mode=copy because the cache is a mount that does not exist in the final image: hardlinks
# into it would dangle. --compile-bytecode so the first `import pandas` in a fresh container does
# not spend seconds writing .pyc files.
#
# The lockfile records the index it was resolved against (the Tsinghua PyPI mirror), and
# `uv sync --frozen` downloads from exactly those URLs, so PYPI_INDEX below only affects packages
# added later with `uv add`.
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
# Shell tools, same pins as k8s-ai-platform so the shell feels identical in every container.
#
# The stall guard and retries are load-bearing: a proxy can accept the connection and then send
# nothing, and without --speed-limit/--speed-time curl waits on that socket forever.
#
# The zsh plugins land under /opt because the agent user does not exist at image build time --
# common-utils creates it when the features run -- so post-install.sh symlinks them into
# $HOME/.local/share/zsh/plugins, the path hack/agent-home/.zshrc reads.
FROM ${PYTHON_IMAGE} AS shell-tools
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG TARGETARCH
ARG GITHUB_MIRROR=https://github.com
ARG STARSHIP_VERSION=1.25.0
ARG SHELLCHECK_VERSION=v0.11.0
ARG ZSH_AUTOSUGGESTIONS_COMMIT=85919cd1ffa7d2d5412f6d3fe437ebdbeeec4fc5
ARG ZSH_SYNTAX_HIGHLIGHTING_COMMIT=e6683f20e5c38d3e5447f45845d04ddda75d0f23
RUN set -eu; \
    curl_opts="--http1.1 --connect-timeout 15 --retry 5 --retry-delay 2 --retry-all-errors --speed-limit 1024 --speed-time 60"; \
    arch="${TARGETARCH:-amd64}"; \
    mkdir -p /opt/shell-tools/bin; \
    case "$arch" in \
      amd64) star_arch=x86_64-unknown-linux-gnu; sc_arch=x86_64 ;; \
      arm64) star_arch=aarch64-unknown-linux-musl; sc_arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH: $arch" >&2; exit 1 ;; \
    esac; \
    curl $curl_opts -fsSL \
        "${GITHUB_MIRROR}/starship/starship/releases/download/v${STARSHIP_VERSION}/starship-${star_arch}.tar.gz" \
        | tar -C /opt/shell-tools/bin -xz starship; \
    curl $curl_opts -fsSL \
        "${GITHUB_MIRROR}/koalaman/shellcheck/releases/download/${SHELLCHECK_VERSION}/shellcheck-${SHELLCHECK_VERSION}.linux.${sc_arch}.tar.xz" \
        | tar -C /tmp -xJ; \
    mv "/tmp/shellcheck-${SHELLCHECK_VERSION}/shellcheck" /opt/shell-tools/bin/shellcheck; \
    rm -rf "/tmp/shellcheck-${SHELLCHECK_VERSION}"; \
    for plugin in "zsh-autosuggestions:${ZSH_AUTOSUGGESTIONS_COMMIT}" \
                  "zsh-syntax-highlighting:${ZSH_SYNTAX_HIGHLIGHTING_COMMIT}"; do \
      name="${plugin%%:*}"; commit="${plugin#*:}"; \
      curl $curl_opts -fsSL -o "/tmp/${name}.tar.gz" \
        "${GITHUB_MIRROR}/zsh-users/${name}/archive/${commit}.tar.gz"; \
      mkdir -p "/opt/zsh-plugins/${name}"; \
      tar -C "/opt/zsh-plugins/${name}" --strip-components=1 -xzf "/tmp/${name}.tar.gz"; \
      test -r "/opt/zsh-plugins/${name}/${name}.zsh"; \
    done; \
    rm -f /tmp/zsh-*.tar.gz; \
    /opt/shell-tools/bin/starship --version | head -1; \
    /opt/shell-tools/bin/shellcheck --version | grep version:

# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
ARG http_proxy
ARG https_proxy
ARG no_proxy

# apt, in one invocation. The list is zsh plus the small tools the python image does not ship,
# and the package set common-utils would otherwise install on every Rebuild (its install step
# skips packages that are already present, which turns that feature into a few seconds of
# user creation).
#
# The base has moved between apt source layouts across releases (deb822 debian.sources vs a plain
# sources.list), so rewrite whichever is present. `if` rather than `[ -f ] && sed` because a
# trailing failed test would leave the loop exit status non-zero.
ARG APT_MIRROR=mirrors.aliyun.com
RUN set -eu; \
    export DEBIAN_FRONTEND=noninteractive; \
    for f in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
        if [ -f "$f" ]; then sed -i "s|deb.debian.org|${APT_MIRROR}|g" "$f"; fi; \
    done; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        zsh \
        fd-find \
        ripgrep \
        jq \
        less \
        tree \
        htop \
        sudo \
        locales \
        lsof \
        man-db \
        nano \
        procps \
        psmisc \
        rsync \
        vim-tiny \
        bash-completion \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v fdfind)" /usr/local/bin/fd

# The environment every shell, the Python extension and Jupyter see.
#
# PYTHONPYCACHEPREFIX keeps __pycache__ out of the workspace. The checkout is shared with the host
# and with the agent container, whose interpreters live at different paths, so .pyc files written
# by one side are stale for the others; under /tmp they stay per-container.
#
# PYTHONPATH makes `import quantlab` work everywhere (notebooks, `python -c`, the REPL) without
# installing the project into the venv -- see `package = false` in pyproject.toml. The path is
# the workspace folder devcontainer.json mounts; pytest gets the same from its own config.
ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${PYPI_INDEX} \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTHONPATH=/workspaces/quant-playground/src:/workspaces/quant-playground \
    LANG=C.UTF-8

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=shell-tools /opt/shell-tools/bin/ /usr/local/bin/
COPY --from=shell-tools /opt/zsh-plugins/ /opt/zsh-plugins/
# Owned by uid 1000, the user common-utils creates, so `uv add` inside the container can write the
# venv without sudo. --link makes this layer independent of the ones above: an apt change does not
# force the (large) venv layer to be re-written, and vice versa.
COPY --link --chown=1000:1000 --from=python-deps /opt/venv /opt/venv

# The commit this image was built from. `hack/devcontainer.sh sync` reads it back from the registry
# manifest to decide whether latest may move, without pulling the image. Declared last so a new
# commit invalidates one metadata layer instead of the layers above it.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
