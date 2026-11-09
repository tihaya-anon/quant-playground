# 常用命令。容器内（devcontainer / hack/agent.sh shell）直接用；宿主机上需要先 `uv sync`。
#
#   make test LAB=03            跑 lab03 的测试（你的实现）
#   make test LAB=03 K=sharpe   只跑名字含 sharpe 的测试
#   make check                  用参考答案跑全部测试（验证测试本身）
#   make progress               每个 lab 通过了多少测试
#   make jupyter                启动 JupyterLab（localhost:8888）

SHELL := /bin/bash
# 在容器里 venv 在 /opt/venv 且已在 PATH；在宿主机上走 uv run。
RUN := $(if $(VIRTUAL_ENV),,uv run --frozen)
LAB ?=
K ?=
LAB_DIR = $(if $(LAB),$(wildcard labs/lab$(LAB)_*),labs)
KFLAG = $(if $(K),-k "$(K)",)

.PHONY: help test check progress lint fmt jupyter clean

help:
	@sed -n '1,9p' Makefile | sed 's/^# \{0,1\}//'

test:
	$(RUN) pytest $(LAB_DIR) $(KFLAG)

check:
	QUANT_SOLUTIONS=1 $(RUN) pytest $(LAB_DIR) $(KFLAG)

progress:
	@for d in labs/lab*/; do \
	  n=$$(basename $$d); \
	  out=$$($(RUN) pytest $$d --tb=no -rN 2>/dev/null | tail -1); \
	  printf '%-36s %s\n' "$$n" "$$out"; \
	done

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

jupyter:
	$(RUN) jupyter lab --no-browser --ip=127.0.0.1 --port=8888

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
