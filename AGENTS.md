# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A set of quantitative-trading learning labs (Chinese prose, Python code). The owner is the
learner. Each `labs/labNN_*/` has `README.md` (theory + tasks), `lab.py` (stubs the learner fills
in), `test_lab.py`; reference answers are in `solutions/labNN_*.py`. `labs/_harness.py` makes the
tests load `lab.py` by default and `solutions/` when `QUANT_SOLUTIONS=1`.

## Rules

- **Do not fill in `labs/*/lab.py` for the learner** unless explicitly asked. Hints, explanations
  and reviews of their code are the default mode of help. When asked to review, run
  `make test LAB=NN` and explain failures in terms of the underlying concept.
- Tests run on synthetic data from `src/quantlab/data.py` with fixed seeds: keep them offline,
  deterministic, and under ~30s per lab. Statistical assertions need generous tolerances.
- Any change to a test must still pass with `make check` (reference solutions).
- Keep `pyproject.toml` limited to dependencies: the Docker venv layer is keyed on it. Tool config
  lives in `pytest.ini` and `ruff.toml`. After `uv add`, rebuild images (`hack/devcontainer.sh image`,
  `hack/agent.sh image`).
- `make lint` (ruff) and `shellcheck -x` on `hack/` must pass; the pre-commit hook enforces both.

## Commands

```bash
make test LAB=03        # learner's implementation
make check              # reference solutions, all labs
make lint / make fmt
hack/agent.sh help      # agent container
hack/devcontainer.sh help
```
