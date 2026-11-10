# quant-playground

一套由浅入深的量化交易 lab，面向**有数理和编程基础**的学习者。

写代码已经不是瓶颈：AI 能在几秒内写出一个回测、扫一千组参数、挑出最好的那组。
这套课练的是 AI 替不了你的部分——**判断**：这个回测能不能信？这个信号是真的还是噪声？
这笔收益是谁给你的？它在什么时候会消失？

## 为什么是 Python + 这些库

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.13 | 量化研究的事实标准；生态（numpy/pandas/scipy/statsmodels/sklearn）无可替代 |
| 依赖管理 | [uv](https://docs.astral.sh/uv/) + `uv.lock` | 快、锁文件精确可复现；锁文件也是 Docker 缓存分层的关键 |
| 数值/数据 | numpy, pandas, pyarrow | 向量化计算、时间序列、parquet 缓存 |
| 统计/计量 | scipy, statsmodels, arch | 回归、ADF/协整、GARCH |
| 机器学习 | scikit-learn, lightgbm | lab09 |
| 组合优化 | cvxpy | lab05 带约束的均值方差 |
| 行情数据（可选） | akshare（A 股）, yfinance（美股） | 只在“拓展练习”里用，见 `src/quantlab/realdata.py` |

**刻意没有选**一个现成的回测框架（backtrader / zipline / vectorbt / Backtesting.py）作为主线：
框架会把前视偏差、成交假设、成本模型这些最容易出错的地方藏起来。`quantlab` 里的向量化回测（lab02）
和事件驱动引擎（lab08）都很短，可以逐行读懂；看清楚这些假设在哪里之后，再去用任何框架都知道它哪里可能骗你。

## Lab 路线

| # | Lab | 难度 | 核心问题 |
| --- | --- | --- | --- |
| 00 | [收益与复利](labs/lab00_returns_basics/) | ★☆☆☆☆ | 收益在哪里可加；波动拖累；期望与中位数；为什么 μ 估不准 |
| 01 | [风险指标藏住了什么](labs/lab01_performance_metrics/) | ★☆☆☆☆ | Sharpe/回撤/VaR 各自隐含的假设；指标怎样被做得好看 |
| 02 | [回测的机制](labs/lab02_vectorized_backtest/) | ★★☆☆☆ | 信息集与成交时间线；前视偏差的各种形态；成本、换手、幸存者偏差 |
| 03 | [信号还是噪声](labs/lab03_backtest_statistics/) | ★★★☆☆ | Sharpe 的标准误；多重检验与 DSR；样本外检验的局限；先验 |
| 04 | [收益从哪来](labs/lab04_factor_investing/) | ★★★☆☆ | alpha 还是因子暴露；IC 与广度；风险溢价/行为偏差/流动性；拥挤与衰减 |
| 05 | [组合构建与估计误差](labs/lab05_portfolio_optimization/) | ★★★☆☆ | 优化器是误差放大器；收缩、约束与 1/N；事前风险为什么被低估 |
| 06 | [均值回归与配对交易](labs/lab06_mean_reversion_pairs/) | ★★★☆☆ | 伪回归；在很多对里“找到”协整；负偏的收益形态；关系断裂 |
| 07 | [波动率与仓位](labs/lab07_volatility_garch/) | ★★★★☆ | 波动率可预测而收益难预测；波动率目标；Kelly 与估计误差下的过度下注 |
| 08 | [成交假设](labs/lab08_event_driven_engine/) | ★★★★☆ | 收盘价能成交吗；限价单的逆向选择；滑点、冲击与成交量约束 |
| 09 | [机器学习与泄漏](labs/lab09_ml_purged_cv/) | ★★★★☆ | 特征/预处理/标签重叠/选择泄漏；Purged CV；有效样本量 |
| 10 | [卖期权，卖的是什么](labs/lab10_options_hedging/) | ★★★★☆ | BS 何时失效；离散对冲误差；波动率风险溢价就是卖保险 |
| 11 | [纸面 alpha 能剩下多少](labs/lab11_optimal_execution/) | ★★★★☆ | 平方根冲击；容量；执行要时间而 alpha 不等人；Almgren-Chriss |

建议顺序：00 → 01 → 02 → 03 是主干，按顺序做，**03 是整门课的核心**。之后 04–07 可以任意顺序；
08 接着 02；09、10、11 都以 03 为前置。

## 每个 lab 怎么做

```text
labs/labNN_xxx/
├── README.md        讲义：推导和直觉，重点在“为什么”和“什么时候不成立”（先读这个）
├── experiments.py   实验：每个 # %% 单元格先写下预测，再运行，对照结果
├── audit.py         审计：一份结论“显著”的研究代码，里面有让结论偏乐观的错误——找出来、修好
└── test_audit.py    审查任何回测时都值得做的性质检查，自动判分
solutions/labNN_xxx.py   修正后的 audit.py（做完再看）
```

1. **读讲义。**
2. **做实验。** 在 VS Code 里打开 `experiments.py`，每个 `# %%` 是一个单元格（`Shift+Enter` 运行）。
   **先在 `预测：` 后面写下答案再运行**——预测错的地方，正是直觉需要修正的地方。
3. **做审计。** 像 code review 一样读 `audit.py`，先在文件开头的「审计记录」里写下你发现的问题，
   再动手改，最后跑测试看看漏了什么。
4. **思考题**写在 notebook 里，写完可以让 Claude 批改。

```bash
make test LAB=03            # 检查你修改后的 audit.py
make test LAB=03 K=future   # 只跑名字含 future 的检查
make check LAB=03           # 用参考答案跑（确认测试本身是对的）
make progress               # 所有 lab 的审计通过情况
make jupyter                # JupyterLab（localhost:8888）
```

所有实现都在 `quantlab` 库里（`returns`、`risk`、`backtest`、`stats`、`factors`、`portfolio`、`pairs`、
`vol`、`engine`、`ml`、`options`、`execution`），直接调用，测试在 `tests/`。
审查任何回测时，配合 [回测陷阱清单](docs/pandas-pitfalls.md) 使用。

实验和测试都跑在**合成数据**上：离线、确定、而且“真相已知”——我们知道真实的 μ、β、半衰期、
GARCH 参数，也知道哪里根本没有信号，所以能检验一个结论到底对不对。
每个 lab 的“拓展”部分再用真实行情（`quantlab.realdata`）重做一遍。

## 环境

三种方式，任选其一。镜像里 Python 依赖已经烘焙在 `/opt/venv`。

### 1. VS Code Dev Container（推荐）

用 VS Code 打开本目录 → “Reopen in Container”。第一次打开时 `initializeCommand` 会调用
`hack/devcontainer.sh ensure` 构建镜像（几分钟），之后都是秒开。

改了 `.devcontainer/Dockerfile` 或 `uv.lock` 后：

```bash
hack/devcontainer.sh image     # 重建镜像（只有变化的 stage 会重跑）
# 然后 VS Code: "Dev Containers: Rebuild Container"
```

### 2. AI agent 容器

和 k8s-ai-platform 一样的 agent 沙箱：带 claude / codex / pi / opencode、Neovim、同一套 zsh 配置，
`--cap-drop ALL`，只挂载本仓库。

```bash
hack/agent.sh image     # 构建
hack/agent.sh shell     # 可写 shell（在里面跑 claude 等）
hack/agent.sh review    # 只读 shell
hack/agent.sh test      # 用参考答案跑全部测试
hack/agent.sh jupyter   # JupyterLab on localhost:8888
hack/agent.sh help
```

### 3. 宿主机

```bash
uv sync --all-extras    # 在 .venv 里装好依赖
make test LAB=00
```

### Docker 缓存设计

两个 Dockerfile 都是 multi-stage，每个下载任务各占一个 stage、都从上游镜像起步，
所以 BuildKit 并行执行，且一个版本号变化只让自己那个 stage 失效：

- `python-deps`：只 COPY `pyproject.toml`、`uv.lock`、`.python-version`。改 lab 代码、README、脚本
  都不会让它失效；只有依赖变化才重建，且 uv 下载缓存是 cache mount，已下过的 wheel 不再下载。
  pytest/ruff 配置刻意放在 `pytest.ini`/`ruff.toml` 而不是 `pyproject.toml`，也是为了这个。
- `shell-tools` / `editor-tools`：starship、shellcheck、zsh 插件、Neovim。
- `agent-cli`（仅 agent 镜像）：每个 CLI 一个 RUN，变化最频繁的 Claude Code 放最后。
- 最终 stage 只有 apt + COPY；venv 用 `COPY --link`，与上面的层互相独立。
- `hack/*.sh` 还把 BuildKit 缓存导出到 `~/.cache/quant-playground/`，Docker 清理缓存后也不用重新下载。

镜像源：apt 默认阿里云、PyPI 锁定为清华源（`uv.lock` 里记录了 URL）、GitHub 下载可用
`GITHUB_MIRROR` 覆盖。代理从宿主机环境变量透传，见 `hack/proxy-env.example`。
