# quant-playground

一套从零开始、由浅入深的量化交易 lab。面向**有数理和编程基础**的学习者：每个 lab 讲清楚
背后的数学，然后让你亲手把它实现出来，用自动化测试检验对错。

## 为什么是 Python + 这些库

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.13 | 量化研究的事实标准；生态（numpy/pandas/scipy/statsmodels/sklearn）无可替代 |
| 依赖管理 | [uv](https://docs.astral.sh/uv/) + `uv.lock` | 快、锁文件精确可复现；锁文件也是 Docker 缓存分层的关键 |
| 数值/数据 | numpy, pandas, pyarrow | 向量化计算、时间序列、parquet 缓存 |
| 统计/计量 | scipy, statsmodels, arch | 回归、ADF/协整、GARCH（lab 里先手写，再和这些库对照） |
| 机器学习 | scikit-learn, lightgbm | lab09 |
| 组合优化 | cvxpy | lab05 带约束的均值方差 |
| 行情数据（可选） | akshare（A 股）, yfinance（美股） | 只在“拓展练习”里用，见 `src/quantlab/realdata.py` |

**刻意没有选**一个现成的回测框架（backtrader / zipline / vectorbt / Backtesting.py）作为主线：
框架会把前视偏差、成交假设、成本模型这些最容易出错的地方藏起来。你会在 lab02 自己写向量化回测、
在 lab08 自己写事件驱动引擎——写过一遍之后，再去用任何框架都知道它在做什么、哪里可能骗你。

## Lab 路线

| # | Lab | 难度 | 核心内容 |
| --- | --- | --- | --- |
| 00 | [收益的算术](labs/lab00_returns_basics/) | ★☆☆☆☆ | 简单/对数收益、组合收益、GBM、μ 为什么估不准 |
| 01 | [绩效与风险指标](labs/lab01_performance_metrics/) | ★☆☆☆☆ | Sharpe/Sortino/回撤/Calmar/VaR/CVaR |
| 02 | [向量化回测](labs/lab02_vectorized_backtest/) | ★★☆☆☆ | 信号→仓位、前视偏差、交易成本、换手 |
| 03 | [回测的统计学](labs/lab03_backtest_statistics/) | ★★★☆☆ | Sharpe 标准误、bootstrap、PSR/DSR、过拟合、walk-forward |
| 04 | [因子投资](labs/lab04_factor_investing/) | ★★★☆☆ | CAPM、IC/ICIR、分组收益、因子中性化 |
| 05 | [组合优化](labs/lab05_portfolio_optimization/) | ★★★☆☆ | 最小方差、切点组合、收缩估计、风险平价、cvxpy |
| 06 | [均值回归与配对交易](labs/lab06_mean_reversion_pairs/) | ★★★☆☆ | OU/半衰期、ADF、协整、z-score、Kalman |
| 07 | [波动率建模](labs/lab07_volatility_garch/) | ★★★★☆ | EWMA、手写 GARCH MLE、波动率目标 |
| 08 | [事件驱动回测引擎](labs/lab08_event_driven_engine/) | ★★★★☆ | 订单/成交/券商/组合的 OO 设计、会计恒等式 |
| 09 | [机器学习与 Purged CV](labs/lab09_ml_purged_cv/) | ★★★★☆ | 三重障碍标签、Purged K-Fold + embargo |
| 10 | [期权定价与对冲](labs/lab10_options_hedging/) | ★★★★☆ | BS、Greeks、隐含波动率、MC、离散 delta 对冲 |
| 11 | [最优执行](labs/lab11_optimal_execution/) | ★★★★★ | TWAP/VWAP、Almgren-Chriss、implementation shortfall |

建议顺序：00 → 01 → 02 → 03 是主干，必须按顺序；之后 04/05/06/07 可以任意顺序；
08 依赖 02 的直觉；09 依赖 03；10、11 相对独立。

## 每个 lab 怎么做

```text
labs/labNN_xxx/
├── README.md     理论 + 任务 + 思考题 + 拓展（先读这个）
├── lab.py        你要实现的函数：签名 + docstring 规格 + raise NotImplementedError
└── test_lab.py   自动化测试
solutions/labNN_xxx.py   参考答案（做完再看）
```

```bash
make test LAB=02            # 跑你的实现
make test LAB=02 K=cost     # 只跑名字含 cost 的测试
make check LAB=02           # 用参考答案跑（确认测试是对的 / 卡住时对照）
make progress               # 所有 lab 的通过情况
make jupyter                # 在 notebooks/ 里画图、做思考题
```

所有测试都跑在 `src/quantlab/data.py` 生成的**合成数据**上：离线、确定、而且“真相已知”——
我们知道真实的 μ、β、半衰期、GARCH 参数，所以能检验你的估计量是不是对的。
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
