# notebooks

- [`pandas_tour.py`](pandas_tour.py)：这门课里用到的 pandas 操作速览。按 `# %%` 分成单元格，
  每格讲一类操作做什么、在哪个 lab 出现、最容易踩什么坑，并附可运行的小例子。
  在 VS Code 里逐格 `Shift+Enter` 运行。

做思考题、画图、跑真实数据的地方。`quantlab` 已在 PYTHONPATH 上（容器内），可以直接：

```python
from quantlab import data, perf, realdata
```

想在 notebook 里调用自己写的 lab 函数：

```python
from labs._harness import load_impl

lab02 = load_impl("labs/lab02_vectorized_backtest/test_lab.py")  # 加载 lab02 的 lab.py
```
