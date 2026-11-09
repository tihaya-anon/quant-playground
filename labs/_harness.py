"""测试框架：决定测试跑的是“你的实现”还是“参考答案”。

每个 lab 目录里的 test_lab.py 都这样开头：

    from labs._harness import load_impl
    lab = load_impl(__file__)

默认加载同目录的 lab.py（你要填的那个）。设置环境变量 QUANT_SOLUTIONS=1 时改为加载
solutions/<lab 目录名>.py —— 这是用来确认测试本身是对的，或者卡住时对照用的。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def impl_path(test_file: str | Path) -> Path:
    lab_dir = Path(test_file).resolve().parent
    if os.environ.get("QUANT_SOLUTIONS") == "1":
        return ROOT / "solutions" / f"{lab_dir.name}.py"
    return lab_dir / "lab.py"


def load_impl(test_file: str | Path) -> ModuleType:
    path = impl_path(test_file)
    name = f"_impl_{path.parent.name}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
