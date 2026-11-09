import os

import matplotlib

# 测试里不弹窗。
matplotlib.use("Agg")


def pytest_report_header(config):
    who = (
        "solutions/ (参考答案)"
        if os.environ.get("QUANT_SOLUTIONS") == "1"
        else "labs/*/lab.py (你的实现)"
    )
    return f"quant-playground: testing {who}"
