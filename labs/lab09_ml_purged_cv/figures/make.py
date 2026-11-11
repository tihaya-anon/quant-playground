"""生成 lab09 讲义里的图。

    PYTHONPATH=src:. uv run --frozen python labs/lab09_ml_purged_cv/figures/make.py

风格（字号、字体、SVG 文字设置）见 quantlab.plotting。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import KFold

from quantlab.ml import PurgedKFold
from quantlab.plotting import COLORS, FULL_WIDTH, save_svg, use_lecture_style

OUT = Path(__file__).resolve().parent

TRAIN, TEST, DROPPED = "#c9d6e8", COLORS[1], "#f2f2f2"


def purged_cv_timeline() -> None:
    """同一折测试集：打乱的 KFold 让测试样本被训练样本包围；Purged K-Fold 在两侧留出空隙。"""
    n, horizon, n_splits, fold = 100, 10, 5, 2
    idx = pd.bdate_range("2020-01-01", periods=n + horizon)
    # 样本 i 在第 i 天预测，标签覆盖 [i, i + horizon]
    t1 = pd.Series(idx[np.arange(n) + horizon], index=idx[:n])

    kf_train, kf_test = list(KFold(n_splits, shuffle=True, random_state=0).split(np.arange(n)))[
        fold
    ]
    pk_train, pk_test = list(PurgedKFold(n_splits, t1, embargo_pct=0.15).split())[fold]

    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 2.9), sharex=True)
    rows = [
        (axes[0], kf_train, kf_test, "Shuffled KFold"),
        (axes[1], pk_train, pk_test, "Purged K-Fold + embargo"),
    ]
    for ax, train, test, title in rows:
        colors = np.full(n, DROPPED, dtype=object)
        colors[train] = TRAIN
        colors[test] = TEST
        ax.bar(np.arange(n), 1, width=1.0, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks([])
        ax.set_ylim(0, 1.55)
        ax.spines[["left"]].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11)

    # 在 purged 那一行标出 purge 和 embargo 的范围
    lo, hi = pk_test.min(), pk_test.max()
    dropped = np.setdiff1d(np.arange(n), np.concatenate([pk_train, pk_test]))
    before = dropped[dropped < lo]
    after = dropped[dropped > hi]
    # 测试折之后：标签区间与测试折重叠的是 purge，再往后的是 embargo
    test_end = hi + horizon
    after_purge = after[after <= test_end]
    after_emb = after[after > test_end]
    ax = axes[1]
    for seg, label in [(before, "purge"), (after_purge, "purge"), (after_emb, "embargo")]:
        if len(seg):
            ax.annotate(
                "",
                xy=(seg.min() - 0.5, 1.2),
                xytext=(seg.max() + 0.5, 1.2),
                arrowprops={"arrowstyle": "<->", "lw": 1.0},
            )
            if label == "embargo":  # 窄，标签放在箭头右侧，避免和旁边的 purge 标签重叠
                ax.text(
                    seg.max() + 1.5,
                    1.2,
                    f"{label} ({len(seg)})",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
            else:
                ax.text(
                    seg.mean(), 1.28, f"{label} ({len(seg)})", ha="center", va="bottom", fontsize=9
                )

    # 打乱那一行：标出一个测试样本的标签区间，它和左右的训练样本标签几乎完全重叠
    k = int(kf_test[(kf_test > 20) & (kf_test < 80)][0])
    ax = axes[0]
    ax.annotate(
        "",
        xy=(k - 0.5, 1.2),
        xytext=(k + horizon + 0.5, 1.2),
        arrowprops={"arrowstyle": "<->", "lw": 1.0},
    )
    ax.text(k + horizon / 2, 1.28, "label window of one test sample", ha="center", fontsize=9)

    axes[1].set_xlabel("sample (time order)")
    fig.legend(
        handles=[
            Patch(color=TRAIN, label="train"),
            Patch(color=TEST, label="test"),
            Patch(facecolor=DROPPED, edgecolor="#bbbbbb", label="dropped"),
        ],
        loc="upper right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1.0, 1.02),
    )
    fig.tight_layout()
    save_svg(fig, OUT / "purged_cv.svg")
    plt.close(fig)

    neighbors = sum(
        ((kf_train >= j - horizon) & (kf_train <= j + horizon)).sum() > 0 for j in kf_test
    )
    print(f"shuffled: test {len(kf_test)}, train {len(kf_train)}")
    print(f"  test samples with a train sample within {horizon} days: {neighbors}/{len(kf_test)}")
    print(
        f"purged: test {len(pk_test)}, train {len(pk_train)}, "
        f"purge before {len(before)}, purge after {len(after_purge)}, embargo {len(after_emb)}"
    )


if __name__ == "__main__":
    use_lecture_style()
    purged_cv_timeline()
