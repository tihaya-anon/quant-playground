"""讲义插图的统一风格。

labs/*/figures/make.py 在作图前调用 use_lecture_style()，
保证所有图看起来一致、在 Markdown 预览里清晰。

两个关键设置（都是为了在浏览器 / VS Code 预览里不发虚、不出锯齿）：

- svg.fonttype = "none"：文字以 <text> 写入 SVG，由浏览器带 hinting 渲染。matplotlib 默认把每个字形
  转成轮廓路径，缩小显示时没有 hinting，小字号会显得粗糙。
- 图宽按讲义的显示宽度（约 760px）来定，字号按 1:1 显示时的大小来定，避免整张图被缩小显示。

图中文字用英文：镜像里没有中文字体，中文说明写在讲义的图注里。
"""

from __future__ import annotations

import matplotlib

# 讲义正文宽度约 760px；matplotlib 的 SVG 以 72pt/in 输出，浏览器按 96px/in 显示，所以 8in ≈ 768px。
FULL_WIDTH = 8.0
HALF_WIDTH = 3.9

COLORS = ["#2a6fb0", "#d1732b", "#3a9a5b", "#8e5bb5", "#c4473f", "#6b6b6b"]


def use_lecture_style() -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=COLORS),
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# ---------------------------------------------------------------------------------------------
# 把字体嵌进 SVG
#
# svg.fonttype = "none" 时，matplotlib 只写字体名：普通文字是 'DejaVu Sans', 'Helvetica', 'Arial',
# sans-serif，数学公式里的字符（u_{i,t}、γ 等）只写 'DejaVu Sans'，没有后备。在没装 DejaVu Sans 的
# 机器上（Windows、macOS），普通文字退回 Arial，公式退回浏览器默认的衬线字体，同一张图里就混着
# 有衬线和无衬线两种字。这里把图中实际用到的字符从 matplotlib 自带的 DejaVu Sans 里子集化出来，
# 以 @font-face 嵌进 SVG，任何机器（包括 GitHub）上看到的都是同一套字体。


def _font_dir() -> str:
    import os

    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


def _subset_woff_base64(font_file: str, text: str) -> str:
    import base64
    import io
    import logging

    from fontTools import subset

    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)  # FFTM 时间戳表无需子集化
    options = subset.Options()
    options.flavor = "woff"  # WOFF1 只需要 zlib
    options.layout_features = ["*"]
    options.name_IDs = []
    options.notdef_outline = True
    font = subset.load_font(font_file, options)
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    buf = io.BytesIO()
    subset.save_font(font, buf, options)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_svg(fig, path) -> None:
    """保存为 SVG，并把用到的字形嵌入文件，保证在任何机器上都用同一套无衬线字体显示。"""
    import html
    import os
    import re
    from pathlib import Path

    path = Path(path)
    fig.savefig(path, format="svg")
    svg = path.read_text(encoding="utf-8")

    # <text>/<tspan> 里的字符，加上一个最小集合（数字、基本拉丁、常用符号），保证不缺字
    used = "".join(html.unescape(t) for t in re.findall(r">([^<>]+)<", svg))
    chars = "".join(sorted(set(used) | set("0123456789.,-+−=()[]%abcdefghijklmnopqrstuvwxyz")))

    faces = []
    for style, file in [("normal", "DejaVuSans.ttf"), ("oblique", "DejaVuSans-Oblique.ttf")]:
        data = _subset_woff_base64(os.path.join(_font_dir(), file), chars)
        faces.append(
            "@font-face { font-family: 'DejaVu Sans'; "
            f"font-style: {style}; font-weight: normal; "
            f"src: url(data:font/woff;base64,{data}) format('woff'); }}"
        )
    style_block = "<defs><style>" + " ".join(faces) + "</style></defs>"

    # 公式里的 'DejaVu Sans' 也补上无衬线后备，万一某个查看器不支持嵌入字体
    svg = re.sub(
        r"font-family: 'DejaVu Sans'(?!,)",
        "font-family: 'DejaVu Sans', 'Helvetica', 'Arial', sans-serif",
        svg,
    )
    svg = re.sub(r"(<svg\b[^>]*>)", lambda m: m.group(1) + "\n" + style_block, svg, count=1)
    path.write_text(svg, encoding="utf-8")
