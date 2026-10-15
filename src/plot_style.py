# -*- coding: utf-8 -*-
"""
所有图共用的样式 (src/plot_style.py)
- 中文字体：Windows 用微软雅黑/黑体，Linux 用 Noto Sans CJK，按顺序找第一个装了的
- 配色：分类色固定顺序取，不循环；最多同时用三种颜色（前三个在色盲模拟下两两都能区分）
- 网格、坐标轴弱化，文字一律用深灰/中灰，不用数据颜色
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"

_installed = {f.name for f in font_manager.fontManager.ttflist}
_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "PingFang SC"]
plt.rcParams["font.sans-serif"] = [f for f in _candidates if f in _installed] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # 蓝、橙、青
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df"}


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=11, color=INK["primary"], loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK["secondary"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK["secondary"])
    ax.grid(True, color=INK["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["muted"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> figures/{name}")
