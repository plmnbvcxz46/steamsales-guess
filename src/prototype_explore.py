# -*- coding: utf-8 -*-
"""
原型验证 (src/prototype_explore.py)

按原型法先跑最小链路：取 2018 抽样池（玩家数 >= 2 万，固定种子打乱）里最先采集到的 200 款付费游戏，
算 k = 2018-07-01 玩家数 / 截至当天的评测数，看两件事：
1. k 的分布长什么样（决定用均值还是中位数、要不要取对数）       -> figures/fig01_prototype_k_dist.png
2. 评测数和玩家数在对数坐标下是不是近似直线（决定用不用对数线性模型） -> figures/fig02_prototype_log_scatter.png
只读 crawler.py export 产出的 data/raw/*.csv，不访问网络。
"""
import random
import pathlib

import numpy as np
import pandas as pd

from plot_style import plt, COLORS, INK, style_axes, save

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
N_PROTO = 200


def load_prototype_sample():
    ars = pd.read_csv(RAW_DIR / "ars_players_2018.csv").drop_duplicates("appid")
    apps = pd.read_csv(RAW_DIR / "steam_appdetails.csv")
    rev = pd.read_csv(RAW_DIR / "steam_reviews_at_cutoff.csv")
    rev = rev[rev["cohort"] == "train_2018"]

    # 复现 crawler.py 第一轮的采集顺序：玩家数 >= 2 万，按种子 2018 打乱
    pool = ars[ars["players"] >= 20000].to_dict("records")
    random.Random(2018).shuffle(pool)
    order = {r["appid"]: i for i, r in enumerate(pool)}

    df = rev.merge(ars, on="appid").merge(apps, on="appid")
    df = df[(df["success"] == 1) & (df["app_type"] == "game") & (df["is_free"] == 0) & (df["initial_price_usd"] > 0)]
    df = df[(df["total_reviews"] > 0) & (df["release_date"] <= "2018-07-01")]
    df = df[df["appid"].isin(order)].copy()
    df["order"] = df["appid"].map(order)
    df = df.sort_values("order").head(N_PROTO).copy()
    df["k"] = df["players"] / df["total_reviews"]
    return df


def main():
    df = load_prototype_sample()
    k = df["k"].values
    lk = np.log(k)
    lr, lp = np.log10(df["total_reviews"].values), np.log10(df["players"].values)
    r = float(np.corrcoef(lr, lp)[0, 1])
    slope, intercept = np.polyfit(lr, lp, 1)
    summary = {
        "n": len(df), "k_mean": k.mean(), "k_median": np.median(k), "k_q25": np.percentile(k, 25),
        "k_q75": np.percentile(k, 75), "lnk_mean": lk.mean(), "lnk_std": lk.std(ddof=1),
        "pearson_r_log10": r, "fit_slope": slope, "fit_intercept": intercept,
    }
    out = BASE_DIR / "data" / "cleaned" / "prototype_2018_k_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["appid", "name", "players", "total_reviews", "k", "release_date", "initial_price_usd"]].to_csv(
        out, index=False, encoding="utf-8-sig")
    for key, v in summary.items():
        print(f"  {key:16s} {v:.4f}" if isinstance(v, float) else f"  {key:16s} {v}")

    # ---------- 图 1：k 和 ln k 的分布 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(k[k < np.percentile(k, 97)], bins=36, color=COLORS[0], edgecolor="white", linewidth=1.5)
    ax.axvline(np.median(k), color=INK["primary"], lw=1.5)
    ax.axvline(k.mean(), color=INK["secondary"], lw=1.5, ls="--")
    ax.text(np.median(k), ax.get_ylim()[1] * 0.92, f" 中位数 {np.median(k):.1f}", color=INK["primary"], fontsize=10)
    ax.text(k.mean(), ax.get_ylim()[1] * 0.80, f" 均值 {k.mean():.1f}", color=INK["secondary"], fontsize=10)
    style_axes(ax, "(a) k 的分布（截去最大的 3% 便于观察）", "k = 玩家数 / 评测数", "游戏数")
    ax = axes[1]
    ax.hist(lk, bins=30, color=COLORS[0], edgecolor="white", linewidth=1.5)
    style_axes(ax, f"(b) ln k 的分布：均值 {lk.mean():.2f}，标准差 {lk.std(ddof=1):.2f}", "ln k", "游戏数")
    fig.suptitle(f"原型样本 {len(df)} 款：k 明显右偏，取对数后接近对称", fontsize=12, color=INK["primary"])
    save(fig, "fig01_prototype_k_dist.png")

    # ---------- 图 2：对数坐标散点 ----------
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(lr, lp, s=22, color=COLORS[0], alpha=0.75, edgecolor="white", linewidth=0.6)
    xs = np.linspace(lr.min(), lr.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=COLORS[1], lw=2)
    ax.text(0.03, 0.95, f"Pearson r = {r:.3f}\nlog₁₀玩家数 = {intercept:.2f} + {slope:.2f}·log₁₀评测数",
            transform=ax.transAxes, va="top", fontsize=10, color=INK["primary"])
    # 原型阶段最重要的发现：散点下沿被抽样门槛"切平"，左侧评测少的游戏只有 k 大的才进得来
    ax.axhline(np.log10(20000), color=INK["secondary"], lw=1.2, ls="--")
    ax.text(lr.max(), np.log10(20000) - 0.06, "抽样门槛：玩家数 2 万", ha="right", va="top",
            fontsize=9, color=INK["secondary"])
    style_axes(ax, "对数坐标下近似直线，但下沿被抽样门槛截平", "log₁₀ 评测数（截至 2018-07-01）",
               "log₁₀ 玩家数（2018-07-01 泄露）")
    save(fig, "fig02_prototype_log_scatter.png")


if __name__ == "__main__":
    main()
