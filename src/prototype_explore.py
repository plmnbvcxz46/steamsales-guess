# -*- coding: utf-8 -*-
"""
阶段二：原型验证脚本 (src/prototype_explore.py)
遵循 CRISP-DM 数据理解与原型法：
先把 2018-07-01 泄露 CSV 与评测数据对齐，过滤免费游戏、Demo 和非游戏软件，
算出每款买断制游戏的实测 Boxleiter 系数 k = 2018玩家数 / 2018评测数，
输出两张原型期探索图：
1. figures/fig01_prototype_k_dist.png（实测 k 分布直方图 + 核密度与中位数/均值对比）
2. figures/fig02_prototype_log_scatter.png（log10(评测数) vs log10(玩家数) 散点与拟合趋势）
"""
import json
import math
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = pathlib.Path("/tmp/steamsales-guess")


def run_prototype():
    raw_csv = BASE_DIR / "data/raw/games_achievements_players_2018-07-01.csv"
    rev_csv = BASE_DIR / "data/raw/steam_reviews_2018_aligned.csv"
    app_json = BASE_DIR / "data/raw/steam_appdetails_2018_aligned.json"

    df_players = pd.read_csv(raw_csv)
    df_reviews = pd.read_csv(rev_csv)
    with open(app_json, "r", encoding="utf-8") as f:
        app_details = json.load(f)

    # 解析商店 JSON 字段
    meta_rows = []
    for appid_str, payload in app_details.items():
        if not payload.get("success"):
            continue
        d = payload["data"]
        price_obj = d.get("price_overview")
        price_usd = (price_obj["initial"] / 100.0) if price_obj else 0.0
        rel_str = d.get("release_date", {}).get("date", "2015")
        rel_year = int(rel_str.split(",")[-1].strip()) if "," in rel_str else 2015
        genres = d.get("genres", [])
        main_genre = genres[0]["description"] if genres else "Unknown"
        meta_rows.append({
            "appid": int(appid_str),
            "app_type": d.get("type", "game"),
            "is_free": 1 if d.get("is_free") else 0,
            "price_usd": price_usd,
            "release_year": rel_year,
            "genre": main_genre,
            "dirty_note": d.get("dirty_note", "")
        })
    df_meta = pd.DataFrame(meta_rows)

    # 关联三份数据
    df = df_players.merge(df_meta, on="appid", how="inner").merge(
        df_reviews.drop(columns=["name"]), on="appid", how="inner"
    )

    print(f"[原型探索] 原始关联总记录数: {len(df)}")
    print(f"  - 其中免费游戏 (is_free=1): {(df['is_free'] == 1).sum()} 款")
    print(f"  - 其中非游戏软件/Demo (app_type!='game'): {(df['app_type'] != 'game').sum()} 款")
    print(f"  - 其中价格或评测数异常记录: {((df['price_usd'] <= 0) & (df['is_free'] == 0) | (df['total_reviews_2018'] <= 0)).sum()} 款")

    # 过滤出正常买断制正式版游戏
    df_clean = df[
        (df["is_free"] == 0)
        & (df["app_type"] == "game")
        & (df["price_usd"] > 0)
        & (df["total_reviews_2018"] > 0)
    ].copy()

    # 计算实测 Boxleiter 倍数 k
    df_clean["k_ratio"] = df_clean["players_2018_07_01"] / df_clean["total_reviews_2018"]
    df_clean["log10_reviews"] = np.log10(df_clean["total_reviews_2018"])
    df_clean["log10_players"] = np.log10(df_clean["players_2018_07_01"])

    # 保存原型验证汇总表
    out_csv = BASE_DIR / "data/cleaned/prototype_2018_k_summary.csv"
    df_clean.to_csv(out_csv, index=False, encoding="utf-8")

    k_mean = df_clean["k_ratio"].mean()
    k_median = df_clean["k_ratio"].median()
    k_q25 = df_clean["k_ratio"].quantile(0.25)
    k_q75 = df_clean["k_ratio"].quantile(0.75)
    corr_log = np.corrcoef(df_clean["log10_reviews"], df_clean["log10_players"])[0, 1]

    print(f"[原型探索] 清洗后买断制游戏样本数: {len(df_clean)} 款")
    print(f"[原型探索] 实测 k 均值: {k_mean:.2f}, 中位数: {k_median:.2f}, 四分位距 (IQR): [{k_q25:.2f}, {k_q75:.2f}]")
    print(f"[原型探索] log10(评测数) 与 log10(玩家数) 皮尔逊相关系数 r = {corr_log:.4f}")

    # 图1：实测 k 的分布图（为什么同时看原始分布和对数分布：因为 k 呈右偏态，少数低评测高销量游戏拉高均值，中位数更稳健）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    ax1, ax2 = axes

    ax1.hist(df_clean["k_ratio"], bins=24, color="#2B5C8F", edgecolor="white", alpha=0.88)
    ax1.axvline(k_median, color="#D9534F", linestyle="--", linewidth=2, label=f"中位数 = {k_median:.1f}")
    ax1.axvline(k_mean, color="#F0AD4E", linestyle="-.", linewidth=2, label=f"均值 = {k_mean:.1f}")
    ax1.axvline(30.0, color="#2E8B57", linestyle=":", linewidth=2, label="传统固定倍数 M0 = 30")
    ax1.set_title("2018年买断制游戏实测 k 值原始分布（右偏特征明显）", fontsize=12, fontweight="bold")
    ax1.set_xlabel("实测系数 k = 2018玩家数 / 2018评测数", fontsize=10.5)
    ax1.set_ylabel("游戏数量（款）", fontsize=10.5)
    ax1.legend(frameon=True, fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    log_k = np.log(df_clean["k_ratio"])
    ax2.hist(log_k, bins=22, color="#388E3C", edgecolor="white", alpha=0.85, density=True)
    xs = np.linspace(log_k.min() - 0.2, log_k.max() + 0.2, 200)
    mu, sigma = log_k.mean(), log_k.std()
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    ax2.plot(xs, pdf, color="#1B365D", linewidth=2.2, label=f"正态拟合 N({mu:.2f}, {sigma:.2f}²)")
    ax2.set_title("取对数后 ln(k) 分布（近似对称正态，支持对数线性回归 M2）", fontsize=12, fontweight="bold")
    ax2.set_xlabel("ln(k)", fontsize=10.5)
    ax2.set_ylabel("概率密度", fontsize=10.5)
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle(f"原型验证阶段图1：Steam 买断制游戏评测-玩家倍数 k 分布特征 (N = {len(df_clean)})", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig1_path = BASE_DIR / "figures/fig01_prototype_k_dist.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    # 图2：对数散点图（log10(评测数) vs log10(玩家数)）
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=180)
    sc = ax.scatter(
        df_clean["log10_reviews"],
        df_clean["log10_players"],
        c=df_clean["positive_rate_2018"] * 100,
        cmap="viridis",
        s=46,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("2018年好评率 (%)", fontsize=10.5)

    # 拟合最小二乘直线与固定倍数参考线
    slope, intercept = np.polyfit(df_clean["log10_reviews"], df_clean["log10_players"], 1)
    x_line = np.linspace(df_clean["log10_reviews"].min() - 0.1, df_clean["log10_reviews"].max() + 0.1, 100)
    ax.plot(x_line, slope * x_line + intercept, color="#D9534F", linewidth=2.2,
            label=f"OLS 拟合线: log10(Y) = {slope:.3f}·log10(X) + {intercept:.3f} (r={corr_log:.3f})")
    ax.plot(x_line, x_line + math.log10(k_median), color="#1B365D", linestyle="--", linewidth=1.8,
            label=f"2018中位数倍数参考线 (k={k_median:.1f})")
    ax.plot(x_line, x_line + math.log10(30.0), color="#777777", linestyle=":", linewidth=1.8,
            label="传统经验倍数参考线 (k=30)")

    # 标注几款代表性游戏
    highlight_names = ["PLAYERUNKNOWN'S BATTLEGROUNDS", "The Witcher 3: Wild Hunt", "Stardew Valley", "Hollow Knight", "No Man's Sky"]
    for _, r in df_clean[df_clean["name"].isin(highlight_names)].iterrows():
        short_name = "PUBG" if "BATTLEGROUNDS" in r["name"] else r["name"]
        ax.annotate(
            f"{short_name} (k={r['k_ratio']:.0f})",
            xy=(r["log10_reviews"], r["log10_players"]),
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#222222"
        )

    ax.set_title("原型验证阶段图2：Steam 评测总数与真实玩家数的对数线性关系（2018-07-01快照对齐）", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("log10(截至2018-07-01累计评测总数)", fontsize=11)
    ax.set_ylabel("log10(2018-07-01成就接口泄露真实玩家数)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig2_path = BASE_DIR / "figures/fig02_prototype_log_scatter.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    print(f"[原型探索] 已生成探索图表: {fig1_path.name}, {fig2_path.name}")


if __name__ == "__main__":
    run_prototype()
