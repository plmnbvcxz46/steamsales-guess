# -*- coding: utf-8 -*-
"""
报告用统计图 (src/visualize_all.py)：FineBI 不擅长的统计图（拟合线、误差带、对角线散点等）用 Python 画。
全部从 data/steam_sales_dw.db 读数，图上的每个数字都来自数据库，不手写。

fig03 数据概览            fig04 评测提示政策：月度评测指数 + 各发售年份 k
fig05 影响 k 的因素        fig06 真值集上 估算 vs 真实
fig07 回归系数与树模型重要性  fig08 ±30% 命中率对标 Gamalytic
fig09 真实价格拉链（档案馆）  fig10 系统架构
"""
import re
import sqlite3
import pathlib

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from plot_style import plt, COLORS, INK, style_axes, save

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"


def q(conn, sql):
    return pd.read_sql_query(sql, conn)


def fig03_overview(conn):
    d = q(conn, "SELECT cohort, model_eligible, total_reviews, release_year, price_usd FROM dwd_game_sample")
    tr = d[(d["cohort"] == "train_2018") & (d["model_eligible"] == 1)]
    te = d[d["cohort"] == "truth"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 6.5, 33)
    ax.hist(np.log10(tr["total_reviews"]), bins=bins, color=COLORS[0], alpha=0.85, edgecolor="white", label=f"2018 训练集 ({len(tr)})")
    ax.hist(np.log10(te["total_reviews"]), bins=bins, color=COLORS[1], alpha=0.85, edgecolor="white", label=f"真值集 ({len(te)})")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, "(a) 评测数（对数）", "lg 截止日评测数", "样本数")
    ax = axes[1]
    years = np.arange(min(d["release_year"].min(), 2004), 2024)
    c1 = tr["release_year"].value_counts().reindex(years, fill_value=0)
    c2 = te["release_year"].value_counts().reindex(years, fill_value=0)
    ax.bar(years - 0.2, c1.values, width=0.4, color=COLORS[0], label="2018 训练集")
    ax.bar(years + 0.2, c2.values, width=0.4, color=COLORS[1], label="真值集")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, "(b) 发售年份", "发售年份", "样本数")
    ax = axes[2]
    pb = np.arange(0, 65, 2.5)
    ax.hist(tr["price_usd"].clip(upper=62), bins=pb, color=COLORS[0], alpha=0.85, edgecolor="white", label="2018 训练集")
    ax.hist(te["price_usd"].clip(upper=62), bins=pb, color=COLORS[1], alpha=0.85, edgecolor="white", label="真值集")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, "(c) 美区标价（>$60 归入最右一档）", "美元", "样本数")
    save(fig, "fig03_data_overview_3panels.png")


def fig04_policy(conn):
    m = q(conn, "SELECT * FROM ads_review_monthly_index ORDER BY month")
    yk = q(conn, "SELECT * FROM vw_finebi_k_median_by_year ORDER BY release_year")
    te = q(conn, "SELECT release_year, k_ratio, era_group FROM dwd_game_sample WHERE cohort = 'truth'")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    x = np.arange(len(m))
    ax.fill_between(x, m["q25"], m["q75"], color=COLORS[0], alpha=0.18, lw=0, label="四分位范围")
    ax.plot(x, m["median_index"], color=COLORS[0], lw=2, marker="o", ms=4, label="40 款老游戏的中位数")
    pm = int(np.where(m["is_policy_month"] == 1)[0][0])
    ax.axvline(pm, color=INK["primary"], lw=1.2, ls="--")
    ax.text(pm + 0.3, ax.get_ylim()[1] * 0.92, f"{m['month'][pm]}\n持续跳升起点", fontsize=9, color=INK["primary"], va="top")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(m["month"][::3], rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax, "(a) 老游戏每月新增评测 ÷ 自身 2018-07~2019-06 月中位数", None, "评测指数（基线 = 1）")
    ax = axes[1]
    tr = yk[(yk["cohort"] == "train_2018") & (yk["game_count"] >= 10)]
    ax.plot(tr["release_year"], tr["median_k"], color=COLORS[0], lw=2, marker="o", ms=5, label="2018 训练集（真值=玩家数）")
    tb = te.copy()
    tb["bucket"] = np.where(tb["release_year"] < 2019, "2019 年以前", np.where(tb["release_year"] < 2020, "2019", tb["release_year"].astype(str)))
    grp = te.groupby("release_year")["k_ratio"].agg(["median", "count"]).reset_index()
    grp = grp[grp["count"] >= 3]
    ax.plot(grp["release_year"], grp["median"], color=COLORS[1], lw=2, marker="s", ms=5, label="真值集（真值=销量，≥3 款的年份）")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, "(b) 各发售年份 k 的中位数（对数坐标）", "发售年份", "k 中位数")
    save(fig, "fig04_k_by_year_2019_cliff.png")


def fig05_factors(conn):
    f = q(conn, "SELECT * FROM vw_finebi_k_median_by_factor")
    d = q(conn, "SELECT steam_purchase_ratio, k_ratio FROM dwd_game_sample WHERE cohort='train_2018' AND model_eligible=1")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    def bars(ax, factor, title, horizontal=False):
        s = f[f["factor"] == factor].sort_values("factor_value")
        labels = [f"{v}\n(n={n})" if not horizontal else f"{v} (n={n})" for v, n in zip(s["factor_value"], s["game_count"])]
        if horizontal:
            s = s.sort_values("median_k")
            labels = [f"{v} (n={n})" for v, n in zip(s["factor_value"], s["game_count"])]
            ax.barh(labels, s["median_k"], color=COLORS[0], height=0.6)
            for i, v in enumerate(s["median_k"]):
                ax.text(v, i, f" {v:.0f}", va="center", fontsize=9, color=INK["secondary"])
            style_axes(ax, title, "k 中位数", None)
        else:
            ax.bar(labels, s["median_k"], color=COLORS[0], width=0.6)
            for i, v in enumerate(s["median_k"]):
                ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK["secondary"])
            style_axes(ax, title, None, "k 中位数")

    bars(axes[0, 0], "好评率档", "(a) 好评率档")
    bars(axes[0, 1], "价格档", "(b) 价格档")
    ax = axes[1, 0]
    d["bin"] = pd.qcut(d["steam_purchase_ratio"], 8, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=("steam_purchase_ratio", "median"), k=("k_ratio", "median")).reset_index()
    ax.scatter(d["steam_purchase_ratio"], d["k_ratio"], s=8, color=COLORS[0], alpha=0.25, lw=0)
    ax.plot(g["x"], g["k"], color=COLORS[1], lw=2, marker="o", ms=6, label="分 8 组的中位数")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax, "(c) Steam 内购买评测占比（越低 = 激活码/赠送玩家越多）", "Steam 内购买评测占比", "k（对数）")
    bars(axes[1, 1], "主类型", "(d) 主类型", horizontal=True)
    fig.suptitle("2018 训练集：k 与各因素（分组中位数由 SQL 窗口函数视图算出）", fontsize=12, color=INK["primary"])
    save(fig, "fig05_k_factors_4panels.png")


def fig06_pred_vs_true(conn):
    f = q(conn, "SELECT method_code, true_value, estimated_sales, estimated_low_bound, estimated_high_bound FROM fact_sales_estimation WHERE cohort='truth'")
    ev = q(conn, "SELECT method_code, sample_size, hit_rate_30pct FROM ads_model_eval_summary WHERE eval_cohort='真值集 全部'").set_index("method_code")
    panels = [("M1", "M1 文献 k"), ("M4", "M4 树模型 + 近年校准"), ("Signal_B", "信号 B 在线×时长"), ("Fusion_AB", "融合 A+B")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    lo, hi = 2.5, 7.8
    for ax, (code, title) in zip(axes.flat, panels):
        s = f[f["method_code"] == code]
        xs = np.linspace(lo, hi, 10)
        ax.fill_between(xs, xs + np.log10(0.7), xs + np.log10(1.3), color=INK["grid"], alpha=0.9, lw=0, label="±30% 范围")
        ax.plot(xs, xs, color=INK["secondary"], lw=1)
        yerr = np.vstack([np.log10(s["estimated_sales"]) - np.log10(s["estimated_low_bound"]),
                          np.log10(s["estimated_high_bound"]) - np.log10(s["estimated_sales"])])
        if code == "M4":
            ax.errorbar(np.log10(s["true_value"]), np.log10(s["estimated_sales"]), yerr=yerr, fmt="none",
                        ecolor=COLORS[0], alpha=0.35, lw=1)
        ax.scatter(np.log10(s["true_value"]), np.log10(s["estimated_sales"]), s=22, color=COLORS[0], edgecolor="white", lw=0.6, zorder=3)
        n, hit = int(ev.loc[code, "sample_size"]), ev.loc[code, "hit_rate_30pct"]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        style_axes(ax, f"{title}：±30% 命中 {hit:.1f}%（n={n}）", "lg 公开销量", "lg 估算销量")
    axes[0, 1].text(0.03, 0.97, "竖线：交叉验证残差的 10%~90% 区间", transform=axes[0, 1].transAxes, va="top", fontsize=9, color=INK["secondary"])
    fig.suptitle("真值集：估算 vs 公开销量（对角线 = 估得正好，灰带 = 误差 30% 以内）", fontsize=12, color=INK["primary"])
    save(fig, "fig06_pred_vs_true_intervals.png")


def fig07_features(conn):
    fi = q(conn, "SELECT * FROM ads_feature_importance WHERE feature_code <> 'intercept'")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ax = axes[0]
    s = fi.iloc[::-1]
    y = np.arange(len(s))
    ax.errorbar(s["m2_ols_coef"], y, xerr=1.96 * s["m2_std_err"], fmt="o", color=COLORS[0], ecolor=COLORS[0], ms=5, capsize=0)
    ax.axvline(0, color=INK["secondary"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(s["feature_cn_name"])
    style_axes(ax, "(a) M2 回归系数与 95% 置信区间（因变量 ln k）", "系数", None)
    ax = axes[1]
    imp = fi.assign(avg=(fi["m4_rf_importance_pct"] + fi["m4_gbdt_importance_pct"]) / 2).sort_values("avg")
    ax.barh(imp["feature_cn_name"], imp["avg"], color=COLORS[0], height=0.6)
    for i, v in enumerate(imp["avg"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=9, color=INK["secondary"])
    style_axes(ax, "(b) M4 特征重要性（随机森林与梯度提升树的平均）", "分裂带来的误差下降占比", None)
    save(fig, "fig07_feature_importance_m2_m4.png")


def fig08_benchmark(conn):
    ev = q(conn, "SELECT * FROM ads_model_eval_summary")
    ours = ev[(ev["eval_cohort"] == "真值集 全部")]
    gam = ev[ev["eval_cohort"] == "Gamalytic 公布(约120款)"]
    rows = [(f"本项目 {r.method_code}（n={r.sample_size}）", r.hit_rate_30pct, 0) for r in ours.itertuples() if r.method_code != "Gamalytic_same"]
    same = ours[ours["method_code"] == "Gamalytic_same"]
    rows += [(f"Gamalytic 自己的估算，同一批游戏（n={int(same.sample_size.iloc[0])}）", float(same.hit_rate_30pct.iloc[0]), 1)]
    rows += [(f"{r.method_name}（原文）", r.hit_rate_30pct, 1) for r in gam.itertuples()]
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [COLORS[0] if r[2] == 0 else COLORS[1] for r in rows]
    ax.barh(labels, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=9, color=INK["secondary"])
    ax.set_xlim(0, 105)
    ax.legend(handles=[mpatches.Patch(color=COLORS[0], label="本项目（真值集，留一法）"),
                       mpatches.Patch(color=COLORS[1], label="Gamalytic")], frameon=False, fontsize=9, loc="lower right")
    style_axes(ax, "估算落在公开销量 ±30% 以内的比例", "±30% 命中率 (%)", None)
    save(fig, "fig08_method_benchmark_gamalytic.png")


def fig09_price(conn):
    p = q(conn, "SELECT * FROM dim_price_scd2 ORDER BY appid, version_no")
    if p.empty:
        print("  (没有价格拉链数据，跳过 fig09)")
        return
    counts = p.groupby("appid").size().sort_values(ascending=False)
    picks = counts.index[:3].tolist()
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.2), squeeze=False)
    for ax, appid in zip(axes[0], picks):
        s = p[p["appid"] == appid].copy()
        s["start"] = pd.to_datetime(s["effective_start"])
        s["end"] = pd.to_datetime(s["effective_end"].replace("9999-12-31", pd.Timestamp.today().strftime("%Y-%m-%d")))
        for r in s.itertuples():
            ax.hlines(r.final_price_usd, r.start, r.end, color=COLORS[0], lw=2.5)
            if r.discount_percent > 0:
                ax.hlines(r.base_price_usd, r.start, r.end, color=INK["muted"], lw=1, ls="--")
            ax.plot([r.start], [r.final_price_usd], "o", color=COLORS[0], ms=5)
            ax.text(r.start, r.final_price_usd, f" v{r.version_no}", fontsize=8, color=INK["secondary"], va="bottom")
        ax.set_ylim(0, max(s["base_price_usd"].max() * 1.25, 1))
        style_axes(ax, f"{s['name'].iloc[0]}（{len(s)} 个版本）", None, "美元")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("价格拉链表 dim_price_scd2：实线 = 观测到的成交价，虚线 = 打折时的原价（数据：互联网档案馆存档 + 本次采集）",
                 fontsize=11, color=INK["primary"])
    save(fig, "fig09_price_scd2_review_timeline.png")


def fig10_architecture(conn):
    rep = (BASE_DIR / "data" / "cleaned" / "explain_comparison_report.txt").read_text(encoding="utf-8")
    speed = re.search(r"提速倍数: ([\d.]+)x", rep)
    n = dict(conn.execute("SELECT cohort, COUNT(*) FROM dwd_game_sample WHERE model_eligible=1 GROUP BY cohort").fetchall())
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    layers = [
        ("1 数据采集", "Ars Technica 2018 泄露表 · Steam appdetails · GetAppReviews(input_json, 按日期截断) · SteamCharts 月度在线\n"
                     "Gamalytic 测试表(118 条公开销量及出处) · 互联网档案馆商店页存档(价格历史) · 每小时快照"),
        ("2 ETL", "Python：请求缓存与断点续传、分批事务 + etl_checkpoint 续跑、清洗规则写 etl_clean_error_log、\n"
                  "SCD2 价格拉链、按 fetched_at 水位线增量抽取快照；Kettle：clean_steam_games.ktr / scd2_price_history.ktr"),
        ("3 数据仓库", "原始层 ods_*  →  明细层 dwd_game_sample + 维度 dim_game / dim_genre / dim_release_date / dim_estimation_method / dim_price_scd2\n"
                      "→  结果层 fact_sales_estimation（样本 × 方法）、fact_snapshot、ads_*；SQLite 构建，脚本可导入 MySQL 8.0"),
        ("4 分析建模", f"SQL：M0、M1、窗口函数求 k 中位数；Python：M2 回归、M3 分位数、M4 树模型、信号 B、融合\n"
                      f"2018 训练 {n.get('train_2018', 0)} 款五折交叉验证；真值 {n.get('truth', 0)} 款，政策系数与融合权重一律留一法"),
        ("5 视图与索引", "8 个 FineBI 只读视图；复合索引 + EXPLAIN 对比" + (f"（本机实测提速 {speed.group(1)}x）" if speed else "")),
        ("6 展示", "FineBI 大屏（估算方法切片器整屏联动）· 报告统计图 · 与 Gamalytic 同口径对标"),
    ]
    h, gap, top = 13.2, 3.0, 97
    for i, (title, body) in enumerate(layers):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(mpatches.FancyBboxPatch((3, y), 94, h, boxstyle="round,pad=0.4",
                                             facecolor="#f4f7fc", edgecolor=COLORS[0], linewidth=1.4))
        ax.text(5, y + h - 3.0, title, fontsize=11, fontweight="bold", color=INK["primary"])
        ax.text(5, y + 1.6, body, fontsize=8.8, color=INK["secondary"], linespacing=1.4)
        if i < len(layers) - 1:
            ax.annotate("", xy=(50, y - gap + 0.3), xytext=(50, y - 0.3),
                        arrowprops=dict(arrowstyle="->", color=INK["muted"], lw=1.5))
    save(fig, "fig10_system_architecture_and_er.png")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    for fn in [fig03_overview, fig04_policy, fig05_factors, fig06_pred_vs_true, fig07_features,
               fig08_benchmark, fig09_price, fig10_architecture]:
        fn(conn)
    conn.close()


if __name__ == "__main__":
    main()
