# -*- coding: utf-8 -*-
"""
单款游戏实测 (src/case_study.py)：给一个 appid 和几个日期，采集截至每个日期的数据，用系统里的全部方法估算 Steam 销量。

用法：
    python src/case_study.py --appid 3321460 --dates 2026-03-20,2026-04-15,2026-06-10,2026-10-14 --name crimson_desert \
        --reference "2026-04-15:2000000:2500000:全平台500万×PC约一半,2026-06-10:2400000:3000000:全平台600万×PC约一半"

和建模时的区别：
- 被估的游戏不在真值集里，所以近年校准系数、融合权重都直接用全部 107 款真值拟合（不需要留一法）；
- 2018 模型（M2 回归、M3 分位数、M4 树模型）、信号 B 的人均小时换算，都和 models.py 的全量拟合完全一样。
输出：data/cleaned/case_<name>.csv 与 figures/case_<name>.png
"""
import sys
import math
import sqlite3
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import crawler  # noqa: E402
import models  # noqa: E402
from etl_pipeline import main_genre, charts_hours_before, load_policy  # noqa: E402
from plot_style import plt, COLORS, INK, style_axes, save  # noqa: E402

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"


def collect(appid, cutoff):
    """采集截至 cutoff 当天的评测汇总、中文评测、Steam 内购买评测、100 条评测时长样本，以及商店和 SteamCharts。"""
    end_ts = crawler.end_of_day_ts(cutoff)
    d = crawler.fetch_appdetails(appid)
    ok, reason = crawler.is_paid_game(d)
    if not ok:
        raise SystemExit(f"{appid} 不是付费游戏：{reason}")
    rev_all = crawler.fetch_reviews(appid, end_ts, sample_size=100)
    rev_sc = crawler.fetch_reviews(appid, end_ts, languages=("schinese",))
    rev_steam = crawler.fetch_reviews(appid, end_ts, purchase_type=0)
    charts = crawler.fetch_steamcharts(appid)
    n, med, _ = crawler.summarize_samples(rev_all["samples"])
    return d["data"], rev_all, rev_sc, rev_steam, charts, med


def build_row(appid, cutoff, policy):
    data, rev_all, rev_sc, rev_steam, charts, pt_med = collect(appid, cutoff)
    rel = crawler.parse_release_date(data.get("release_date", {}).get("date", ""))
    tot = rev_all["total_reviews"]
    price = data["price_overview"]["initial"] / 100.0
    month_rows = [(r["month"], r["avg_players"]) for r in sorted(charts["rows"], key=lambda r: r["month"])]
    hours = charts_hours_before(month_rows, cutoff.isoformat(), rel.isoformat()) if month_rows else 0.0
    policy_date = dt.date.fromisoformat(policy["policy_month"] + "-01")
    before = 0 if rel >= policy_date else None  # 本脚本只处理政策后发售的游戏；老游戏需要再查政策前评测数
    if before is None:
        before = crawler.fetch_reviews(appid, policy["policy_ts"])["total_reviews"]
    genres = "|".join(g["description"] for g in data.get("genres", []))
    return {
        "appid": appid, "name": data.get("name"), "cutoff_date": cutoff.isoformat(), "release_date": rel.isoformat(),
        "game_age_years": max((cutoff - rel).days / 365.25, 1 / 12), "genre": main_genre(genres),
        "is_indie": int("Indie" in genres.split("|")), "price_usd": price, "log_price": math.log(price),
        "total_reviews": tot, "positive_rate": rev_all["total_positive"] / tot, "log_reviews": math.log(tot),
        "schinese_ratio": rev_sc["total_reviews"] / tot, "steam_purchase_ratio": rev_steam["total_reviews"] / tot,
        "playtime_review_median_hrs": pt_med, "reviews_before_policy": before, "reviews_after_policy": tot - before,
        "charts_total_hours": hours, "charts_months": len([m for m, _ in month_rows if m < cutoff.isoformat()[:7]]),
    }


def fit_everything(conn):
    """和 models.py 相同的全量拟合；近年校准与融合权重用全部真值。"""
    df = pd.read_sql_query("SELECT * FROM dwd_game_sample ORDER BY sample_id", conn)
    tr = df[(df["cohort"] == "train_2018") & (df["model_eligible"] == 1)].reset_index(drop=True)
    te = df[df["cohort"] == "truth"].reset_index(drop=True)
    pt_fill = float(tr["playtime_review_median_hrs"].median())
    X_tr, _ = models.build_features(tr, pt_fill)
    X_te, _ = models.build_features(te, pt_fill)
    y_tr = tr["log_k"].values
    ols = models.fit_ols(X_tr, y_tr)["beta"]
    q = {tau: models.fit_quantile(X_tr, y_tr, tau) for tau in (0.1, 0.5, 0.9)}
    tree = models.TreeEnsemble(seed=99).fit(X_tr[:, 1:], y_tr)

    trb = tr[(tr["charts_coverage"] == 1) & tr["playtime_review_median_hrs"].notna()]
    Zb = np.column_stack([np.ones(len(trb)), np.log(trb["playtime_review_median_hrs"].clip(lower=0.1))])
    ab, _, _, _ = np.linalg.lstsq(Zb, np.log(trb["charts_total_hours"] / trb["truth_value"]), rcond=None)

    y_lk = np.log(te["truth_value"].values / te["total_reviews"].values)
    Z = models.recal_design(te, X_te)
    w2 = models.fit_recal(Z, X_te @ ols, y_lk)
    off4 = tree.predict(X_te[:, 1:])
    w4 = models.fit_recal(Z, off4, y_lk)
    has_b = ((te["charts_coverage"] == 1) & te["playtime_review_median_hrs"].notna()).values
    ln_a = np.log(te["total_reviews"].values) + off4 + Z @ w4
    pt = te["playtime_review_median_hrs"].clip(lower=0.1).values
    ln_b = np.log(te["charts_total_hours"].values / np.exp(ab[0] + ab[1] * np.log(pt)))
    F = np.column_stack([np.ones(has_b.sum()), ln_a[has_b], ln_b[has_b]])
    wf, _, _, _ = np.linalg.lstsq(F, np.log(te["truth_value"].values[has_b]), rcond=None)
    return dict(pt_fill=pt_fill, ols=ols, q=q, tree=tree, ab=ab, w2=w2, w4=w4, wf=wf)


def estimate(row_df, fit):
    X, _ = models.build_features(row_df, fit["pt_fill"])
    Z = models.recal_design(row_df, X)
    n = row_df["total_reviews"].values.astype(float)
    out = pd.DataFrame({"cutoff_date": row_df["cutoff_date"], "total_reviews": n.astype(int)})
    year = pd.to_datetime(row_df["release_date"]).dt.year.values
    out["M0"] = n * 30
    out["M1"] = n * np.array([models.m1_k(y) for y in year])
    out["M2"] = n * np.exp(X @ fit["ols"] + Z @ fit["w2"])
    out["M3_P10"] = out["M2"] * np.exp(X @ fit["q"][0.1] - X @ fit["q"][0.5])
    out["M3_P90"] = out["M2"] * np.exp(X @ fit["q"][0.9] - X @ fit["q"][0.5])
    ln_a = np.log(n) + fit["tree"].predict(X[:, 1:]) + Z @ fit["w4"]
    out["M4"] = np.exp(ln_a)
    pt = row_df["playtime_review_median_hrs"].clip(lower=0.1).values
    out["Signal_B"] = row_df["charts_total_hours"].values / np.exp(fit["ab"][0] + fit["ab"][1] * np.log(pt))
    has_b = row_df["charts_total_hours"].values > 0
    fused = fit["wf"][0] + fit["wf"][1] * ln_a + fit["wf"][2] * np.log(np.where(has_b, out["Signal_B"], 1.0))
    out["Fusion_AB"] = np.where(has_b, np.exp(fused), out["M4"])
    out.loc[~has_b, "Signal_B"] = np.nan
    return out


def plot_case(result, name, reference):
    """估算随时间的变化；reference = [(日期, 下限, 上限, 说明)]，画成带区间的黑点。"""
    x = pd.to_datetime(result["cutoff_date"])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, result["M0"] / 1e4, color=INK["muted"], lw=1.2, ls="--", marker="o", ms=4, label="M0 固定 30 倍")
    for col, color, label in [("M4", COLORS[0], "M4 树模型 + 近年校准"), ("Signal_B", COLORS[1], "信号 B 在线×时长"),
                              ("Fusion_AB", COLORS[2], "融合 A+B")]:
        ax.plot(x, result[col] / 1e4, color=color, lw=2, marker="o", ms=6, label=label)
    ax.fill_between(x, result["M3_P10"] / 1e4, result["M3_P90"] / 1e4, color=COLORS[0], alpha=0.10, lw=0,
                    label="M3 80% 区间")
    for d, lo, hi, note in reference:
        dd = pd.Timestamp(d)
        ax.errorbar([dd], [(lo + hi) / 2 / 1e4], yerr=[[(hi - lo) / 2 / 1e4]], fmt="s", color=INK["primary"],
                    ms=6, capsize=4, lw=1.5)
        ax.text(dd, hi / 1e4, f"  {note}", fontsize=9, color=INK["primary"], va="bottom")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax, f"{result['name'].iloc[0]}：各截止日的 Steam 销量估算（黑色方块 = 由官方数字推出的参照区间）",
               "截止日", "万份")
    save(fig, f"case_{name}.png")


def main():
    args = sys.argv[1:]
    appid = int(args[args.index("--appid") + 1])
    dates = [dt.date.fromisoformat(s) for s in args[args.index("--dates") + 1].split(",")]
    name = args[args.index("--name") + 1] if "--name" in args else str(appid)
    policy = load_policy()
    rows = pd.DataFrame([build_row(appid, d, policy) for d in dates])
    conn = sqlite3.connect(str(DB_PATH))
    fit = fit_everything(conn)
    conn.close()
    est = estimate(rows, fit)
    result = rows.merge(est, on=["cutoff_date", "total_reviews"])
    out = BASE_DIR / "data" / "cleaned" / f"case_{name}.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    show = ["cutoff_date", "total_reviews", "positive_rate", "steam_purchase_ratio", "schinese_ratio",
            "playtime_review_median_hrs", "charts_months", "M0", "M1", "M2", "M3_P10", "M3_P90", "M4", "Signal_B", "Fusion_AB"]
    print(result[show].round(3).to_string(index=False))
    print(f"-> {out.relative_to(BASE_DIR)}")
    # --reference 2026-04-15:2000000:2500000:说明,...   上下限之间是参照区间
    reference = []
    if "--reference" in args:
        for item in args[args.index("--reference") + 1].split(","):
            d, lo, hi, note = item.split(":", 3)
            reference.append((d, float(lo), float(hi), note))
    plot_case(result, name, reference)
    return result


if __name__ == "__main__":
    main()
