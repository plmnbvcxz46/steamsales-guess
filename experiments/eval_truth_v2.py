# -*- coding: utf-8 -*-
"""
独立检验：用现有系统（2018 模型 + 只在旧的 107 款真值上校准）去估新真值集，看准确率。

新真值集（data/raw/official_sales_truth_v2.csv）和旧的 107 款没有一款重复，模型拟合时完全没见过，
所以这里不需要留一法——它本身就是一份全新的测试集。
清洗规则直接复用 etl_pipeline.clean_one（免费游戏、发售晚于公布日、评测数为 0 等一律同样处理），
估算直接复用 case_study.fit_everything / estimate（和《红色沙漠》实测是同一套代码）。

用法：python experiments/eval_truth_v2.py
输出：data/cleaned/eval_truth_v2_rows.csv（逐条估算）、data/cleaned/eval_truth_v2_summary.csv（分组指标）
"""
import sys
import math
import sqlite3
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
sys.path.insert(0, str(BASE / "src"))
import etl_pipeline as E  # noqa: E402
import case_study as CS  # noqa: E402

METHODS = ["M0", "M1", "M2", "M4", "Signal_B", "Fusion_AB"]
MIN_REVIEWS = 20  # 评测太少时评测数本身的随机波动就有 ±20% 以上，k 没有意义


def nn(v):
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def load_inputs():
    apps = {}
    for r in pd.read_csv(RAW / "steam_appdetails_truth_v2.csv").itertuples():
        apps[r.appid] = (r.appid, r.success, nn(r.name), nn(r.app_type), r.is_free, nn(r.release_date),
                         nn(r.initial_price_usd), nn(r.genres))
    reviews = {}
    for r in pd.read_csv(RAW / "steam_reviews_at_cutoff_truth_v2.csv").itertuples():
        reviews[(r.appid, "truth_v2", r.cutoff_date)] = (
            r.appid, "truth_v2", r.cutoff_date, nn(r.total_reviews), nn(r.total_positive), nn(r.schinese_reviews),
            nn(r.steam_purchase_reviews), nn(r.reviews_before_policy), nn(r.sample_n),
            nn(r.sample_median_playtime_at_review_hrs), nn(r.first_review_date))
    charts = {}
    for r in pd.read_csv(RAW / "steamcharts_monthly_truth_v2.csv").sort_values("month").itertuples():
        charts.setdefault(r.appid, []).append((r.month, r.avg_players))
    return apps, reviews, charts


def build_rows(truth, apps, reviews, charts, policy):
    rows, dropped = [], []
    for t in truth.itertuples():
        value = t.steam_truth if pd.notna(t.steam_truth) else t.reported_number
        c = {"key": f"V{t.truth_v2_id:05d}", "appid": int(t.appid), "cohort": "truth_v2", "cutoff_date": t.as_of_date,
             "truth_value": float(value), "truth_type": t.tier, "source_url": t.source_url, "fallback_name": t.game}
        row, issues = E.clean_one(c, apps, reviews, charts, policy)
        if row is None:
            dropped.append({"truth_v2_id": t.truth_v2_id, "game": t.game, "tier": t.tier,
                            "reason": issues[-1][0] if issues else "UNKNOWN"})
            continue
        row.update({"truth_v2_id": t.truth_v2_id, "tier": t.tier, "bucket": t.bucket, "collect_round": t.collect_round,
                    "is_lower_bound": t.is_lower_bound, "quote_found_on_page": t.quote_found_on_page,
                    "early_access_at_report": t.early_access_at_report, "reported_number": t.reported_number,
                    "report_date": t.report_date, "as_of_rule": t.as_of_rule})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(dropped)


def summarize(df, label):
    out = {"subset": label, "rows": len(df), "games": df["appid"].nunique()}
    if len(df) == 0:
        return out
    for m in METHODS:
        ok = df[df[m].notna()]
        if len(ok) == 0:
            continue
        le = np.log(ok[m] / ok["truth_value"])
        out[f"{m}_n"] = len(ok)
        out[f"{m}_hit30"] = round(100 * (np.abs(ok[m] / ok["truth_value"] - 1) <= 0.3).mean(), 1)
        out[f"{m}_med_abs_log"] = round(float(np.median(np.abs(le))), 3)
        out[f"{m}_bias"] = round(float(np.median(le)), 3)  # >0 高估，<0 低估
    return out


def main():
    policy = E.load_policy()
    truth = pd.read_csv(RAW / "official_sales_truth_v2.csv")
    truth = truth[~truth["tier"].isin(["X_wiki", "X_not_sales", "X_num_mismatch"])]
    apps, reviews, charts = load_inputs()
    df, dropped = build_rows(truth, apps, reviews, charts, policy)
    print(f"新真值集可评估 {len(df)} 条 / {df.appid.nunique()} 款；清洗掉 {len(dropped)} 条：",
          dropped["reason"].value_counts().to_dict())
    few = df["total_reviews"] < MIN_REVIEWS
    print(f"评测数 < {MIN_REVIEWS} 的 {few.sum()} 条单独列出，不进主结果")

    conn = sqlite3.connect(str(BASE / "data" / "steam_sales_dw.db"))
    # 外部检验：只用旧真值（truth）校准、原有特征集（base）。这批行直接从 CSV 构造，没有补充特征
    fit = CS.fit_everything(conn, truth_cohorts=("truth",), feature_set="base")
    conn.close()
    df["charts_total_hours"] = df["charts_total_hours"].fillna(0.0)
    est = CS.estimate(df.reset_index(drop=True), fit)
    for m in METHODS + ["M3_P10", "M3_P90"]:
        df[m] = est[m].values
    df["log_err_fusion"] = np.log(df["Fusion_AB"] / df["truth_value"])
    df.to_csv(BASE / "data" / "cleaned" / "eval_truth_v2_rows.csv", index=False, encoding="utf-8-sig")
    dropped.to_csv(BASE / "data" / "cleaned" / "eval_truth_v2_dropped.csv", index=False, encoding="utf-8-sig")

    main_df = df[~few]
    steam = main_df[main_df["tier"].isin(["S1_steam_explicit", "S2_steam_labelled", "Q_steam_split"])]
    latest = steam.sort_values("cutoff_date").groupby("appid").tail(1)
    big = (steam["price_usd"] >= 30) | (steam["total_reviews"] >= 30000)
    subsets = [
        ("Steam 口径（S1+S2+Q）全部里程碑", steam),
        ("Steam 口径，每款只取最近一条", latest),
        ("S1 原句明确写 Steam", steam[steam.tier == "S1_steam_explicit"]),
        ("S2 标为 Steam 但原句没说平台", steam[steam.tier == "S2_steam_labelled"]),
        ("Steam 口径 · 大作（价格 >= $30 或评测 >= 3 万）", steam[big]),
        ("Steam 口径 · 小游戏（真值 < 3 万份）", steam[steam.truth_value < 30000]),
        ("Steam 口径 · 中间段", steam[~big & (steam.truth_value >= 30000)]),
        ("Steam 口径 · 公布日 < 2020（老）", steam[steam.cutoff_date < "2020-01-01"]),
        ("Steam 口径 · 公布时仍在抢先体验", steam[steam.early_access_at_report == "yes"]),
        ("Steam 口径 · 原文是'超过 X'（下限）", steam[steam.is_lower_bound.astype(str).str.upper() == "TRUE"]),
        ("P：PC 全商店（真实 Steam 销量 <= 真值，偏差应为负）", main_df[main_df.tier == "P_pc"]),
        ("X：全平台（只作参考，偏差应明显为负）", main_df[main_df.tier == "X_all_platforms"]),
        (f"评测数 < {MIN_REVIEWS}", df[few]),
    ]
    summ = pd.DataFrame([summarize(d, l) for l, d in subsets])
    summ.to_csv(BASE / "data" / "cleaned" / "eval_truth_v2_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    cols = ["subset", "rows", "games"] + [c for m in ["M0", "M4", "Signal_B", "Fusion_AB"]
                                          for c in (f"{m}_hit30", f"{m}_med_abs_log", f"{m}_bias")]
    print(summ[[c for c in cols if c in summ]].to_string(index=False))


if __name__ == "__main__":
    main()
