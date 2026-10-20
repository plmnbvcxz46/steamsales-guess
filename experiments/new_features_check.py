# -*- coding: utf-8 -*-
"""
新特征初筛：各语言评测占比、ITAD 价格历史 / 区域价 / 慈善包，和现有模型残差有没有关系。

残差 = ln(真值 / 估算)，取自 fact_sales_estimation（留一法估算，已经用过 简中占比、价格等特征）。
残差和某个新特征显著相关，说明这个特征里有现有模型没吃到的信息。
只是单变量初筛：107 条上测 20 多个特征，按 p<0.05 会冒出一两个假阳性，结论要等 2018 全量数据上重新训练再说。

用法：python experiments/new_features_check.py [--cohort truth|train_2018]
"""
import sys
import math
import sqlite3
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
cohort = sys.argv[sys.argv.index("--cohort") + 1] if "--cohort" in sys.argv else "truth"

con = sqlite3.connect(BASE / "data" / "steam_sales_dw.db")
samp = pd.read_sql("""SELECT appid, cohort, cutoff_date, total_reviews, schinese_ratio, price_usd, release_date,
                             log_k, truth_value
                      FROM dwd_game_sample WHERE model_eligible = 1 AND cohort = ?""", con, params=(cohort,))
est_cohort = "truth" if cohort == "truth" else "train_2018"
est = pd.read_sql("""SELECT appid, method_code, true_value, estimated_sales FROM fact_sales_estimation
                     WHERE cohort = ? AND estimated_sales > 0""", con, params=(est_cohort,))
est["resid"] = np.log(est.true_value / est.estimated_sales)
res = est.pivot_table(index="appid", columns="method_code", values="resid")
methods = [m for m in ["M4", "Fusion_AB"] if m in res.columns]
df = samp.merge(res[methods], left_on="appid", right_index=True, how="left")
df["cutoff"] = pd.to_datetime(df.cutoff_date, utc=True) + pd.Timedelta(hours=23, minutes=59)

# ---------------- 各语言占比 ----------------
lang = pd.read_csv(RAW / "review_language_counts.csv")
lang = lang[lang.cohort == cohort]
lw = lang.pivot_table(index="appid", columns="language", values="total_reviews", aggfunc="first")
df = df.merge(lw, left_on="appid", right_index=True, how="left")
for c in lw.columns:
    df[f"share_{c}"] = df[c] / df.total_reviews
df["share_spanish_all"] = (df.get("spanish", 0) + df.get("latam", 0)) / df.total_reviews
df["share_other"] = 1 - df[[f"share_{c}" for c in lw.columns]].sum(axis=1) - df.schinese_ratio
lang_feats = [f"share_{c}" for c in lw.columns] + ["share_spanish_all", "share_other"]

# ---------------- ITAD 价格历史 ----------------
hist = pd.read_csv(RAW / "itad_price_history.csv")
hist["ts"] = pd.to_datetime(hist.timestamp, utc=True)
bund = pd.read_csv(RAW / "itad_bundles.csv")
bund["pub"] = pd.to_datetime(bund.publish, utc=True, errors="coerce")


def price_feats(row):
    out = {}
    h = hist[(hist.appid == row.appid) & (hist.country == "US") & (hist.ts <= row.cutoff)].sort_values("ts")
    if len(h):
        out["launch_regular_usd"] = h.regular.iloc[0]
        out["regular_at_cutoff"] = h.regular.iloc[-1]
        out["price_drop_ratio"] = h.regular.iloc[-1] / h.regular.iloc[0] if h.regular.iloc[0] else np.nan
        cut = h.cut.fillna(0).values
        out["n_discounts"] = int(((cut > 0) & (np.r_[0, cut[:-1]] == 0)).sum())
        out["max_cut"] = cut.max()
        # 按时间加权：每条记录的折扣持续到下一条记录（最后一条持续到截止日）
        t = h.ts.values.astype("datetime64[s]").astype(np.int64)
        end = np.r_[t[1:], int(row.cutoff.timestamp())]
        dur = np.clip(end - t, 0, None)
        tot = dur.sum()
        out["frac_time_discounted"] = dur[cut > 0].sum() / tot if tot else np.nan
        out["frac_time_cut75"] = dur[cut >= 75].sum() / tot if tot else np.nan
        # 截止日前一年内折扣：近期打折对"评测/销量"影响最直接
        yr = t >= int(row.cutoff.timestamp()) - 365 * 86400
        out["n_discounts_last_year"] = int(((cut > 0) & (np.r_[0, cut[:-1]] == 0) & yr).sum())
        out["itad_history_days"] = tot / 86400
    for c in ["CN", "TR", "BR"]:
        hc = hist[(hist.appid == row.appid) & (hist.country == c) & (hist.ts <= row.cutoff)].sort_values("ts")
        if len(hc) and len(h) and h.regular.iloc[-1]:
            out[f"ln_regional_{c}"] = math.log(hc.regular.iloc[-1] / h.regular.iloc[-1])
    b = bund[(bund.appid == row.appid) & (bund.pub <= row.cutoff)]
    out["n_bundles"] = len(b)
    out["ever_bundled"] = int(len(b) > 0)
    return pd.Series(out, dtype=float)


pf = df.apply(price_feats, axis=1)
df = pd.concat([df, pf], axis=1)
# 区域价：各区的标准推荐价对美元价大体是固定比例，看的是"相对同类游戏偏便宜还是偏贵"
for c in ["CN", "TR", "BR"]:
    col = f"ln_regional_{c}"
    if col in df:
        df[col] = df[col] - df[col].median()
df["ln_launch_over_current"] = np.log(df.launch_regular_usd / df.price_usd)
df["ln_launch_price"] = np.log(df.launch_regular_usd)
price_list = ["ln_launch_price", "ln_launch_over_current", "price_drop_ratio", "n_discounts", "n_discounts_last_year",
              "max_cut", "frac_time_discounted", "frac_time_cut75", "n_bundles", "ever_bundled",
              "ln_regional_CN", "ln_regional_TR", "ln_regional_BR"]

print(f"样本：{cohort}，{len(df)} 条；有语言数据 {df.share_english.notna().sum()}，"
      f"有 ITAD 美区历史 {df.launch_regular_usd.notna().sum()}，进过慈善包 {int(df.ever_bundled.sum())}")
print(f"首发价与现价不同的 {(df.ln_launch_over_current.abs() > 0.01).sum()} 款\n")

target_cols = methods + (["log_k"] if cohort != "truth" else [])
rows = []
for f in lang_feats + price_list:
    if f not in df:
        continue
    r = {"feature": f, "n": int(df[f].notna().sum()), "median": df[f].median()}
    for m in target_cols:
        ok = df[[f, m]].dropna()
        if len(ok) > 10 and ok[f].nunique() > 1:
            rho = ok[f].rank().corr(ok[m].rank())
            n = len(ok)
            t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
            p = math.erfc(abs(t) / math.sqrt(2))  # 正态近似的双侧 p 值，n 在 100 左右够用
            r[f"rho_{m}"], r[f"p_{m}"] = rho, p
    rows.append(r)
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
out.to_csv(BASE / "experiments" / f"new_features_check_{cohort}.csv", index=False, encoding="utf-8-sig")
df.to_csv(BASE / "experiments" / f"new_features_{cohort}.csv", index=False, encoding="utf-8-sig")
