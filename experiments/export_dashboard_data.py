# -*- coding: utf-8 -*-
"""
把数据库里的结果导出成看板用的 JSON（data/cleaned/dashboard/*.json）。
看板：https://claude.ai/artifact/K1uKL2acdTFuyshh3Ri9xt（Dashboard 类型的 artifact）。
每次重新训练后跑一遍，再把变了的 JSON 作为 asset 上传、更新看板里对应数据集的 source.url（见 AGENTS.md 第十节）。
待办（todo.json）、候选信号（signals.json）、术语（glossary.json）是手写的，不在这里生成。

用法：python experiments/export_dashboard_data.py
"""
import json
import sqlite3
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "cleaned" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(BASE / "data" / "steam_sales_dw.db")

METHOD_NAME = {"M0": "M0 固定 30 倍", "M1": "M1 按年份文献 k", "M2": "M2 回归", "M3": "M3 分位数回归", "M4": "M4 树模型",
               "Signal_B": "信号 B", "Fusion_AB": "融合 A+B", "Gamalytic_same": "Gamalytic 自己的估算"}
GROUP_NAME = {"2018 五折交叉验证": "2018 训练集（交叉验证）", "合并真值集 全部": "新旧真值合并",
              "真值集 全部": "旧真值集（Gamalytic 表）", "新真值集 全部": "新真值集（原句写明 Steam）",
              "合并真值集 信号B可用子集": "合并 · 有信号 B 的", "外部检验 base（旧真值校准→新真值集）": "外部检验 · 原有特征",
              "外部检验 ext（旧真值校准→新真值集）": "外部检验 · 加新特征", "合并真值集 政策前公布": "合并 · 2019-11 前公布",
              "合并真值集 跨时代": "合并 · 跨政策时间点", "合并真值集 政策后发售": "合并 · 2019-11 后发售"}


def dump(name, rows):
    with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"-> data/cleaned/dashboard/{name}.json  {len(rows)} 行")


def num(v, nd=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


# 1. 各测试集 × 方法的命中率
e = pd.read_sql("SELECT eval_cohort, method_code, sample_size, hit_rate_30pct, hit_rate_50pct, mdape_pct FROM ads_model_eval_summary", con)
e = e[e.eval_cohort.isin(GROUP_NAME) & e.method_code.isin(METHOD_NAME)]
dump("metrics", [{"group": GROUP_NAME[r.eval_cohort], "method": METHOD_NAME[r.method_code], "code": r.method_code,
                  "n": int(r.sample_size), "hit30": num(r.hit_rate_30pct / 100), "hit50": num(r.hit_rate_50pct / 100),
                  "mdape": num(r.mdape_pct / 100)} for r in e.itertuples()])

# 公共：参与校准的真值行 + 各方法估算（按游戏留一法）
d = pd.read_sql("""SELECT sample_id, name, is_indie, price_usd, price_at_cutoff_usd, truth_value FROM dwd_game_sample
                   WHERE cohort IN ('truth', 'truth_v2') AND model_eligible = 1""", con)
f = pd.read_sql("SELECT sample_id, method_code, estimated_sales FROM fact_sales_estimation WHERE cohort IN ('truth', 'truth_v2')", con)
d = d.merge(f.pivot_table(index="sample_id", columns="method_code", values="estimated_sales"), left_on="sample_id", right_index=True)
d["price"] = d.price_at_cutoff_usd.fillna(d.price_usd)
aaa, big = (d.price >= 40) & (d.is_indie == 0), d.truth_value >= 1e6
LINE = {"M4": "M4（评测数推算）", "Signal_B": "信号 B（在线 × 时长）", "Fusion_AB": "融合 A+B"}


def stats(g, m):
    x = g[g[m].notna()]
    if len(x) == 0:
        return None
    le = np.log(x[m] / x.truth_value)
    return {"n": int(len(x)), "hit30": num(np.mean(np.abs(np.exp(le) - 1) <= 0.3)),
            "med_err": num(np.exp(np.median(np.abs(le))) - 1), "bias": num(np.exp(np.median(le)) - 1)}


# 2. 按游戏规模分组
segs = [("3A", "无独立标签、截止日原价 ≥ $40", d[aaa]), ("爆款独立", "独立标签、Steam 销量 ≥ 100 万", d[big & (d.is_indie == 1)]),
        ("大型作品", "Steam 销量 ≥ 100 万（全部）", d[big]), ("中型作品", "10 万 ~ 100 万份，不含 3A", d[(d.truth_value >= 1e5) & ~big & ~aaa]),
        ("小型作品", "< 10 万份，不含 3A", d[(d.truth_value < 1e5) & ~aaa]), ("全部", f"新旧合并 {len(d)} 条", d)]
rows = []
for s, defn, g in segs:
    for m in ["M4", "Signal_B", "Fusion_AB"]:
        st = stats(g, m)
        if st:
            rows.append({"segment": s, "definition": defn, "method": METHOD_NAME[m], **st})
dump("segments", rows)

# 3. 按销量分档
edges = [0, 1e4, 5e4, 2e5, 1e6, 5e6, 1e12]
labels = ["< 1 万", "1 万 ~ 5 万", "5 万 ~ 20 万", "20 万 ~ 100 万", "100 万 ~ 500 万", "≥ 500 万"]
d["bin"] = pd.cut(d.truth_value, edges, labels=labels, right=False)
rows = []
for i, b in enumerate(labels):
    g = d[d.bin == b]
    for m in ["M4", "Signal_B", "Fusion_AB"]:
        st = stats(g, m) or {"n": 0, "hit30": None, "med_err": None}
        rows.append({"bin_order": i, "sales_bin": b, "method": LINE[m], "code": m, "n": st["n"], "hit30": st["hit30"],
                     "med_err": st.get("med_err")})
dump("by_sales", rows)

# 4. 有没有信号 B、A/B 分歧度
nb = d[d.Signal_B.isna()]
rows = [{"group": "没有信号 B", "n": int(len(nb)), "hit30": num(np.mean(np.abs(nb.Fusion_AB / nb.truth_value - 1) <= 0.3))}]
b = d[d.Signal_B.notna()].copy()
b["q"] = pd.qcut(np.abs(np.log(b.M4 / b.Signal_B)), 4, labels=["分歧最小 1/4", "第 2 个 1/4", "第 3 个 1/4", "分歧最大 1/4"])
for q, g in b.groupby("q", observed=True):
    rows.append({"group": "有 B · " + q, "n": int(len(g)), "hit30": num(np.mean(np.abs(g.Fusion_AB / g.truth_value - 1) <= 0.3))})
dump("confidence", rows)

# 5. 新真值集口径分层
DESC = {"S1_steam_explicit": ("S1", "原句写明 Steam", "参与校准"), "Q_steam_split": ("Q", "给了 Steam 占比，按比例推算", "参与校准"),
        "S2_steam_labelled": ("S2", "标了 Steam，原句没说平台", "只作参考"), "P_pc": ("P", "PC 全商店口径（Steam ≤ 该数）", "只作参考"),
        "X_all_platforms": ("X", "全平台、没有拆分", "排除"), "X_wiki": ("X", "抄自维基百科", "排除"),
        "X_not_sales": ("X", "不是销量（玩家数等）", "排除"), "X_num_mismatch": ("X", "数字和原句对不上", "排除")}
t = pd.read_sql("SELECT tier, COUNT(*) n FROM ods_official_sales_truth_v2 GROUP BY tier", con)
u = pd.read_sql("""SELECT truth_type tier, COUNT(*) dwd, SUM(model_eligible) used FROM dwd_game_sample
                   WHERE cohort = 'truth_v2' GROUP BY truth_type""", con)
t = t.merge(u, on="tier", how="left").fillna(0)
dump("tiers", [{"tier_code": r.tier, "tier": DESC[r.tier][0], "meaning": DESC[r.tier][1], "use": DESC[r.tier][2],
                "collected": int(r.n), "in_sample": int(r.dwd), "used": int(r.used)} for r in t.itertuples()])

# 6. 特征重要性
NEW = {"ln_launch_over_cutoff", "log1p_discounts", "max_cut", "frac_time_discounted", "frac_time_cut75", "log1p_bundles",
       "ln_regional_cn", "ln_regional_tr", "ln_regional_br", "log_price"}
fi = pd.read_sql("""SELECT feature_code, feature_cn_name, m2_ols_coef, m2_p_value, m4_rf_importance_pct, m4_gbdt_importance_pct
                    FROM ads_feature_importance WHERE feature_code <> 'intercept'""", con)
dump("features", [{"feature": r.feature_code, "name": r.feature_cn_name,
                   "importance": num((r.m4_rf_importance_pct + r.m4_gbdt_importance_pct) / 200),
                   "coef": num(r.m2_ols_coef, 3), "p": num(r.m2_p_value),
                   "is_new": bool(r.feature_code in NEW or r.feature_code.startswith("share_"))} for r in fi.itertuples()])

# 7. 融合方式对比
fc = pd.read_sql("""SELECT method_code, method_name, hit_rate_30pct, note FROM ads_model_eval_summary
                    WHERE eval_cohort = '融合方式对比（合并真值，按游戏留一法）'""", con)
rows = []
for r in fc.itertuples():
    logo = float(r.note.split("平均 |ln 误差| ")[1].split("；")[0])
    ext = float(r.note.split("外部检验 ")[1].split("；")[0])
    rows.append({"variant": r.method_code, "name": r.method_name, "hit30": num(r.hit_rate_30pct / 100),
                 "logo_abs_ln": num(logo, 3), "ext_abs_ln": num(ext, 3), "chosen": "选用" in r.note})
dump("fusion", rows)
