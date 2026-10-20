# -*- coding: utf-8 -*-
"""
合并新真值集 (src/truth_v2_merge.py)：两轮人工智能收集的"官方公布销量"-> data/raw/official_sales_truth_v2.csv

输入（data/raw/source/）：
- steam_official_sales_ground_truth_audited.csv  第一轮 304 条，附逐条核对结果（quote_found_on_page 等）
- steam_official_sales_round2.csv                第二轮 214 条，专找大作 / 老游戏的 Steam 口径

为什么不直接信收集方给的标签：
1. 第二轮有 146 条抄自维基百科"最畅销 PC 游戏"列表——不是开发商原话，很多是 Steam 出现以前的零售销量，
   而且 quote 里的数字和 number_text 对不上（Half-Life 写 930 万，引文是 250 万）。一律排除。
2. 第二轮标成 quote_says_steam 的 203 条里，原句真正出现 "Steam" 的只有 58 条。所以"Steam 口径"由我们自己按原句判断。
3. 第一轮的逐条核对不可靠：我们自己打开确认过原话存在的几个页面（寂静岭 2、Content Warning、NEBULOUS、钢铁雄心 4），
   核对结果却是"失效"或"找不到"。所以 quote_found_on_page 只作为参考列保留，不用来筛样本。

分层（tier）：
- S1_steam_explicit  标为 Steam 口径，且原句出现 "Steam"                      -> 最干净
- S2_steam_labelled  标为 Steam 口径，原句没说平台（多为 r/gamedev 小开发者）   -> 大概率是 Steam
- Q_steam_split      原文给了 Steam 占比，Steam 销量 = 总数 × 占比              -> 推算值
- P_pc               PC 全商店 / PC 未说明（含 GOG、Epic 等，真实 Steam 销量 <= 这个数）
- X_all_platforms    全平台、没有拆分                                         -> 不进准确率评估
- X_wiki / X_not_sales / X_num_mismatch                                       -> 排除

as_of_date（这个数字对应的时间点）：
很多数字是"首日 / 首周 / 首月卖了多少"，但帖子是几个月甚至几年后才发的（《Open The Gates!》首日 100 份，10 个月后发帖）。
如果按发帖日截止去数评测，评测是 10 个月的累计、销量只是首日的，k 会被严重低估。
所以从原句和备注里识别"首日/首周/前 N 天/首月/首年"，把截止日改成 发售日 + 对应天数（不晚于发帖日）。
"""
import re
import pathlib

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
SRC = BASE / "data" / "raw" / "source"
OUT = BASE / "data" / "raw" / "official_sales_truth_v2.csv"

SALES_WORDS = re.compile(r"sold|sales|sell|copies|units|purchas|bought|buyers|销量|售出|卖出|万份", re.I)


def norm(s):
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def steam_share(text):
    m = re.search(r"([0-9.]+)\s*\(\s*Steam\s*\)", str(text))
    return float(m.group(1)) if m else None


PERIODS = [
    (r"first (24|48) hours|day ?1\b|first day|launch day|on day one|at launch|opening day", 2),
    (r"first (two|2) days", 2), (r"first (three|3) days", 3), (r"first (\d+) days", None),
    (r"first week|week ?1\b|launch week|opening week|first 7 days|in a week", 7),
    (r"first (two|2) weeks|within two weeks", 14),
    (r"first month|month ?1\b|first 30 days|in a month|within a month", 30),
    (r"first (two|2) months", 61), (r"first (three|3) months", 92),
    (r"first year|year ?1\b|first 12 months", 365),
]


def period_days(text):
    t = str(text).lower()
    for pat, days in PERIODS:
        m = re.search(pat, t)
        if m:
            return int(m.group(1)) if days is None else days
    return None


def tier_of(r):
    if "wikipedia.org" in str(r.source_url):
        return "X_wiki"
    if not SALES_WORDS.search(str(r.quote)):
        return "X_not_sales"
    num_ok = norm(r.number_text) in norm(r.quote) or str(int(r.reported_number)) in norm(r.quote)
    if not num_ok and not re.search(r"million|thousand|half|万", str(r.quote), re.I):
        return "X_num_mismatch"
    steam_in = "steam" in str(r.quote).lower()
    if r.platform_scope == "steam":
        return "S1_steam_explicit" if steam_in else "S2_steam_labelled"
    if pd.notna(r.steam_share):
        return "Q_steam_split"
    if r.platform_scope in ("pc_all_stores", "pc_unspecified"):
        return "P_pc"
    return "X_all_platforms"


def main():
    a = pd.read_csv(SRC / "steam_official_sales_ground_truth_audited.csv")
    a["collect_round"] = 1
    b = pd.read_csv(SRC / "steam_official_sales_round2.csv").rename(columns={"quote_verified": "quote_found_on_page"})
    b["collect_round"] = 2
    d = pd.concat([a, b], ignore_index=True)
    d = d.drop(columns=["confidence"])
    d["report_date"] = pd.to_datetime(d["report_date"]).dt.strftime("%Y-%m-%d")
    d["steam_share"] = d["steam_or_pc_share"].map(steam_share)
    d["tier"] = d.apply(tier_of, axis=1)
    # Steam 口径的"真值"：S1/S2 直接用；Q 用总数 × Steam 占比；P 是 Steam 销量的上界
    d["steam_truth"] = None
    s = d["tier"].isin(["S1_steam_explicit", "S2_steam_labelled", "P_pc"])
    d.loc[s, "steam_truth"] = d.loc[s, "reported_number"]
    q = d["tier"] == "Q_steam_split"
    d.loc[q, "steam_truth"] = (d.loc[q, "reported_number"] * d.loc[q, "steam_share"]).round()
    d["period_days"] = [period_days(f"{q} {n}") for q, n in zip(d["quote"], d["notes"].fillna(""))]
    rel = pd.to_datetime(d["steam_release_date"], errors="coerce")
    rep = pd.to_datetime(d["report_date"])
    asof = rel + pd.to_timedelta(d["period_days"], unit="D")
    use = d["period_days"].notna() & rel.notna() & (asof < rep)
    d["as_of_date"] = d["report_date"]
    d.loc[use, "as_of_date"] = asof[use].dt.strftime("%Y-%m-%d")
    d["as_of_rule"] = ""
    d.loc[use, "as_of_rule"] = "release+" + d.loc[use, "period_days"].astype(int).astype(str) + "d"
    d.insert(0, "truth_v2_id", range(1, len(d) + 1))
    d.to_csv(OUT, index=False, encoding="utf-8")
    print(f"-> {OUT.relative_to(BASE)}：{len(d)} 行 / {d.appid.nunique()} 款")
    print(pd.crosstab(d["tier"], d["collect_round"], margins=True))
    print(f"按'首日/首周/首月'改了截止日的：{(d['as_of_rule'] != '').sum()} 条")


if __name__ == "__main__":
    main()
