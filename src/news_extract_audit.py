# -*- coding: utf-8 -*-
"""
官方公告销量句子：结构化抽取 + 自动交叉核对 + 人工核对清单 (src/news_extract_audit.py)

流程（设计文档第 5 节）：
1. crawler.py news_scan 从 117 款真值游戏的官方公告里摘出 957 个"同时含销量关键词和数字"的句子；
   再用一条更严的规则（数字紧挨 copies/units/sold）筛到 99 句候选，见 candidate_filter()。
2. 大模型抽取：由 Claude（本次协作的 AI）逐句阅读候选句，只根据句子本身填写
   数字 / 指标类型 / 平台口径 / 初审意见，写在下面的 LLM_EXTRACTION 里；不凭记忆补任何数字。
   只对"像是在说销量"的句子做了抽取，其余候选句视为"不是销量"。
3. 规则解析：用 parse_number_rule() 独立地从原句解析数字，和大模型抽取的数字对比，算"字面数字一致率"。
4. 交叉核对：同一款游戏、公布日期相差不超过 31 天、数字相差不超过 5% 的，与 Gamalytic 真值表自动对上。
5. 输出 data/cleaned/news_extraction_audit.csv，human_audit_status 一栏留空，由 Rex 人工逐条确认。
"""
import re
import pathlib
import datetime as dt

import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT = BASE_DIR / "data" / "cleaned" / "news_extraction_audit.csv"

STRICT = re.compile(r"(\d[\d,.'’]*\s*(k|m|million|thousand)?\s*(copies|units|sales)|(sold|sales of)\s.{0,40}?\d)", re.I)

# (appid, 公告日期, 句子开头片段, 抽取数字, 指标类型, 平台口径, 初审意见)
# 指标类型：cumulative_copies_sold 累计售出份数 / annual_copies_sold 某年售出 / daily 单日
#           forecast_or_target 预测或目标（不是实际销量） / other_game 说的是别的游戏 / combined 多款游戏合计
# 平台口径：steam / pc / all_platforms / unspecified（原句没说）
LLM_EXTRACTION = [
    (227000, "2025-12-31", "Strangeland also hit some big milestones", 50000, "other_game", "unspecified",
     "句子说的是同一开发商的另一款游戏 Strangeland，不是 Primordia"),
    (227000, "2022-12-05", "306,105 copies sold", 306105, "cumulative_copies_sold", "unspecified",
     "只有一行数字，缺上下文，平台不明"),
    (250620, "2024-05-29", "We would like to thank all of our players", 1000000, "cumulative_copies_sold", "unspecified",
     "over 1 million，平台不明"),
    (250620, "2014-07-23", "Among the Sleep has currently sold more than 30.000", 30000, "cumulative_copies_sold", "all_platforms",
     "across all channels = 全渠道；30.000 是欧洲写法的三万"),
    (250620, "2014-06-06", "So far we've sold 17,603 copies of Among the Sleep on Steam", 17603, "cumulative_copies_sold", "steam",
     "明确是 Steam"),
    (252490, "2025-07-03", "[img src=\" 2025 we hit a huge milestone", 20000000, "cumulative_copies_sold", "pc",
     "on PC alone，PC 端（不一定全是 Steam）"),
    (280160, "2019-10-04", "It became Lince Works ' debut game", 500000, "cumulative_copies_sold", "unspecified",
     "around 500,000，原句没说平台；Gamalytic 用的 32 万是另一个来源的 PC 数"),
    (304730, "2015-07-10", "Since its release in September last year, Train Fever", 100000, "cumulative_copies_sold", "unspecified",
     "100.000 是欧洲写法的十万"),
    (402710, "2023-01-18", "I still can’t believe how far we've come", 600000, "cumulative_copies_sold", "unspecified",
     "over 600,000 units sold，与真值表 2023-01-17 的 60 万一致"),
    (413150, "2026-02-26", "Now, 10 years later, it's one of the highest rated", 50000000, "cumulative_copies_sold", "all_platforms",
     "worldwide = 全平台"),
    (433340, "2022-01-13", "Monomi Park is proud to announce we have officially sold", 5000000, "cumulative_copies_sold", "unspecified",
     "与真值表一致，但原句没说是 Steam 还是全平台，作为 Steam 真值有疑问"),
    (433340, "2020-09-15", "Slime Rancher Hits 3 Million Copies Sold", 3000000, "cumulative_copies_sold", "unspecified", "标题句"),
    (433340, "2019-02-27", "Slime Rancher has sold over 2 MILLION copies", 2000000, "cumulative_copies_sold", "unspecified", ""),
    (538840, "2017-08-07", "In a celebration of selling over 2000 copies", 2000, "cumulative_copies_sold", "unspecified", ""),
    (659540, "2024-08-19", "So starting this off with a HOLY MOLY WARSIM HAS HIT 100K", 100000, "cumulative_copies_sold", "steam",
     "ON STEAM"),
    (659540, "2024-03-21", "2017 - 6370 Copies sold", 6370, "annual_copies_sold", "unspecified", "按年明细，非累计"),
    (659540, "2024-03-21", "2021 - 6871 Copies sold", 6871, "annual_copies_sold", "unspecified",
     "2017~2021 五年相加 31,171，与真值表 2021-12 的 3 万吻合"),
    (659540, "2022-12-20", "But in all seriousness I can't thank everyone enough", 40000, "cumulative_copies_sold", "unspecified", ""),
    (659540, "2021-12-25", "Warsim made $430 yesterday from 81 copies sold", 81, "daily", "unspecified", "单日销量，不是累计"),
    (659540, "2021-12-11", "Hello everyone, I've been working a while on this content update", 30000, "cumulative_copies_sold",
     "unspecified", "30'000 copies sold online，与真值表 2021-12-12 的 3 万一致"),
    (659540, "2020-10-10", "After a few years it ended up on steam", 22000, "cumulative_copies_sold", "unspecified",
     "22'000，自 2017 年底上 Steam 以来"),
    (756800, "2024-11-25", "We are proud to announce that Contraband Police has surpassed", 1000000, "cumulative_copies_sold", "steam",
     "明确是 Steam"),
    (813630, "2024-01-03", "But Supraland was kinda rushed", 10000, "forecast_or_target", "unspecified", "当初的预期，不是销量"),
    (813630, "2019-12-30", "To the contrary, it exceeded my expectations by far", 100000, "cumulative_copies_sold", "unspecified",
     "100,000th unit"),
    (914020, "2024-06-05", "It's crazy that this game that only 3 people worked on", 90000, "cumulative_copies_sold", "steam", "on Steam"),
    (975370, "2025-04-21", "And more good news: Last week we officially surpassed", 1000000, "cumulative_copies_sold", "steam", "on Steam"),
    (1094520, "2022-07-28", "We are excited and proud to announce that Sands of Salzaar", 1500000, "cumulative_copies_sold",
     "all_platforms", "across all platforms 150 万；真值表用的是 2022-07-27 的 100 万，口径可能是 Steam 部分"),
    (1121640, "2022-11-20", "The Wandering Village has now sold over 150’000", 150000, "cumulative_copies_sold", "unspecified", ""),
    (1121640, "2022-09-24", "And oh boy did we have a reason to celebrate", 100000, "cumulative_copies_sold", "steam",
     "first week on Steam"),
    (1121640, "2022-09-16", "The Wandering Village has sold 50’000 copies", 50000, "cumulative_copies_sold", "unspecified", "前两天"),
    (1170950, "2024-07-23", "3 copies of Mortal Online 2", 3, "forecast_or_target", "unspecified", "抽奖送 3 份，不是销量"),
    (1284190, "2026-06-16", "Today we are thrilled to share that Planet Crafter sold more than 2.000.000", 2000000,
     "cumulative_copies_sold", "unspecified", "2.000.000 是欧洲写法的两百万"),
    (1285060, "2022-10-22", "It should sell 3000 copies in the next 12 months", 3000, "forecast_or_target", "unspecified", "预测"),
    (1331550, "2025-12-04", "We are proud to announce that Big Ambitions has now sold", 600000, "cumulative_copies_sold", "unspecified", ""),
    (1331550, "2025-12-04", "Together with Startup Company, this pushes our lifetime", 1100000, "combined", "steam",
     "两款游戏合计，不能当 Big Ambitions 的销量"),
    (1331550, "2024-12-28", "First and foremost, we're thrilled to announce that Big Ambitions", 500000, "cumulative_copies_sold",
     "unspecified", ""),
    (1331550, "2023-08-19", "Launching Big Ambitions, we dared to dream", 50000, "forecast_or_target", "unspecified", "当初的目标"),
    (1383250, "2022-06-28", "Today I've surpassed 10.000 copies sold", 10000, "cumulative_copies_sold", "unspecified",
     "10.000 是欧洲写法的一万；与真值表同日"),
    (1615290, "2022-05-19", "The game has officially reached 100,000 copies sold internationally", 74000, "cumulative_copies_sold",
     "steam", "全平台 10 万，其中 Steam 7.4 万；真值表用了 10 万，按 Steam 口径应为 7.4 万"),
    (1737340, "2023-02-21", "🎉100,000 COPIES SOLD🎉 Thank you for all the support", 100000, "cumulative_copies_sold",
     "unspecified", "与真值表同日"),
]


def candidate_filter(df):
    df = df[df["sentence"].str.contains(r"\b(?:sold|copies|units)\b", case=False, regex=True)]
    df = df.drop_duplicates(["appid", "sentence"])
    return df[df["sentence"].apply(lambda s: bool(STRICT.search(s)))].copy()


def parse_number_rule(sentence):
    """
    规则解析：取句中紧挨 copies/units/sold 的那个数字。
    故意只做最朴素的处理（逗号当千分位、点当小数点），用来和大模型抽取对比，暴露欧洲写法等坑。
    """
    m = re.search(r"(\d[\d,.]*)\s*(k|m|million|thousand)?\s*(copies|units|sold|times)", sentence, re.I) or \
        re.search(r"(?:sold|sales of)\D{0,40}?(\d[\d,.]*)\s*(k|m|million|thousand)?", sentence, re.I)
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", "").rstrip("."))
    except ValueError:
        return None  # 例如 "2.000.000"：朴素规则解析不了

    unit = (m.group(2) or "").lower()
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(unit, 1)
    return int(round(num * mult))


def main():
    cand = candidate_filter(pd.read_csv(RAW_DIR / "news_sales_sentences.csv"))
    truth = pd.read_csv(RAW_DIR / "recent_sales_truth.csv")
    rows = []
    for appid, date, prefix, num, metric, scope, note in LLM_EXTRACTION:
        hit = cand[(cand["appid"] == appid) & (cand["news_date"] == date) &
                   (cand["sentence"].str.contains(prefix, regex=False))]
        if hit.empty:
            raise SystemExit(f"找不到原句：{appid} {date} {prefix}")
        sent = hit.iloc[0]
        rule_num = parse_number_rule(sent["sentence"])
        match = ""
        t = truth[truth["appid"] == appid]
        for tr in t.itertuples():
            days = abs((dt.date.fromisoformat(date) - dt.date.fromisoformat(tr.report_date)).days)
            if days <= 31 and abs(num - tr.reported_sales) / tr.reported_sales <= 0.05:
                match = f"与真值表 {tr.report_date} 的 {tr.reported_sales} 一致"
        rows.append({
            "appid": appid, "game": t["game"].iloc[0] if len(t) else "", "news_date": date, "news_url": sent["url"],
            "raw_sentence": sent["sentence"], "llm_number": num, "llm_metric": metric, "llm_platform_scope": scope,
            "rule_number": rule_num, "number_agree": int(rule_num == num), "truth_cross_check": match,
            "llm_note": note, "usable_as_steam_truth": int(metric == "cumulative_copies_sold" and scope == "steam"),
            "human_audit_status": "", "human_audit_note": "",
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    n_cand = len(cand)
    sales = out[out["llm_metric"] == "cumulative_copies_sold"]
    print(f"候选句 {n_cand} 句；判为与销量有关并抽取 {len(out)} 句；其中累计售出份数 {len(sales)} 句")
    print(f"规则解析与大模型抽取的数字一致率：{out['number_agree'].mean() * 100:.1f}%（{out['number_agree'].sum()}/{len(out)}）")
    print("  不一致的：")
    for r in out[out["number_agree"] == 0].itertuples():
        print(f"    {r.game[:24]:24s} 规则={r.rule_number}  抽取={r.llm_number}  | {r.raw_sentence[:80]}")
    print(f"可直接当 Steam 口径累计销量的：{int(out['usable_as_steam_truth'].sum())} 句")
    print(f"与 Gamalytic 真值表自动对上的：{(out['truth_cross_check'] != '').sum()} 句")
    print(out["llm_platform_scope"].value_counts().to_string())


if __name__ == "__main__":
    main()
