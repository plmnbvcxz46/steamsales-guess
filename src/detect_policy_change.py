# -*- coding: utf-8 -*-
"""
用真实数据确定"Steam 游玩后提示写评测"的上线月份 (src/detect_policy_change.py)

思路：
- crawler.py policy_scan 已经抓好了 40 款老游戏（2016 年以前发售、2018 年玩家数 >= 50 万）
  在 2018-07 ~ 2020-12 每个月新增的评测数，存于 data/raw/review_monthly_policy_scan.csv；
- 老游戏的销量在这两年里变化不大，所以"每月新增评测数"的整体跳升只能来自评测机制的变化；
- 每款游戏先除以自己在基线期（2018-07 ~ 2019-06）的月中位数，得到"评测指数"，再取 40 款的中位数；
- 判定规则：第一个"指数 >= 2，且之后连续 6 个月都 >= 2"的月份。
  要求连续 6 个月，是为了排除秋促、夏促这种只持续一两个月的尖峰（数据里 2018-11、2019-06/07 就有）。

输出：data/raw/policy_change_detected.json，crawler.py truth 与 ETL 都读它。
"""
import json
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SCAN_CSV = RAW_DIR / "review_monthly_policy_scan.csv"
OUT_JSON = RAW_DIR / "policy_change_detected.json"
MONTH_INDEX_CSV = RAW_DIR / "review_monthly_index.csv"


def main():
    df = pd.read_csv(SCAN_CSV)
    base = (df[(df["month"] >= "2018-07") & (df["month"] <= "2019-06")]
            .groupby("appid")["new_reviews"].median().rename("baseline"))
    df = df.join(base, on="appid")
    df = df[df["baseline"] > 0].copy()
    df["review_index"] = df["new_reviews"] / df["baseline"]
    monthly = (df.groupby("month")["review_index"]
               .agg(median_index="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                    n_games="count")
               .reset_index())
    monthly.to_csv(MONTH_INDEX_CSV, index=False, encoding="utf-8")

    months = monthly["month"].tolist()
    idx = monthly["median_index"].tolist()
    found = None
    for i in range(len(months) - 6):
        if all(v >= 2.0 for v in idx[i:i + 7]):
            found = months[i]
            break
    if found is None:
        raise SystemExit("没有找到持续跳升的月份，请检查数据")

    y, m = map(int, found.split("-"))
    policy_ts = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp())
    pre = monthly[monthly["month"] < found]["median_index"]
    post = monthly[(monthly["month"] >= found) & (monthly["month"] <= "2020-12")]["median_index"]
    result = {
        "policy_month": found,
        "policy_ts": policy_ts,
        "rule": "第一个评测指数中位数 >= 2 且之后连续 6 个月都 >= 2 的月份",
        "n_games": int(df["appid"].nunique()),
        "median_index_before": round(float(np.median(pre)), 3),
        "median_index_after": round(float(np.median(post)), 3),
        "jump_ratio": round(float(np.median(post) / np.median(pre)), 3),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
