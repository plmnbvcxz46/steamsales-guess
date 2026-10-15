# -*- coding: utf-8 -*-
"""
周期快照采集 (src/snapshot_collector.py)

每隔一段时间（默认 1 小时）对一批游戏各抓一次"此刻"的状态，追加到 data/raw/snapshots.csv：
    当前在线人数   ISteamUserStats/GetNumberOfCurrentPlayers（只有此刻的值，没有历史，所以必须自己定期攒）
    累计评测数     IUserReviewsService/GetAppReviews（num_per_page=0 只拿汇总）
    当前价格/折扣  store appdetails（filters=price_overview，只要价格字段）

每一行都带 fetched_at（真实抓取时间）。ETL 用它做增量抽取的水位线：只读 fetched_at 比上次水位线新的行。

为什么是"每小时"而不是"每天"：
课程项目没有几周时间慢慢攒每日快照。机制和每日快照完全一样，只是粒度更细；
如果之后想改成每天一次，用 Windows 任务计划程序每天运行一次  python src/snapshot_collector.py --rounds 1  即可。

用法：python src/snapshot_collector.py --rounds 6 --interval 3600
"""
import sys
import csv
import json
import time
import pathlib
import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_FILE = RAW_DIR / "snapshots.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 SteamSalesCourseProject/2.0"}
FIELDS = ["fetched_at", "round_no", "appid", "name", "current_players", "total_reviews",
          "total_positive", "currency", "initial_price", "final_price", "discount_percent"]


def get_json(url, params):
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(20 * attempt)
    return None


def choose_games():
    """快照对象：真值集里的全部游戏（去重），FineBI 价格时间线和增量演示都用它们。"""
    rows = list(csv.DictReader(open(RAW_DIR / "recent_sales_truth.csv", encoding="utf-8")))
    seen, games = set(), []
    for r in rows:
        a = int(r["appid"])
        if a not in seen:
            seen.add(a)
            games.append((a, r["game"]))
    return games


def snapshot_one(appid):
    players = get_json("https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/", {"appid": appid})
    time.sleep(0.6)
    reviews = get_json("https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/",
                       {"input_json": json.dumps({"appid": appid, "num_per_page": 0, "languages": ["all"],
                                                  "purchase_type": 1, "filter_offtopic_activity": False})})
    time.sleep(0.6)
    price = get_json("https://store.steampowered.com/api/appdetails",
                     {"appids": appid, "cc": "us", "filters": "price_overview"})
    time.sleep(1.5)
    po = {}
    if price and price.get(str(appid), {}).get("success") and isinstance(price[str(appid)].get("data"), dict):
        po = price[str(appid)]["data"].get("price_overview", {}) or {}
    qs = (reviews or {}).get("response", {}).get("query_summary", {})
    return {
        "current_players": (players or {}).get("response", {}).get("player_count", ""),
        "total_reviews": qs.get("total_reviews", ""),
        "total_positive": qs.get("total_positive", ""),
        "currency": po.get("currency", ""),
        "initial_price": po.get("initial", 0) / 100.0 if po else "",
        "final_price": po.get("final", 0) / 100.0 if po else "",
        "discount_percent": po.get("discount_percent", "") if po else "",
    }


def main():
    rounds, interval = 1, 3600
    args = sys.argv[1:]
    if "--rounds" in args:
        rounds = int(args[args.index("--rounds") + 1])
    if "--interval" in args:
        interval = int(args[args.index("--interval") + 1])
    games = choose_games()
    start_round = 1
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            prev = [int(r["round_no"]) for r in csv.DictReader(f)]
        start_round = (max(prev) + 1) if prev else 1
    for k in range(rounds):
        round_no = start_round + k
        t0 = time.time()
        new_file = not OUT_FILE.exists()
        with open(OUT_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for appid, name in games:
                snap = snapshot_one(appid)
                w.writerow({"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "round_no": round_no,
                            "appid": appid, "name": name, **snap})
                f.flush()
        print(f"[snapshot] 第 {round_no} 轮完成，{len(games)} 款，用时 {time.time() - t0:.0f} 秒", flush=True)
        if k < rounds - 1:
            time.sleep(max(0, interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
