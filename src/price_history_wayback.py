# -*- coding: utf-8 -*-
"""
真实价格历史采集：从互联网档案馆 (Wayback Machine) 的 Steam 商店页存档里提取当时的标价 (src/price_history_wayback.py)

为什么用档案馆：
- IsThereAnyDeal 有完整的价格历史，但要注册拿 Key；SteamDB 禁止爬取。
- 档案馆免费、无需 Key，保存了大量 Steam 商店页的历史快照，页面里直接写着当时的价格和折扣。

局限（写进报告）：
- 每款游戏每年只取一个存档（见 pick_yearly 的取舍说明）；
- 存档是"抽查"，不是每天都有，所以只能看到被存下来的那几天的价格，短促销很可能漏掉；
- 存档机器人有时从美国以外访问，页面显示加元、欧元等，我们只保留美元 ($) 的快照；
- 一款游戏在多个时间点价格不变时，SCD2 只保留一个版本，价格变了才新开版本。

用法：python src/price_history_wayback.py      # 输出 data/raw/price_history_wayback.csv
"""
import re
import csv
import json
import time
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / "data" / "cache" / "wayback"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 SteamSalesCourseProject/2.0 (price history for a database course)"}
GAP_SECONDS = 2.0  # 每个线程两次请求之间至少隔 2 秒，别给档案馆添负担
N_WORKERS = 2      # 档案馆单页要 5~30 秒，串行太慢；但 6 个线程时被档案馆断开过连接，最后定为 2 个
N_GAMES = 12
_local = threading.local()


def polite_get(url):
    for attempt in range(1, 5):
        last = getattr(_local, "last", 0.0)
        passed = time.time() - last
        if passed < GAP_SECONDS:
            time.sleep(GAP_SECONDS - passed)
        _local.last = time.time()
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(15 * attempt)
    return None


def list_snapshots(appid):
    """用 CDX 接口列出这款游戏商店页的所有存档时间点，每个月最多取一个。"""
    p = CACHE_DIR / f"cdx_{appid}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    url = ("https://web.archive.org/cdx/search/cdx?url=store.steampowered.com/app/%d/&output=json"
           "&from=2014&to=2026&collapse=timestamp:6&filter=statuscode:200" % appid)
    r = polite_get(url)
    if r is None:
        return []  # 请求失败时不写缓存，否则下次会一直以为"没有存档"（踩过的坑）
    rows = r.json()[1:] if r.text.strip() else []
    stamps = [row[1] for row in rows]
    p.write_text(json.dumps(stamps), encoding="utf-8")
    return stamps


def pick_yearly(stamps):
    """
    每年只留第一个存档，控制请求量。
    取舍过程：一开始按季度、24 款游戏，单线程估计要两个多小时；改成半年、20 款、4 线程，
    实测档案馆单页 10~30 秒，还是要一个多小时；最后定为每年一个、12 款、6 线程。
    SCD2 只是演示"价格变了就开新版本"的机制，十几款游戏、每年一个观测点已经够用。
    """
    seen, out = set(), []
    for s in sorted(stamps):
        q = s[:4]
        if q not in seen:
            seen.add(q)
            out.append(s)
    return out


def parse_price(html):
    """
    在第一个"购买本游戏"区块里找价格。三种情况：
    1. 原价：<div class="game_purchase_price price">$14.99</div>
    2. 打折：discount_pct / discount_original_price / discount_final_price
    3. 找不到（年龄验证页、下架页等）：返回 None
    """
    i = html.find('class="game_area_purchase_game')
    if i < 0:
        return None
    block = html[i:i + 6000]
    m_final = re.search(r'discount_final_price">\s*([^<]+?)\s*<', block)
    m_orig = re.search(r'discount_original_price">\s*([^<]+?)\s*<', block)
    m_pct = re.search(r'discount_pct">\s*-?(\d+)%', block)
    m_plain = re.search(r'game_purchase_price price"[^>]*>\s*([^<]+?)\s*<', block)
    if m_final and m_orig:
        orig_txt, final_txt, pct = m_orig.group(1), m_final.group(1), int(m_pct.group(1)) if m_pct else None
    elif m_plain:
        orig_txt = final_txt = m_plain.group(1)
        pct = 0
    else:
        return None

    def usd(t):
        t = t.strip()
        if not re.fullmatch(r"\$\s?[\d,]+\.\d\d( USD)?", t):
            return None  # 只认美元，CDN$、€、£、Free 之类都不要
        return float(re.sub(r"[^\d.]", "", t))

    orig, final = usd(orig_txt), usd(final_txt)
    if orig is None or final is None:
        return None
    if pct is None:
        pct = int(round((1 - final / orig) * 100)) if orig > 0 else 0
    return {"base_price_usd": orig, "final_price_usd": final, "discount_percent": pct,
            "raw_original": orig_txt, "raw_final": final_txt}


def fetch_snapshot_price(appid, stamp):
    p = CACHE_DIR / f"page_{appid}_{stamp}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # id_ 后缀让档案馆返回原始页面，不插入它自己的工具栏
    r = polite_get(f"https://web.archive.org/web/{stamp}id_/https://store.steampowered.com/app/{appid}/")
    result = parse_price(r.text) if r is not None else None
    p.write_text(json.dumps(result), encoding="utf-8")
    return result


def choose_games():
    """选真值集里销量最高的 N_GAMES 款（去重），它们在 FineBI 大屏上最常出现，存档也最多。"""
    rows = list(csv.DictReader(open(RAW_DIR / "recent_sales_truth.csv", encoding="utf-8")))
    rows.sort(key=lambda r: -int(r["reported_sales"]))
    seen, picked = set(), []
    for r in rows:
        a = int(r["appid"])
        if a not in seen:
            seen.add(a)
            picked.append((a, r["game"]))
        if len(picked) >= N_GAMES:
            break
    return picked


def crawl_game(game):
    appid, name = game
    stamps = pick_yearly(list_snapshots(appid))
    rows = []
    for s in stamps:
        price = fetch_snapshot_price(appid, s)
        if price:
            rows.append({"appid": appid, "name": name,
                         "observed_date": f"{s[:4]}-{s[4:6]}-{s[6:8]}",
                         "wayback_timestamp": s,
                         "archive_url": f"https://web.archive.org/web/{s}/https://store.steampowered.com/app/{appid}/",
                         **price})
    print(f"[wayback] {appid} {name}: {len(stamps)} 个年度存档，其中 {len(rows)} 个能读出美元价格", flush=True)
    return rows


def main():
    out = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for rows in pool.map(crawl_game, choose_games()):
            out.extend(rows)
    out.sort(key=lambda r: (r["appid"], r["wayback_timestamp"]))
    with open(RAW_DIR / "price_history_wayback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["appid", "name", "observed_date", "wayback_timestamp", "archive_url",
                                          "base_price_usd", "final_price_usd", "discount_percent",
                                          "raw_original", "raw_final"])
        w.writeheader()
        w.writerows(out)
    print(f"[wayback] 共 {len(out)} 条价格观测 -> data/raw/price_history_wayback.csv", flush=True)


if __name__ == "__main__":
    main()
