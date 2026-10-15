# -*- coding: utf-8 -*-
"""
Steam 多源数据采集 (src/crawler.py)

用法（在项目根目录运行）：
    python src/crawler.py policy_scan          # 抽 40 款老游戏，按月统计评测数，找"游玩后提示写评测"上线的月份
    python src/crawler.py train2018            # 2018 训练集第一轮：Ars 泄露表里玩家数 >= 2 万的游戏
    python src/crawler.py train2018 --extend   # 2018 训练集扩展轮：玩家数 5000 ~ 2 万的游戏
    python src/crawler.py train2018 --limit 300 --shard 0/2   # 只跑前 300 款里的第 0 片（两个进程并行时用）
    python src/crawler.py truth                # 近年真值集：Gamalytic 测试表里的 118 条公开销量
    python src/crawler.py first_review         # 商店发售日晚于截止日的（多为抢先体验转正），补查第一条评测日期
    python src/crawler.py news_scan            # 扫描真值集游戏的官方公告，摘出提到销量的原句
    python src/crawler.py export               # 把缓存整理成 data/raw/ 下的几张 CSV，供 ETL 读取

数据来源与接口：
1. Ars Technica 2018-07-01 成就接口泄露表（玩家数真值），见 data/raw/source/
2. 商店详情 store.steampowered.com/api/appdetails（价格、类型、发售日）
3. 评测接口 api.steampowered.com/IUserReviewsService/GetAppReviews/v1/
   - 这是 Service 类接口，参数必须放在 input_json 里传，直接拼在 URL 上会被忽略（踩过的坑）
   - date_range_start 传 0 会被当成"没传"，所以从 1 开始
   - languages 不传时只统计英文评测，要传 ["all"]
4. SteamCharts 月度平均在线人数（steamcharts.com/app/<appid>），用于信号 B
不使用 SteamDB（其 FAQ 禁止爬取）；SteamSpy 的 average_forever 现在全部返回 0，已不可用。

为什么每一次请求都单独缓存成一个 JSON：
中途断网、被限速或者手动停掉以后，重新运行会直接读缓存跳过已完成的请求，这就是最朴素的断点续传。
"""
import os
import re
import sys
import csv
import json
import time
import random
import pathlib
import datetime as dt

import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SOURCE_DIR = RAW_DIR / "source"
CACHE_DIR = BASE_DIR / "data" / "cache"
CHECKPOINT_FILE = CACHE_DIR / "crawler_checkpoint.json"
CRAWL_LOG_FILE = CACHE_DIR / "crawl_log.csv"

ARS_CSV = SOURCE_DIR / "games_achievements_players_2018-07-01_ars_original.csv"
GAMALYTIC_CSV = SOURCE_DIR / "gamalytic_benchmark_118.csv"

# 2018-07-01 00:00:00 UTC，泄露快照的日期
CUTOFF_2018_TS = 1530403200
# 2018 训练集抽样门槛。
# 第一版只取泄露玩家数 >= 2 万的游戏（小游戏的泄露估计误差大）。跑出 600 多款后发现 k 中位数高达 112：
# 按"玩家数"这个预测目标本身来筛样本，会让评测少的小游戏只有 k 特别大时才能入选（截断偏差）。
# 所以扩大到 >= 5000，并且建模时只用评测数足够多、不受这个门槛影响的游戏（见 etl_pipeline.py）。
MIN_PLAYERS_2018 = 5000
FIRST_ROUND_MIN_PLAYERS = 20000

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SteamSalesCourseProject/2.0"}

# 每个网站各自限速（秒）。appdetails 官方没写限额，社区经验是 5 分钟约 200 次，所以给 1.5 秒
MIN_INTERVAL = {
    "store.steampowered.com": 1.5,
    "api.steampowered.com": 0.8,
    "steamcharts.com": 1.5,
}
_last_call_time = {}


# ------------------------------------------------------------------------------
# 基础工具：限速请求、缓存、日志
# ------------------------------------------------------------------------------
def wait_for_host(host):
    gap = MIN_INTERVAL.get(host, 1.0)
    passed = time.time() - _last_call_time.get(host, 0)
    if passed < gap:
        time.sleep(gap - passed)
    _last_call_time[host] = time.time()


def http_get(url, params=None, max_retries=5, final_codes=(404,)):
    """带限速和退避重试的 GET。返回 (状态码, 响应对象)。final_codes 里的状态码不重试，直接返回。"""
    host = url.split("/")[2]
    for attempt in range(1, max_retries + 1):
        wait_for_host(host)
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            wait = 10 * attempt
            print(f"  [网络异常] {e.__class__.__name__}，{wait} 秒后重试 ({attempt}/{max_retries})", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code == 200 or resp.status_code in final_codes:
            return resp.status_code, resp
        # 429 被限速、5xx 服务器出错：等一会儿再试，等待时间逐次翻倍
        wait = 30 * (2 ** (attempt - 1))
        print(f"  [HTTP {resp.status_code}] {host}，{wait} 秒后重试 ({attempt}/{max_retries})", flush=True)
        time.sleep(wait)
    return -1, None


def cache_path(kind, name):
    p = CACHE_DIR / kind / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_cache(kind, name):
    p = cache_path(kind, name)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(kind, name, obj):
    p = cache_path(kind, name)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(p)  # 先写临时文件再改名，避免写到一半被中断留下坏文件


def write_crawl_log(task, appid, step, status, note=""):
    CRAWL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CRAWL_LOG_FILE.exists()
    with open(CRAWL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["logged_at", "task", "appid", "step", "status", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), task, appid, step, status, note])


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}  # 检查点只用于记录进度，读坏了不影响数据（数据在缓存里）
    return {}


def save_checkpoint(task, done_count, last_appid, status):
    cp = load_checkpoint()
    cp[task] = {
        "done_count": done_count,
        "last_appid": last_appid,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 坑：两个采集进程同时写这个文件，一个进程读到了另一个写了一半的内容，JSON 解析失败直接崩溃。
    # 改成先写临时文件再整体替换（替换是原子操作），读的一方要么看到旧文件，要么看到新文件。
    tmp = CHECKPOINT_FILE.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    tmp.replace(CHECKPOINT_FILE)


# ------------------------------------------------------------------------------
# 三个数据源
# ------------------------------------------------------------------------------
def fetch_appdetails(appid):
    """商店详情。下架的游戏会返回 success=false，也照样缓存，避免下次重复请求。"""
    cached = load_cache("appdetails", appid)
    if cached is not None:
        return cached
    code, resp = http_get("https://store.steampowered.com/api/appdetails",
                          params={"appids": appid, "cc": "us", "l": "english"})
    if code != 200:
        return None
    payload = resp.json() or {}
    result = payload.get(str(appid), {"success": False})
    result["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_cache("appdetails", appid, result)
    return result


def fetch_reviews(appid, end_ts, start_ts=1, languages=("all",), purchase_type=1, sample_size=0):
    """
    统计 [start_ts, end_ts] 之间创建的评测。
    purchase_type: 0=只算在 Steam 买的, 1=全部（含激活码、赠送）, 2=非 Steam 购买
    sample_size > 0 时顺带拿回这一时间段内"最新的" sample_size 条评测，只保留时长等字段，不保存评测正文。
    """
    name = f"{appid}_{start_ts}_{end_ts}_{'-'.join(languages)}_p{purchase_type}_n{sample_size}"
    cached = load_cache("reviews", name)
    if cached is not None:
        return cached
    query = {
        "appid": int(appid),
        "num_per_page": int(sample_size),
        "languages": list(languages),
        "purchase_type": int(purchase_type),
        "filter": 1,  # 1 = 按时间倒序；用默认的"最有帮助"会自动放宽日期范围
        "filter_offtopic_activity": False,  # 不过滤"刷差评"期间的评测，泄露玩家数同样包含这些人
        "date_range_start": int(start_ts),
        "date_range_end": int(end_ts),
    }
    code, resp = http_get("https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/",
                          params={"input_json": json.dumps(query)})
    if code != 200:
        return None
    body = resp.json().get("response", {})
    summary = body.get("query_summary", {})
    samples = []
    for r in body.get("reviews", []):
        a = r.get("author", {})
        samples.append({
            "timestamp_created": r.get("timestamp_created"),
            "playtime_at_review": a.get("playtime_at_review"),
            "playtime_forever": a.get("playtime_forever"),
            "language": r.get("language"),
            "voted_up": r.get("voted_up"),
            "steam_purchase": r.get("steam_purchase"),
            "received_for_free": r.get("received_for_free"),
        })
    result = {
        "total_reviews": summary.get("total_reviews"),
        "total_positive": summary.get("total_positive"),
        "total_negative": summary.get("total_negative"),
        "samples": samples,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_cache("reviews", name, result)
    return result


# Steam 用户评测系统 2013 年 11 月上线，二分查找的下界取 2013-10-01
REVIEW_SYSTEM_START_TS = 1380585600


def fetch_first_review_ts(appid, end_ts):
    """
    二分查找这款游戏"第一条评测"出现的日期（精确到天）。
    为什么需要：抢先体验游戏转正后，商店 release_date 显示的是正式版日期，可能晚于销量公布日，
    但它在抢先体验期间早就在卖了。第一条评测的日期是更可靠的"实际上架日期"。
    每次探测只要汇总数（num_per_page=0），大约 12 次请求。
    """
    cached = load_cache("first_review", f"{appid}_{end_ts}")
    if cached is not None:
        return cached.get("first_review_ts")
    lo, hi = REVIEW_SYSTEM_START_TS, int(end_ts)
    total = fetch_reviews(appid, hi, start_ts=lo)
    if total is None:
        return None
    if not total["total_reviews"]:
        save_cache("first_review", f"{appid}_{end_ts}", {"first_review_ts": None})
        return None
    day = 86400
    while hi - lo > day:
        # 坑：区间剩 1~2 天时 (hi-lo)//(2*day) = 0，mid 等于 lo，循环永远不结束；至少前进一天
        mid = lo + max(day, ((hi - lo) // (2 * day)) * day)
        res = fetch_reviews(appid, mid, start_ts=REVIEW_SYSTEM_START_TS)
        if res is None:
            return None
        if res["total_reviews"]:
            hi = mid
        else:
            lo = mid
    save_cache("first_review", f"{appid}_{end_ts}", {"first_review_ts": hi})
    return hi


MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def fetch_steamcharts(appid):
    """解析 SteamCharts 页面里的月度表：月份、月平均在线、月峰值。没有收录的游戏返回空列表。"""
    cached = load_cache("steamcharts", appid)
    if cached is not None:
        return cached
    # 坑：SteamCharts 没收录的游戏不是返回 404，而是返回 500 加一个首页标题的通用页面。
    # 第一次没区分，每款小游戏都白等了 3 分多钟的重试。现在 500 时先看页面标题再决定。
    code, resp = http_get(f"https://steamcharts.com/app/{appid}", final_codes=(404, 500))
    if code == 500 and "Tracking What's Played" not in resp.text:
        return None  # 真正的服务器错误，下次再试
    rows = []
    if code == 200:
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", resp.text, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) < 5:
                continue
            parts = cells[0].split()
            if len(parts) != 2 or parts[0] not in MONTHS:
                continue  # 跳过 "Last 30 Days" 这一行
            try:
                avg = float(cells[1].replace(",", ""))
                peak = int(cells[4].replace(",", ""))
            except ValueError:
                continue
            rows.append({"month": f"{parts[1]}-{MONTHS[parts[0]]:02d}", "avg_players": avg, "peak_players": peak})
    elif code == -1:
        return None
    result = {"status": code, "rows": rows, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_cache("steamcharts", appid, result)
    return result


# ------------------------------------------------------------------------------
# 读输入表
# ------------------------------------------------------------------------------
def read_ars_table():
    """
    Ars 原始 CSV 的坑：换行符是老式 Mac 的 \\r，文件编码是 Mac Roman 里混了 UTF-8，
    所以游戏名可能有乱码。我们只用 appid 关联，名字以 appdetails 为准，乱码不影响结果。
    """
    rows = []
    with open(ARS_CSV, "r", encoding="mac_roman", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for rec in reader:
            if len(rec) < 3 or not rec[2].strip().isdigit():
                continue
            players = int(rec[1].replace(",", "").strip())
            rows.append({"appid": int(rec[2]), "title": rec[0].strip(), "players": players})
    return rows


def parse_report_date(text):
    return dt.datetime.strptime(text.strip(), "%b %d, %Y").date()


def parse_k_number(text):
    """把 '12.4M'、'76.3K'、'893' 这类写法转成整数。"""
    t = text.strip().upper().replace(",", "")
    if not t:
        return None
    mult = 1
    if t.endswith("K"):
        mult, t = 1000, t[:-1]
    elif t.endswith("M"):
        mult, t = 1000000, t[:-1]
    return int(round(float(t) * mult))


def read_gamalytic_table():
    rows = []
    with open(GAMALYTIC_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def end_of_day_ts(d):
    return int(dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=dt.timezone.utc).timestamp())


def is_paid_game(details):
    """付费买断制游戏：接口成功、类型是 game、不是免费、有美元价格。"""
    if not details or not details.get("success"):
        return False, "DELISTED_OR_UNAVAILABLE"
    d = details.get("data", {})
    if d.get("type") != "game":
        return False, f"TYPE_{str(d.get('type')).upper()}"
    if d.get("is_free"):
        return False, "FREE_GAME"
    if not d.get("price_overview"):
        return False, "NO_PRICE"
    return True, "OK"


# ------------------------------------------------------------------------------
# 任务
# ------------------------------------------------------------------------------
def crawl_one_game(task, appid, end_ts, extra_split_ts=None):
    """对一款游戏依次抓：商店详情 -> 评测(全部+100条样本) -> 中文评测数 -> Steam内购买评测数 -> SteamCharts"""
    details = fetch_appdetails(appid)
    if details is None:
        write_crawl_log(task, appid, "appdetails", "FAILED")
        return False
    ok, reason = is_paid_game(details)
    if not ok:
        write_crawl_log(task, appid, "appdetails", "SKIPPED", reason)
        return True  # 免费/下架的游戏不再抓后面的接口，ETL 里再统一记入清洗日志

    steps = [
        ("reviews_all", lambda: fetch_reviews(appid, end_ts, sample_size=100)),
        ("reviews_schinese", lambda: fetch_reviews(appid, end_ts, languages=("schinese",))),
        ("reviews_steam_purchase", lambda: fetch_reviews(appid, end_ts, purchase_type=0)),
        ("steamcharts", lambda: fetch_steamcharts(appid)),
    ]
    if extra_split_ts is not None:
        # 跨时代游戏：多统计一次"政策上线之前"的评测数，用于新旧两段分别乘不同的 k
        steps.append(("reviews_before_policy", lambda: fetch_reviews(appid, extra_split_ts)))
    for step_name, fn in steps:
        if fn() is None:
            write_crawl_log(task, appid, step_name, "FAILED")
            return False
    write_crawl_log(task, appid, "all", "OK")
    return True


def parse_args(argv):
    opts = {"limit": None, "shard": (0, 1), "extend": False, "reverse": False}
    i = 0
    while i < len(argv):
        if argv[i] == "--extend":
            opts["extend"] = True; i += 1
        elif argv[i] == "--reverse":
            opts["reverse"] = True; i += 1
        elif argv[i] == "--limit":
            opts["limit"] = int(argv[i + 1]); i += 2
        elif argv[i] == "--shard":
            a, b = argv[i + 1].split("/"); opts["shard"] = (int(a), int(b)); i += 2
        else:
            i += 1
    return opts


def task_train2018(opts):
    # 第一轮（>= 2 万）和扩展轮（5000 ~ 2 万）各自用固定种子打乱：
    # 这样只跑完前 N 款时也是随机样本；扩展轮单独排序，不打乱第一轮已经跑过的顺序
    ars = read_ars_table()
    if opts.get("extend"):
        pool = [r for r in ars if MIN_PLAYERS_2018 <= r["players"] < FIRST_ROUND_MIN_PLAYERS]
    else:
        pool = [r for r in ars if r["players"] >= FIRST_ROUND_MIN_PLAYERS]
    random.Random(2018).shuffle(pool)
    if opts.get("reverse"):
        # 第二个进程从列表另一头往回抓，两个进程在中间会合；已缓存的请求会直接跳过
        pool = pool[::-1]
    if opts["limit"]:
        pool = pool[:opts["limit"]]
    shard_i, shard_n = opts["shard"]
    pool = [r for idx, r in enumerate(pool) if idx % shard_n == shard_i]
    task = f"train2018{'_extend' if opts.get('extend') else ''}{'_rev' if opts.get('reverse') else ''}_shard{shard_i}of{shard_n}"
    print(f"[{task}] 待处理 {len(pool)} 款游戏", flush=True)
    done = 0
    t0 = time.time()
    for r in pool:
        crawl_one_game(task, r["appid"], CUTOFF_2018_TS)
        done += 1
        if done % 20 == 0 or done == len(pool):
            save_checkpoint(task, done, r["appid"], "RUNNING" if done < len(pool) else "COMPLETED")
            speed = (time.time() - t0) / done
            print(f"[{task}] {done}/{len(pool)}，平均 {speed:.1f} 秒/款，预计还要 {speed * (len(pool) - done) / 60:.0f} 分钟", flush=True)


def task_first_review(opts):
    """对"商店发售日晚于截止日、但截止日前已有评测"的样本，补查第一条评测日期。"""
    jobs = []
    pool_ids = {r["appid"] for r in read_ars_table() if r["players"] >= MIN_PLAYERS_2018}
    for a in sorted({int(p.stem) for p in (CACHE_DIR / "appdetails").glob("*.json")}):
        if a in pool_ids:
            jobs.append((a, CUTOFF_2018_TS))
    for r in read_gamalytic_table():
        jobs.append((int(r["appid"]), end_of_day_ts(parse_report_date(r["report_date_text"]))))
    done = 0
    for appid, end_ts in jobs:
        d = load_cache("appdetails", appid)
        ok, _ = is_paid_game(d)
        if not ok:
            continue
        rel = parse_release_date(d["data"].get("release_date", {}).get("date", ""))
        cutoff = dt.datetime.fromtimestamp(end_ts, dt.timezone.utc).date()
        if rel is not None and rel <= cutoff:
            continue
        ts = fetch_first_review_ts(appid, end_ts)
        done += 1
        print(f"[first_review] {appid} 商店发售日 {rel}，截止 {cutoff}，第一条评测 "
              f"{dt.datetime.fromtimestamp(ts, dt.timezone.utc).date() if ts else '无'}", flush=True)
    print(f"[first_review] 共补查 {done} 款", flush=True)


# ------------------------------------------------------------------------------
# 官方公告扫描：找"卖了多少份"的原句（设计文档第 5 节的真值获取流程）
# ------------------------------------------------------------------------------
# 坑：第一版用一个大正则直接在全文里匹配"关键词…数字…"，遇到几万字的长公告发生灾难性回溯，进程卡死。
# 改成先按句号/换行切句，再逐句做两个简单判断，耗时和文本长度成正比。
SALES_WORD = re.compile(r"\b(sold|sales|copies|units|players)\b", re.I)
HAS_NUMBER = re.compile(r"\d")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def fetch_news(appid):
    cached = load_cache("news", appid)
    if cached is not None:
        return cached
    code, resp = http_get("https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/",
                          params={"appid": appid, "count": 100, "maxlength": 0, "format": "json"})
    if code != 200:
        return None
    items = []
    for it in resp.json().get("appnews", {}).get("newsitems", []):
        items.append({k: it.get(k) for k in ["gid", "title", "url", "author", "feedname", "date", "contents"]})
    result = {"items": items, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_cache("news", appid, result)
    return result


def task_news_scan(opts):
    """
    对真值集里每款游戏拉最近 100 条公告，只看开发者官方公告（feedname = steam_community_announcements），
    把同时出现"销量类关键词"和数字的句子原样摘出来，写 data/raw/news_sales_sentences.csv。
    后续抽取和人工核对见 src/news_extract_audit.py。
    """
    rows = []
    appids = sorted({int(r["appid"]) for r in read_gamalytic_table()})
    for appid in appids:
        news = fetch_news(appid)
        if not news:
            continue
        for it in news["items"]:
            if it.get("feedname") != "steam_community_announcements":
                continue
            text = re.sub(r"\[/?[a-z0-9*=_ ]+\]|<[^>]+>|https?://\S+", " ", it.get("contents") or "")
            text = re.sub(r"[ \t]+", " ", text)
            for sent in SENTENCE_SPLIT.split(it.get("title", "") + ". " + text):
                sent = sent.strip()
                if 12 <= len(sent) <= 400 and SALES_WORD.search(sent) and HAS_NUMBER.search(sent):
                    rows.append({"appid": appid, "news_gid": it["gid"],
                                 "news_date": dt.datetime.fromtimestamp(it["date"], dt.timezone.utc).strftime("%Y-%m-%d"),
                                 "title": it.get("title", ""), "url": it.get("url", ""), "sentence": sent})
    with open(RAW_DIR / "news_sales_sentences.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["appid", "news_gid", "news_date", "title", "url", "sentence"])
        w.writeheader()
        w.writerows(rows)
    print(f"[news_scan] {len(appids)} 款游戏，摘出 {len(rows)} 个候选句子", flush=True)


def load_policy_ts():
    p = RAW_DIR / "policy_change_detected.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return int(json.load(f)["policy_ts"])
    return None


def task_truth(opts):
    rows = read_gamalytic_table()
    policy_ts = load_policy_ts()
    if policy_ts is None:
        print("请先运行 policy_scan，确定评测提示政策的上线时间", flush=True)
        return
    task = "truth_gamalytic"
    done = 0
    for r in rows:
        appid = int(r["appid"])
        end_ts = end_of_day_ts(parse_report_date(r["report_date_text"]))
        split_ts = policy_ts if end_ts > policy_ts else None
        crawl_one_game(task, appid, end_ts, extra_split_ts=split_ts)
        done += 1
        if done % 10 == 0 or done == len(rows):
            save_checkpoint(task, done, appid, "RUNNING" if done < len(rows) else "COMPLETED")
            print(f"[{task}] {done}/{len(rows)}", flush=True)


def month_starts(y0, m0, y1, m1):
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def task_policy_scan(opts):
    """
    找"游玩后提示写评测"是哪个月上线的：挑 40 款 2016 年以前发售、2018 年玩家数最多的付费游戏，
    统计 2018-07 到 2020-12 每个月新增的评测数。老游戏的销量在这段时间里大体平稳，
    如果某个月评测数突然整体跳升，那就是政策上线的月份。
    """
    pool = sorted([r for r in read_ars_table() if r["players"] >= 500000], key=lambda r: -r["players"])
    picked = []
    for r in pool:
        details = fetch_appdetails(r["appid"])
        ok, _ = is_paid_game(details)
        if not ok:
            continue
        rel = parse_release_date(details["data"].get("release_date", {}).get("date", ""))
        if rel is None or rel.year > 2016:
            continue
        picked.append(r["appid"])
        if len(picked) >= 40:
            break
    months = month_starts(2018, 7, 2021, 1)
    out_rows = []
    for appid in picked:
        for a, b in zip(months[:-1], months[1:]):
            res = fetch_reviews(appid, int(b.timestamp()) - 1, start_ts=int(a.timestamp()))
            if res is None:
                continue
            out_rows.append([appid, a.strftime("%Y-%m"), res["total_reviews"]])
        print(f"[policy_scan] {appid} 完成", flush=True)
    with open(RAW_DIR / "review_monthly_policy_scan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["appid", "month", "new_reviews"])
        w.writerows(out_rows)
    print(f"[policy_scan] 写出 {len(out_rows)} 行到 data/raw/review_monthly_policy_scan.csv", flush=True)


# ------------------------------------------------------------------------------
# 导出：把缓存整理成 ETL 读取的 CSV
# ------------------------------------------------------------------------------
def parse_release_date(text):
    text = (text or "").strip()
    for fmt in ("%d %b, %Y", "%b %d, %Y", "%d %B, %Y", "%B %d, %Y", "%b %Y", "%B %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    m = re.search(r"(20\d\d|19\d\d)", text)
    if m:
        return dt.date(int(m.group(1)), 7, 1)  # 只有年份时按年中处理，并在 ETL 里标记
    return None


def summarize_samples(samples):
    hrs = sorted([s["playtime_at_review"] / 60.0 for s in samples if s.get("playtime_at_review") is not None])
    if not hrs:
        return 0, None, None
    n = len(hrs)
    median = hrs[n // 2] if n % 2 else (hrs[n // 2 - 1] + hrs[n // 2]) / 2
    return n, round(median, 3), round(sum(hrs) / n, 3)


def export_game_rows(appid, end_ts, cohort, split_ts=None):
    details = load_cache("appdetails", appid)
    rev_all = load_cache("reviews", f"{appid}_1_{end_ts}_all_p1_n100")
    rev_sc = load_cache("reviews", f"{appid}_1_{end_ts}_schinese_p1_n0")
    rev_steam = load_cache("reviews", f"{appid}_1_{end_ts}_all_p0_n0")
    rev_pre = load_cache("reviews", f"{appid}_1_{split_ts}_all_p1_n0") if split_ts else None
    if not (details and rev_all and rev_sc and rev_steam):
        return None
    n, med, mean = summarize_samples(rev_all["samples"])
    fr = load_cache("first_review", f"{appid}_{end_ts}")
    first_review_date = ""
    if fr and fr.get("first_review_ts"):
        first_review_date = dt.datetime.fromtimestamp(fr["first_review_ts"], dt.timezone.utc).strftime("%Y-%m-%d")
    return {
        "appid": appid, "cohort": cohort,
        "cutoff_date": dt.datetime.fromtimestamp(end_ts, dt.timezone.utc).strftime("%Y-%m-%d"),
        "cutoff_ts": end_ts,
        "total_reviews": rev_all["total_reviews"], "total_positive": rev_all["total_positive"],
        "total_negative": rev_all["total_negative"],
        "schinese_reviews": rev_sc["total_reviews"], "steam_purchase_reviews": rev_steam["total_reviews"],
        "reviews_before_policy": rev_pre["total_reviews"] if rev_pre else "",
        "sample_n": n, "sample_median_playtime_at_review_hrs": med, "sample_mean_playtime_at_review_hrs": mean,
        "first_review_date": first_review_date,
        "fetched_at": rev_all["fetched_at"],
    }


def appdetails_row(appid):
    d0 = load_cache("appdetails", appid)
    if d0 is None:
        return None
    d = d0.get("data", {}) if d0.get("success") else {}
    p = d.get("price_overview") or {}
    rel_text = d.get("release_date", {}).get("date", "")
    rel = parse_release_date(rel_text)
    langs = d.get("supported_languages", "") or ""
    return {
        "appid": appid, "success": int(bool(d0.get("success"))), "name": d.get("name", ""),
        "app_type": d.get("type", ""), "is_free": int(bool(d.get("is_free"))),
        "release_date_text": rel_text, "release_date": rel.isoformat() if rel else "",
        "currency": p.get("currency", ""),
        "initial_price_usd": p.get("initial", 0) / 100.0 if p else "",
        "final_price_usd": p.get("final", 0) / 100.0 if p else "",
        "discount_percent": p.get("discount_percent", "") if p else "",
        "genres": "|".join(g["description"] for g in d.get("genres", [])),
        "categories": "|".join(c["description"] for c in d.get("categories", [])),
        "developers": "|".join(d.get("developers", []) or []),
        "publishers": "|".join(d.get("publishers", []) or []),
        "metacritic": (d.get("metacritic") or {}).get("score", ""),
        "achievements_total": (d.get("achievements") or {}).get("total", ""),
        "supports_schinese": int("Simplified Chinese" in langs),
        "fetched_at": d0.get("fetched_at", ""),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[export] {path.relative_to(BASE_DIR)}: {len(rows)} 行", flush=True)


def task_export(opts):
    ars = read_ars_table()
    write_csv(RAW_DIR / "ars_players_2018.csv", ars, ["appid", "title", "players"])

    policy_ts = load_policy_ts()
    gam = read_gamalytic_table()
    truth_rows = []
    for r in gam:
        d = parse_report_date(r["report_date_text"])
        truth_rows.append({
            "appid": int(r["appid"]), "game": r["game"], "report_date": d.isoformat(),
            "reported_sales": parse_k_number(r["reported_units_text"]),
            "reported_sales_text": r["reported_units_text"], "source_url": r["source_url"],
            "gamalytic_estimate": parse_k_number(r["gamalytic_estimate_text"]),
            "benchmark_source": "https://gamalytic.com/blog/how-to-accurately-estimate-steam-sales",
        })
    write_csv(RAW_DIR / "recent_sales_truth.csv", truth_rows, list(truth_rows[0].keys()))

    # 商店详情：2018 抽样池 + 真值集里抓过的全部 appid
    all_ids = sorted({p.stem for p in (CACHE_DIR / "appdetails").glob("*.json")}, key=int)
    app_rows = [x for x in (appdetails_row(int(a)) for a in all_ids) if x]
    write_csv(RAW_DIR / "steam_appdetails.csv", app_rows, list(app_rows[0].keys()))

    review_rows = []
    pool_ids = {r["appid"] for r in ars if r["players"] >= MIN_PLAYERS_2018}
    for a in all_ids:
        a = int(a)
        if a in pool_ids:
            row = export_game_rows(a, CUTOFF_2018_TS, "train_2018")
            if row:
                review_rows.append(row)
    for r in truth_rows:
        end_ts = end_of_day_ts(dt.date.fromisoformat(r["report_date"]))
        split_ts = policy_ts if (policy_ts and end_ts > policy_ts) else None
        row = export_game_rows(r["appid"], end_ts, "truth", split_ts)
        if row:
            review_rows.append(row)
    write_csv(RAW_DIR / "steam_reviews_at_cutoff.csv", review_rows, list(review_rows[0].keys()))

    chart_rows = []
    for p in sorted((CACHE_DIR / "steamcharts").glob("*.json"), key=lambda x: int(x.stem)):
        obj = load_cache("steamcharts", p.stem)
        for m in obj.get("rows", []):
            chart_rows.append({"appid": int(p.stem), **m})
    write_csv(RAW_DIR / "steamcharts_monthly.csv", chart_rows, ["appid", "month", "avg_players", "peak_players"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    opts = parse_args(sys.argv[2:])
    {"train2018": task_train2018, "truth": task_truth, "policy_scan": task_policy_scan,
     "first_review": task_first_review, "news_scan": task_news_scan, "export": task_export}[cmd](opts)
