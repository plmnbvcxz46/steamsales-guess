# -*- coding: utf-8 -*-
"""
补充采集 (src/crawler_extra.py)：各语言评测数 + IsThereAnyDeal 价格历史 / 慈善包

用法（在项目根目录运行）：
    python src/crawler_extra.py jobs                       # 从数据库导出采集清单 data/raw/extra_crawl_jobs.csv（本机运行）
    python src/crawler_extra.py langs   [--cohort truth]   # 截止日前各语言的评测数
    python src/crawler_extra.py itad    [--cohort truth]   # ITAD：appid -> ITAD id，价格历史（美/中/土/巴），慈善包
    python src/crawler_extra.py truth_v2                   # 新真值集：按每条销量的公布日期抓评测汇总、样本、SteamCharts 等
    python src/crawler_extra.py export                     # 缓存 -> data/raw/ 下的 CSV

新真值集 data/raw/official_sales_truth_v2.csv（两轮收集合并、分层后的结果，见 src/truth_v2_merge.py）
在采集清单里记为 cohort = truth_v2，和旧的 Gamalytic 真值集（truth）分开。

为什么要这两类数据：
- 现有模型里唯一的地区信息是"简中评测占比"。不同语言玩家写评测的习惯差别很大，k 也就不同。
- 现有价格是 2026 年抓取时的商店原价，不是首发价，也不知道打过几次折、进过几次慈善包。
  深度折扣和慈善包带来的玩家很少写评测，会把 k 推高。

ITAD 的 Key 放在项目根目录 .env 里：ITAD_API_KEY=xxxx（.env 已加入 .gitignore）。
复用 crawler.py 的限速、退避重试、请求级缓存和采集日志，断了重新运行会跳过已缓存的请求。
"""
import os
import sys
import csv
import json
import time
import datetime as dt

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crawler as C  # noqa: E402

JOBS_CSV = C.RAW_DIR / "extra_crawl_jobs.csv"
TRUTH_V2_CSV = C.RAW_DIR / "official_sales_truth_v2.csv"  # 由 src/truth_v2_merge.py 合并两轮收集结果生成
TRUTH_V2_SKIP_TIERS = {"X_wiki", "X_not_sales", "X_num_mismatch"}  # 明显不能用的不抓

# 现有数据已经有 all（总数）和 schinese，这里补其余主要语言。
# Steam 的语言代码：brazilian = 巴西葡语，latam = 拉美西语，spanish = 西班牙西语，koreana = 韩语
LANGUAGES = ["english", "russian", "brazilian", "spanish", "latam", "german", "french",
             "japanese", "koreana", "polish", "turkish", "tchinese"]

# 评测接口：crawler.py 里给的是 0.8 秒一次（保守值）。2018 样本 1795 款 × 12 种语言约 2.2 万次请求，要跑 5 个小时，
# 而之前几千次请求一次 429 都没遇到，瓶颈完全是我们自己设的间隔。补充采集放快到 0.3 秒，遇到 429 仍会自动退避。
C.MIN_INTERVAL["api.steampowered.com"] = float(os.environ.get("STEAM_API_INTERVAL", "0.3"))

ITAD_API = "https://api.isthereanydeal.com"
ITAD_STEAM_SHOP = 61
# 实测（2026-10-20）：CN/TR/BR 有 2016~2017 年起的人民币/里拉/雷亚尔价格；
# RU 不支持（返回的是美区价格），AR 只有 2026 年以后，所以不抓这两个
ITAD_COUNTRIES = ["US", "CN", "TR", "BR"]
# 限额：账号验证邮箱后是 5 分钟 1000 次（ITAD 应用页面 Limits 一栏可以看到）。
# 坑：一开始用 Steam 登录、没绑邮箱，限额低得多，0.35 秒一次跑了约 180 次就 429；绑定并验证邮箱后才解锁。
# 0.4 秒一次约 750 次 / 5 分钟，留出余量；再遇到 429 就按响应头 Retry-After 等够时间，并把间隔放大 25%
C.MIN_INTERVAL["api.isthereanydeal.com"] = 0.4


def load_env():
    p = C.BASE_DIR / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("ITAD_API_KEY")
    if not key:
        sys.exit("缺少 ITAD_API_KEY：在项目根目录 .env 里写一行 ITAD_API_KEY=你的key")
    return key


def read_jobs(cohort=None):
    with open(JOBS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if cohort:
        rows = [r for r in rows if r["cohort"] == cohort]
    return rows


def parse_args(argv):
    opts = {"cohort": None}
    if "--cohort" in argv:
        opts["cohort"] = argv[argv.index("--cohort") + 1]
    return opts


# ------------------------------------------------------------------------------
# 采集清单：只取参与建模的样本（2018 年评测数 >= 500 的，加上真值集）
# ------------------------------------------------------------------------------
def task_jobs(opts):
    import sqlite3
    con = sqlite3.connect(C.BASE_DIR / "data" / "steam_sales_dw.db")
    rows = con.execute("""
        SELECT d.appid, d.cohort, r.cutoff_ts, d.cutoff_date
        FROM dwd_game_sample d
        JOIN ods_steam_reviews_at_cutoff r
          ON r.appid = d.appid AND r.cohort = d.cohort AND r.cutoff_date = d.cutoff_date
        WHERE d.model_eligible = 1
        ORDER BY d.cohort DESC, d.appid""").fetchall()
    # 新真值集：每条销量按 as_of_date 当天 23:59:59 UTC 截止；同一款游戏同一天的多条只抓一次
    seen = set()
    with open(TRUTH_V2_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tier"] in TRUTH_V2_SKIP_TIERS:
                continue
            d = dt.date.fromisoformat(r["as_of_date"])  # 数字对应的时间点（首日/首周等已换算），不一定是发帖日
            key = (int(r["appid"]), C.end_of_day_ts(d))
            if key not in seen:
                seen.add(key)
                rows.append((key[0], "truth_v2", key[1], d.isoformat()))
    with open(JOBS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["appid", "cohort", "cutoff_ts", "cutoff_date"])
        w.writerows(rows)
    print(f"[jobs] {len(rows)} 个样本写入 {JOBS_CSV.relative_to(C.BASE_DIR)}", flush=True)


# ------------------------------------------------------------------------------
# 各语言评测数：和现有的 all / schinese 用同一个接口、同一个截止时间
# ------------------------------------------------------------------------------
def task_langs(opts):
    jobs = read_jobs(opts["cohort"])
    task = f"langs_{opts['cohort'] or 'all'}"
    print(f"[{task}] {len(jobs)} 个样本 × {len(LANGUAGES)} 种语言", flush=True)
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        appid, end_ts = int(j["appid"]), int(j["cutoff_ts"])
        failed = [lang for lang in LANGUAGES if C.fetch_reviews(appid, end_ts, languages=(lang,)) is None]
        C.write_crawl_log(task, appid, "langs", "FAILED" if failed else "OK", ",".join(failed))
        if i % 20 == 0 or i == len(jobs):
            C.save_checkpoint(task, i, appid, "RUNNING" if i < len(jobs) else "COMPLETED")
            speed = (time.time() - t0) / i
            print(f"[{task}] {i}/{len(jobs)}，{speed:.1f} 秒/款，预计还要 {speed * (len(jobs) - i) / 60:.0f} 分钟", flush=True)


# ------------------------------------------------------------------------------
# IsThereAnyDeal
# ------------------------------------------------------------------------------
def itad_get(path, key, params, max_retries=8):
    host = "api.isthereanydeal.com"
    for attempt in range(1, max_retries + 1):
        C.wait_for_host(host)
        try:
            resp = requests.get(ITAD_API + path, params={"key": key, **params}, headers=C.HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  [ITAD 网络异常] {e.__class__.__name__}，{10 * attempt} 秒后重试", flush=True)
            time.sleep(10 * attempt)
            continue
        if resp.status_code == 200:
            return 200, resp.json()
        if resp.status_code in (400, 404):
            return resp.status_code, None
        wait = int(resp.headers.get("Retry-After", 60)) + 2 if resp.status_code == 429 else 30 * attempt
        if resp.status_code == 429:
            C.MIN_INTERVAL[host] = min(10.0, C.MIN_INTERVAL[host] * 1.25)
        print(f"  [ITAD HTTP {resp.status_code}] 等 {wait} 秒；之后间隔 {C.MIN_INTERVAL[host]:.1f} 秒", flush=True)
        time.sleep(wait)
    return -1, None


def itad_lookup(appids, key):
    """批量把 Steam appid 换成 ITAD 的游戏 id。一次最多 200 个，结果按 appid 单独缓存。"""
    todo = [a for a in appids if C.load_cache("itad_lookup", a) is None]
    for k in range(0, len(todo), 200):
        batch = todo[k:k + 200]
        C.wait_for_host("api.isthereanydeal.com")
        for attempt in range(5):
            try:
                resp = requests.post(f"{ITAD_API}/lookup/id/shop/{ITAD_STEAM_SHOP}/v1", params={"key": key},
                                     json=[f"app/{a}" for a in batch], headers=C.HEADERS, timeout=60)
                if resp.status_code == 200:
                    break
                print(f"  [ITAD lookup HTTP {resp.status_code}] {resp.text[:120]}", flush=True)
            except requests.RequestException as e:
                print(f"  [ITAD lookup 网络异常] {e.__class__.__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
        else:
            continue
        data = resp.json()
        for a in batch:
            C.save_cache("itad_lookup", a, {"itad_id": data.get(f"app/{a}"),
                                            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"[itad] lookup {min(k + 200, len(todo))}/{len(todo)}", flush=True)


def itad_history(itad_id, country, key):
    name = f"{itad_id}_{country}"
    cached = C.load_cache("itad_history", name)
    if cached is not None:
        return cached
    code, data = itad_get("/games/history/v2", key, {"id": itad_id, "country": country,
                                                     "shops": ITAD_STEAM_SHOP, "since": "2005-01-01T00:00:00Z"})
    if code == -1:
        return None
    result = {"status": code, "rows": data if isinstance(data, list) else [],
              "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.save_cache("itad_history", name, result)
    return result


def itad_bundles(itad_id, key):
    cached = C.load_cache("itad_bundles", itad_id)
    if cached is not None:
        return cached
    code, data = itad_get("/games/bundles/v2", key, {"id": itad_id, "expired": "true"})
    if code == -1:
        return None
    result = {"status": code, "rows": data if isinstance(data, list) else [],
              "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.save_cache("itad_bundles", itad_id, result)
    return result


def task_itad(opts):
    key = load_env()
    jobs = read_jobs(opts["cohort"])
    appids = sorted({int(j["appid"]) for j in jobs})
    task = f"itad_{opts['cohort'] or 'all'}"
    print(f"[{task}] {len(appids)} 款游戏", flush=True)
    itad_lookup(appids, key)
    t0 = time.time()
    for i, a in enumerate(appids, 1):
        lk = C.load_cache("itad_lookup", a) or {}
        gid = lk.get("itad_id")
        if not gid:
            C.write_crawl_log(task, a, "lookup", "SKIPPED", "ITAD 没有收录")
            continue
        failed = [c for c in ITAD_COUNTRIES if itad_history(gid, c, key) is None]
        if itad_bundles(gid, key) is None:
            failed.append("bundles")
        C.write_crawl_log(task, a, "itad", "FAILED" if failed else "OK", ",".join(failed))
        if i % 50 == 0 or i == len(appids):
            C.save_checkpoint(task, i, a, "RUNNING" if i < len(appids) else "COMPLETED")
            speed = (time.time() - t0) / i
            print(f"[{task}] {i}/{len(appids)}，{speed:.1f} 秒/款，预计还要 {speed * (len(appids) - i) / 60:.0f} 分钟", flush=True)


# ------------------------------------------------------------------------------
# 新真值集的基础采集：和旧真值集同一套（商店详情、评测汇总+100 条样本、简中、Steam 内购买、SteamCharts、政策前评测数）
# ------------------------------------------------------------------------------
def task_truth_v2(opts):
    policy_ts = C.load_policy_ts()
    if policy_ts is None:
        sys.exit("缺少 data/raw/policy_change_detected.json")
    jobs = read_jobs("truth_v2")
    task = "truth_v2"
    print(f"[{task}] {len(jobs)} 个（游戏, 截止日）", flush=True)
    for i, j in enumerate(jobs, 1):
        appid, end_ts = int(j["appid"]), int(j["cutoff_ts"])
        ok = C.crawl_one_game(task, appid, end_ts, extra_split_ts=policy_ts if end_ts > policy_ts else None)
        d = C.load_cache("appdetails", appid)
        paid, _ = C.is_paid_game(d)
        if ok and paid:
            # 抢先体验转正后商店显示正式版日期，可能晚于公布日：补查第一条评测日期
            rel = C.parse_release_date(d["data"].get("release_date", {}).get("date", ""))
            if rel is None or rel > dt.date.fromisoformat(j["cutoff_date"]):
                C.fetch_first_review_ts(appid, end_ts)
        if i % 20 == 0 or i == len(jobs):
            C.save_checkpoint(task, i, appid, "RUNNING" if i < len(jobs) else "COMPLETED")
            print(f"[{task}] {i}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------------------
# 导出
# ------------------------------------------------------------------------------
def task_export(opts):
    jobs = read_jobs()
    lang_rows = []
    for j in jobs:
        appid, end_ts = int(j["appid"]), int(j["cutoff_ts"])
        for lang in LANGUAGES:
            r = C.load_cache("reviews", f"{appid}_1_{end_ts}_{lang}_p1_n0")
            if r is not None:
                lang_rows.append({"appid": appid, "cohort": j["cohort"], "cutoff_ts": end_ts,
                                  "cutoff_date": j["cutoff_date"], "language": lang,
                                  "total_reviews": r["total_reviews"], "total_positive": r["total_positive"]})
    C.write_csv(C.RAW_DIR / "review_language_counts.csv", lang_rows,
                ["appid", "cohort", "cutoff_ts", "cutoff_date", "language", "total_reviews", "total_positive"])

    hist_rows, bundle_rows = [], []
    # 价格历史只保留到这款游戏最晚的截止日为止（之后的价格用不到）。不筛的话 2018 样本会带上 2018~2026 的全部记录，
    # CSV 有七十多 MB；筛完只剩一小部分
    max_cut = {}
    for j in jobs:
        a = int(j["appid"])
        max_cut[a] = max(max_cut.get(a, 0), int(j["cutoff_ts"]))
    for a in sorted({int(j["appid"]) for j in jobs}):
        lk = C.load_cache("itad_lookup", a) or {}
        gid = lk.get("itad_id")
        if not gid:
            continue
        for c in ITAD_COUNTRIES:
            h = C.load_cache("itad_history", f"{gid}_{c}")
            for x in (h or {}).get("rows", []):
                if dt.datetime.fromisoformat(x.get("timestamp")).timestamp() > max_cut[a]:
                    continue
                deal = x.get("deal") or {}
                hist_rows.append({"appid": a, "itad_id": gid, "country": c, "timestamp": x.get("timestamp"),
                                  "currency": (deal.get("price") or {}).get("currency"),
                                  "price": (deal.get("price") or {}).get("amount"),
                                  "regular": (deal.get("regular") or {}).get("amount"),
                                  "cut": deal.get("cut")})
        b = C.load_cache("itad_bundles", gid)
        for x in (b or {}).get("rows", []):
            bundle_rows.append({"appid": a, "itad_id": gid, "bundle_id": x.get("id"), "title": x.get("title"),
                                "page": (x.get("page") or {}).get("name"), "publish": x.get("publish"),
                                "expiry": x.get("expiry"), "is_mature": x.get("isMature")})
    C.write_csv(C.RAW_DIR / "itad_price_history.csv", hist_rows,
                ["appid", "itad_id", "country", "timestamp", "currency", "price", "regular", "cut"])
    C.write_csv(C.RAW_DIR / "itad_bundles.csv", bundle_rows,
                ["appid", "itad_id", "bundle_id", "title", "page", "publish", "expiry", "is_mature"])

    # 新真值集：和 steam_reviews_at_cutoff.csv 同样的列，单独一张表
    policy_ts = C.load_policy_ts()
    v2_rows = []
    for j in read_jobs("truth_v2"):
        end_ts = int(j["cutoff_ts"])
        row = C.export_game_rows(int(j["appid"]), end_ts, "truth_v2",
                                 policy_ts if (policy_ts and end_ts > policy_ts) else None)
        if row:
            v2_rows.append(row)
    if v2_rows:
        C.write_csv(C.RAW_DIR / "steam_reviews_at_cutoff_truth_v2.csv", v2_rows, list(v2_rows[0].keys()))
    app_rows = [x for x in (C.appdetails_row(a) for a in sorted({int(j["appid"]) for j in read_jobs("truth_v2")})) if x]
    if app_rows:
        C.write_csv(C.RAW_DIR / "steam_appdetails_truth_v2.csv", app_rows, list(app_rows[0].keys()))
    chart_rows = []
    for a in sorted({int(j["appid"]) for j in read_jobs("truth_v2")}):
        obj = C.load_cache("steamcharts", a)
        for m in (obj or {}).get("rows", []):
            chart_rows.append({"appid": a, **m})
    C.write_csv(C.RAW_DIR / "steamcharts_monthly_truth_v2.csv", chart_rows, ["appid", "month", "avg_players", "peak_players"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    opts = parse_args(sys.argv[2:])
    {"jobs": task_jobs, "langs": task_langs, "itad": task_itad, "truth_v2": task_truth_v2,
     "export": task_export}[cmd](opts)
