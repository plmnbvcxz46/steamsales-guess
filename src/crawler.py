# -*- coding: utf-8 -*-
"""
Steam 多源数据采集模块 (src/crawler.py)
包含：
1. 商店详情接口 (store.steampowered.com/api/appdetails)
2. 新版官方评测接口 (api.steampowered.com/IUserReviewsService/GetAppReviews/v1/)
   —— 旧版 store.steampowered.com/appreviews 已弃用，新版支持按时间戳截断对齐 2018-07-01
3. 官方新闻接口 (api.steampowered.com/ISteamNews/GetNewsForApp/v2/)
   —— 用于扫描含 sold / copies / million 的销量里程碑公告
4. SteamSpy 接口 (steamspy.com/api.php) 与 SteamCharts 月度在线解析
   —— 严格遵守频率限制，绝不爬取 SteamDB（其 FAQ 明确禁止爬虫）
5. 本地 JSON 缓存、429 指数退避重试、检查点断点续传
"""
import os
import re
import time
import json
import pathlib
import requests

BASE_DIR = pathlib.Path("/tmp/steamsales-guess")
CACHE_DIR = BASE_DIR / "data" / "cache"
CHECKPOINT_FILE = CACHE_DIR / "crawler_checkpoint.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 统一请求头，模拟正常浏览器访问，避免被网关直接拦截
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SteamSalesAcademicETL/1.0"
}

# 2018-07-01 00:00:00 UTC 的 Unix 时间戳，用于和 Ars Technica 2018-07-01 玩家数快照严格对齐
TIMESTAMP_2018_07_01 = 1530403200


def load_checkpoint(task_name):
    """读取本地断点续传文件，避免爬虫中途报错或触发 429 后从头重跑"""
    if not CHECKPOINT_FILE.exists():
        return {"completed_appids": [], "last_cursor": "*", "updated_at": ""}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(task_name, {"completed_appids": [], "last_cursor": "*", "updated_at": ""})


def save_checkpoint(task_name, completed_appids, last_cursor="*"):
    """每爬完一款游戏就更新检查点，保证随时中断都能无缝续传"""
    all_cp = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            try:
                all_cp = json.load(f)
            except Exception:
                all_cp = {}
    all_cp[task_name] = {
        "completed_appids": list(completed_appids),
        "last_cursor": last_cursor,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_cp, f, ensure_ascii=False, indent=2)


def request_with_retry(url, params=None, sleep_sec=1.2, max_retries=4):
    """
    带限速与 HTTP 429 退避重试的 GET 请求函数。
    为什么不用多线程狂扫：Steam 匿名接口限额较严，触发 429 后如果不等待直接重试会被封 IP。
    """
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(sleep_sec)
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait_time = (2 ** attempt) * 5
                print(f"[警告] 触发 429 频率限制，退避等待 {wait_time} 秒后重试 (第 {attempt}/{max_retries} 次)...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == max_retries:
                print(f"[错误] 请求最终失败: {url}, 原因: {e}")
                return None
            wait_time = (2 ** attempt) * 2
            print(f"[提示] 请求异常 ({e})，{wait_time} 秒后重试...")
            time.sleep(wait_time)
    return None


def fetch_appdetails(appid, use_cache=True):
    """
    获取 Steam 商店详情 (store.steampowered.com/api/appdetails)
    统一用 cc=us 美元区价格，避免不同国家汇率造成价格特征不可比。
    """
    cache_path = CACHE_DIR / f"appdetails_{appid}.json"
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://store.steampowered.com/api/appdetails"
    resp = request_with_retry(url, params={"appids": appid, "cc": "us"}, sleep_sec=1.5)
    if resp is None:
        return None

    data = resp.json()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def fetch_reviews_aligned(appid, end_timestamp=TIMESTAMP_2018_07_01, max_pages=3, use_cache=True):
    """
    调用新版评测接口 IUserReviewsService/GetAppReviews/v1/ 获取截至指定日期的评测统计。
    注意：
    1. 汇总字段 total_reviews / total_positive / total_negative 只在第一页且 review_type=all 时返回；
    2. 用 date_range_end=1530403200 把评测时间严格卡在 2018-07-01 之前，与 2018 泄露玩家数对齐；
    3. 抽样前几页评测计算中文评测占比 (schinese + tchinese) 和写评测时的平均游玩时长 (playtime_at_review)。
    """
    cache_path = CACHE_DIR / f"reviews_{appid}_{end_timestamp}.json"
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
    cursor = "*"
    summary_info = {}
    sampled_reviews = []

    for page in range(1, max_pages + 1):
        params = {
            "appid": appid,
            "json": 1,
            "filter": "recent",
            "language": "all",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 100,
            "cursor": cursor,
            "date_range_end": end_timestamp,
            "filter_offtopic_activity": 0
        }
        resp = request_with_retry(url, params=params, sleep_sec=1.2)
        if resp is None:
            break
        payload = resp.json()
        if page == 1 and "query_summary" in payload:
            summary_info = payload["query_summary"]
        reviews_batch = payload.get("reviews", [])
        if not reviews_batch:
            break
        sampled_reviews.extend(reviews_batch)
        next_cursor = payload.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    # 统计抽样评测中的中文占比与写评测时游玩时长（接口返回单位是分钟，除以 60 转为小时）
    cn_count = sum(1 for r in sampled_reviews if r.get("language") in ("schinese", "tchinese"))
    playtimes = [
        r.get("author", {}).get("playtime_at_review", 0) / 60.0
        for r in sampled_reviews
        if r.get("author", {}).get("playtime_at_review", 0) > 0
    ]
    result = {
        "appid": appid,
        "end_timestamp": end_timestamp,
        "total_reviews": summary_info.get("total_reviews", 0),
        "total_positive": summary_info.get("total_positive", 0),
        "total_negative": summary_info.get("total_negative", 0),
        "sampled_count": len(sampled_reviews),
        "schinese_ratio": round(cn_count / len(sampled_reviews), 4) if sampled_reviews else 0.0,
        "avg_playtime_at_review_hrs": round(sum(playtimes) / len(playtimes), 2) if playtimes else 0.0,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def scan_news_for_sales_milestone(appid, count=50):
    """
    调用 ISteamNews/GetNewsForApp/v2/ 扫描官方公告，寻找包含销量里程碑的候选句子。
    为什么要保留原句和链接：大模型只能做“从给定原文抽取结构化字段”，绝不能凭记忆报销量。
    """
    url = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
    params = {"appid": appid, "maxlength": 0, "count": count, "format": "json"}
    resp = request_with_retry(url, params=params, sleep_sec=1.0)
    if resp is None:
        return []

    news_items = resp.json().get("appnews", {}).get("newsitems", [])
    pattern = re.compile(r"(sold|sales|copies|units|million|突破|销量|售出)", re.IGNORECASE)
    candidates = []
    for item in news_items:
        title = item.get("title", "")
        contents = item.get("contents", "")
        if pattern.search(title) or pattern.search(contents):
            candidates.append({
                "appid": appid,
                "gid": item.get("gid"),
                "title": title,
                "date": time.strftime("%Y-%m-%d", time.gmtime(item.get("date", 0))),
                "url": item.get("url", ""),
                "excerpt": (title + " | " + contents[:300]).replace("\n", " ")
            })
    return candidates


def fetch_steamspy_metadata(appid):
    """
    调用 SteamSpy 接口获取 average_forever（分钟）和 median_forever（分钟），用于信号 B 推算玩家数。
    遵守 SteamSpy 频率限制：普通请求每秒最多 1 次。
    """
    url = "https://steamspy.com/api.php"
    resp = request_with_retry(url, params={"request": "appdetails", "appid": appid}, sleep_sec=1.1)
    if resp is None:
        return None
    return resp.json()


if __name__ == "__main__":
    print("Steam 数据采集模块就绪。")
    print("支持离线缓存读取模式与在线增量断点续传模式。")
