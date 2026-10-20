# -*- coding: utf-8 -*-
"""
Steam 游戏销量估算系统 - Python ETL 管道 (src/etl_pipeline.py)

输入：crawler.py export、price_history_wayback.py、snapshot_collector.py、detect_policy_change.py 产出的 data/raw/*.csv
输出：本地 SQLite 数据仓库 data/steam_sales_dw.db（表结构与 sql/01_schema_mysql.sql 一一对应，可导入 MySQL）

分层：
    原始层 ODS  ：接口返回的数据原样落库（ods_*）
    明细层 DWD  ：清洗、关联、算好特征的样本宽表（dwd_game_sample），以及维度表（dim_*）
    结果层 ADS  ：模型输出的事实表与汇总表（fact_*、ads_*，由 models.py 写入）
运维表：etl_job_log（每个步骤一行日志）、etl_checkpoint（断点续传/增量水位线）、etl_clean_error_log（被拦截或标记的记录）

用法：
    python src/etl_pipeline.py                  # 全量重建 ODS/DWD/维度；快照表按水位线增量追加
    python src/etl_pipeline.py --demo-resume    # 额外演示一次"清洗到一半中断、从检查点续跑"
"""
import sys
import csv
import json
import math
import time
import sqlite3
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"

CUTOFF_2018 = "2018-07-01"
ARS_SOURCE_URL = "https://arstechnica.com/wp-content/uploads/2018/07/games_achievements_players_2018-07-01.csv"
# 2018 抽样门槛（与 crawler.py 一致）。为了避开截断偏差，建模只用评测数 >= MIN_REVIEWS_2018 的游戏，见 clean 步骤说明
MIN_PLAYERS_2018 = 5000
MIN_REVIEWS_2018 = 500
BATCH_SIZE = 200

# 新真值集（src/truth_v2_merge.py 合并两轮收集结果、按口径分层）：哪些层进明细层、哪些参与校准
# 第一次全量跑时 S2（标了 Steam、原句却没说平台）也参与了校准，结果它的误差是 S1 的两倍（融合中位 |ln 误差| 0.75 vs 0.37），
# 里面有不少"首日卖了多少"却按发帖日截止、或者把全平台当 Steam 的条目，把校准带偏。所以 S2 改为只作参考。
TRUTH_V2_TIERS_FIT = ("S1_steam_explicit", "Q_steam_split")
TRUTH_V2_TIERS_REF = ("S2_steam_labelled", "P_pc")  # 进明细层但不参与校准，只作参考（P 是 PC 全商店口径，Steam 销量 <= 这个数）
MIN_REVIEWS_TRUTH = 20  # 真值样本的最少评测数：评测太少时，评测数本身的随机波动就有 ±20% 以上，k 没有意义
# 各语言评测数（src/crawler_extra.py langs）；spanish = 西班牙西语，latam = 拉美西语，两者合并成一个特征
LANGS = ["english", "russian", "german", "french", "polish", "brazilian", "spanish", "latam",
         "japanese", "koreana", "tchinese", "turkish"]
LANG_FEATURES = ["english", "russian", "german", "french", "polish", "brazilian", "spanish_all",
                 "japanese", "koreana", "tchinese", "turkish"]

# 主类型优先级：一款游戏往往同时挂 Action、Adventure、Indie 等多个标签，取最"具体"的那个当主类型
GENRE_PRIORITY = ["RPG", "Strategy", "Simulation", "Racing", "Sports", "Adventure", "Action",
                  "Casual", "Massively Multiplayer"]
GENRE_META = [
    (1, "Action", "动作"), (2, "Adventure", "冒险"), (3, "RPG", "角色扮演"), (4, "Strategy", "策略"),
    (5, "Simulation", "模拟"), (6, "Casual", "休闲"), (7, "Sports & Racing", "体育竞速"),
    (8, "Massively Multiplayer", "大型多人"), (9, "Indie Only", "仅标独立"), (10, "Other", "其他"),
]


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ------------------------------------------------------------------------------
# 建表
# ------------------------------------------------------------------------------
DDL = [
    # ---------------- 原始层 ----------------
    # 原始 CSV 里有 6 组完全重复的行，原始层照原样保存，所以用文件行号当主键，去重放到清洗步骤
    """CREATE TABLE IF NOT EXISTS ods_ars_players_2018 (
        row_no INTEGER PRIMARY KEY, appid INTEGER NOT NULL, title TEXT, players INTEGER NOT NULL, source_url TEXT)""",
    """CREATE TABLE IF NOT EXISTS ods_steam_appdetails (
        appid INTEGER PRIMARY KEY, success INTEGER, name TEXT, app_type TEXT, is_free INTEGER,
        release_date_text TEXT, release_date TEXT, currency TEXT, initial_price_usd REAL, final_price_usd REAL,
        discount_percent INTEGER, genres TEXT, categories TEXT, developers TEXT, publishers TEXT,
        metacritic INTEGER, achievements_total INTEGER, supports_schinese INTEGER, fetched_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS ods_steam_reviews_at_cutoff (
        appid INTEGER NOT NULL, cohort TEXT NOT NULL, cutoff_date TEXT NOT NULL, cutoff_ts INTEGER,
        total_reviews INTEGER, total_positive INTEGER, total_negative INTEGER, schinese_reviews INTEGER,
        steam_purchase_reviews INTEGER, reviews_before_policy INTEGER, sample_n INTEGER,
        sample_median_playtime_at_review_hrs REAL, sample_mean_playtime_at_review_hrs REAL, first_review_date TEXT,
        fetched_at TEXT, PRIMARY KEY (appid, cohort, cutoff_date))""",
    """CREATE TABLE IF NOT EXISTS ods_steamcharts_monthly (
        appid INTEGER NOT NULL, month TEXT NOT NULL, avg_players REAL, peak_players INTEGER,
        PRIMARY KEY (appid, month))""",
    """CREATE TABLE IF NOT EXISTS ods_recent_sales_truth (
        truth_id INTEGER PRIMARY KEY, appid INTEGER NOT NULL, game TEXT, report_date TEXT NOT NULL,
        reported_sales INTEGER NOT NULL, reported_sales_text TEXT, source_url TEXT, gamalytic_estimate INTEGER,
        benchmark_source TEXT)""",
    """CREATE TABLE IF NOT EXISTS ods_price_history_wayback (
        appid INTEGER NOT NULL, name TEXT, observed_date TEXT NOT NULL, wayback_timestamp TEXT,
        base_price_usd REAL, final_price_usd REAL, discount_percent INTEGER, archive_url TEXT,
        PRIMARY KEY (appid, wayback_timestamp))""",
    """CREATE TABLE IF NOT EXISTS ods_review_monthly_policy_scan (
        appid INTEGER NOT NULL, month TEXT NOT NULL, new_reviews INTEGER, PRIMARY KEY (appid, month))""",
    # 2026-10-20 新增：第二批真值（开发商/发行商公布的 Steam 销量，两轮收集 + 分层）与补充特征的原始数据
    """CREATE TABLE IF NOT EXISTS ods_official_sales_truth_v2 (
        truth_v2_id INTEGER PRIMARY KEY, appid INTEGER NOT NULL, game TEXT, report_date TEXT NOT NULL,
        as_of_date TEXT NOT NULL, as_of_rule TEXT, reported_number REAL NOT NULL, number_text TEXT, steam_truth REAL,
        tier TEXT NOT NULL, platform_scope TEXT, is_lower_bound TEXT, source_type TEXT, source_url TEXT,
        collect_round INTEGER, quote_found_on_page TEXT)""",
    """CREATE TABLE IF NOT EXISTS ods_review_language_counts (
        appid INTEGER NOT NULL, cohort TEXT NOT NULL, cutoff_date TEXT NOT NULL, language TEXT NOT NULL,
        total_reviews INTEGER, PRIMARY KEY (appid, cohort, cutoff_date, language))""",
    """CREATE TABLE IF NOT EXISTS ods_itad_price_history (
        appid INTEGER NOT NULL, country TEXT NOT NULL, ts TEXT NOT NULL, currency TEXT, price REAL, regular REAL,
        cut INTEGER)""",
    """CREATE TABLE IF NOT EXISTS ods_itad_bundles (
        appid INTEGER NOT NULL, bundle_id INTEGER NOT NULL, title TEXT, page TEXT, publish TEXT,
        PRIMARY KEY (appid, bundle_id))""",
    # ---------------- 运维表 ----------------
    """CREATE TABLE IF NOT EXISTS etl_job_log (
        job_id INTEGER PRIMARY KEY AUTOINCREMENT, job_name TEXT NOT NULL, start_time TEXT NOT NULL,
        end_time TEXT NOT NULL, rows_extracted INTEGER DEFAULT 0, rows_loaded INTEGER DEFAULT 0,
        rows_rejected INTEGER DEFAULT 0, status TEXT NOT NULL, duration_ms INTEGER DEFAULT 0, message TEXT)""",
    """CREATE TABLE IF NOT EXISTS etl_checkpoint (
        job_name TEXT PRIMARY KEY, last_processed_key TEXT, last_update_time TEXT,
        status TEXT DEFAULT 'READY', updated_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS etl_clean_error_log (
        error_id INTEGER PRIMARY KEY AUTOINCREMENT, job_name TEXT NOT NULL, appid INTEGER NOT NULL,
        cohort TEXT, game_name TEXT, rule_code TEXT NOT NULL, raw_value_desc TEXT,
        clean_action TEXT NOT NULL, logged_at TEXT)""",
    # ---------------- 维度表 ----------------
    """CREATE TABLE IF NOT EXISTS dim_genre (
        genre_id INTEGER PRIMARY KEY, genre_name TEXT NOT NULL UNIQUE, genre_cn_name TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS dim_release_date (
        date_key INTEGER PRIMARY KEY, release_year INTEGER NOT NULL UNIQUE, era_code TEXT NOT NULL,
        era_name TEXT NOT NULL, has_review_prompt_policy INTEGER NOT NULL, literature_k_multiple REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS dim_estimation_method (
        method_id INTEGER PRIMARY KEY, method_code TEXT NOT NULL UNIQUE, method_name TEXT NOT NULL,
        signal_source TEXT NOT NULL, impl_tech TEXT NOT NULL, uses_truth_for_fitting INTEGER NOT NULL,
        description TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS dim_game (
        game_key INTEGER PRIMARY KEY, appid INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,
        genre_id INTEGER NOT NULL REFERENCES dim_genre(genre_id),
        date_key INTEGER NOT NULL REFERENCES dim_release_date(date_key),
        release_date TEXT, price_usd REAL NOT NULL, price_tier TEXT NOT NULL, is_indie INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS dim_price_scd2 (
        price_scd_key INTEGER PRIMARY KEY AUTOINCREMENT, appid INTEGER NOT NULL, name TEXT NOT NULL,
        base_price_usd REAL NOT NULL, discount_percent INTEGER NOT NULL, final_price_usd REAL NOT NULL,
        effective_start TEXT NOT NULL, effective_end TEXT NOT NULL, is_current INTEGER NOT NULL,
        version_no INTEGER NOT NULL, observation_source TEXT NOT NULL, source_ref TEXT)""",
    # ---------------- 明细层 ----------------
    """CREATE TABLE IF NOT EXISTS dwd_game_sample (
        sample_id INTEGER PRIMARY KEY, appid INTEGER NOT NULL, name TEXT NOT NULL, cohort TEXT NOT NULL,
        era_group TEXT NOT NULL, cutoff_date TEXT NOT NULL, release_date TEXT NOT NULL, release_date_source TEXT NOT NULL,
        release_year INTEGER NOT NULL,
        game_age_years REAL NOT NULL, genre TEXT NOT NULL, is_indie INTEGER NOT NULL, price_usd REAL NOT NULL,
        price_tier TEXT NOT NULL, total_reviews INTEGER NOT NULL, reviews_before_policy INTEGER NOT NULL,
        reviews_after_policy INTEGER NOT NULL, positive_rate REAL NOT NULL, positive_rate_tier TEXT NOT NULL,
        schinese_ratio REAL NOT NULL, steam_purchase_ratio REAL NOT NULL, playtime_review_median_hrs REAL,
        review_sample_n INTEGER, charts_total_hours REAL, charts_coverage INTEGER NOT NULL,
        truth_value INTEGER NOT NULL, truth_type TEXT NOT NULL, k_ratio REAL NOT NULL, log_k REAL NOT NULL,
        log_reviews REAL NOT NULL, log_price REAL NOT NULL, model_eligible INTEGER NOT NULL,
        eligibility_note TEXT, source_url TEXT)""",
    # ---------------- 结果层（models.py 写入；快照表由本脚本增量写入） ----------------
    """CREATE TABLE IF NOT EXISTS fact_sales_estimation (
        estimation_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER NOT NULL, appid INTEGER NOT NULL,
        game_key INTEGER NOT NULL REFERENCES dim_game(game_key),
        genre_id INTEGER NOT NULL REFERENCES dim_genre(genre_id),
        date_key INTEGER NOT NULL REFERENCES dim_release_date(date_key),
        method_id INTEGER NOT NULL REFERENCES dim_estimation_method(method_id),
        method_code TEXT NOT NULL, cohort TEXT NOT NULL, era_group TEXT NOT NULL, total_reviews INTEGER NOT NULL,
        price_usd REAL NOT NULL, true_value INTEGER NOT NULL, estimated_sales INTEGER NOT NULL,
        estimated_low_bound INTEGER NOT NULL, estimated_high_bound INTEGER NOT NULL,
        estimated_revenue_usd REAL NOT NULL, implied_k REAL NOT NULL, observed_k REAL NOT NULL,
        pct_error REAL NOT NULL, abs_pct_error REAL NOT NULL, within_30pct_flag INTEGER NOT NULL,
        within_50pct_flag INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS fact_snapshot (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, fetched_at TEXT NOT NULL, round_no INTEGER NOT NULL,
        appid INTEGER NOT NULL, name TEXT, current_players INTEGER, total_reviews INTEGER,
        new_reviews_since_prev INTEGER, total_positive INTEGER, final_price_usd REAL, discount_percent INTEGER,
        loaded_at TEXT NOT NULL, UNIQUE (appid, fetched_at))""",
    """CREATE TABLE IF NOT EXISTS ads_review_monthly_index (
        month TEXT PRIMARY KEY, median_index REAL, q25 REAL, q75 REAL, n_games INTEGER, is_policy_month INTEGER)""",
    """CREATE TABLE IF NOT EXISTS ads_model_eval_summary (
        eval_id INTEGER PRIMARY KEY AUTOINCREMENT, eval_cohort TEXT NOT NULL, method_code TEXT NOT NULL,
        method_name TEXT NOT NULL, sample_size INTEGER NOT NULL, mape_pct REAL, mdape_pct REAL, r2_log REAL,
        hit_rate_10pct REAL, hit_rate_30pct REAL, hit_rate_50pct REAL, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS ads_feature_importance (
        feat_id INTEGER PRIMARY KEY AUTOINCREMENT, feature_code TEXT NOT NULL, feature_cn_name TEXT NOT NULL,
        m2_ols_coef REAL, m2_std_err REAL, m2_t_stat REAL, m2_p_value REAL, m4_rf_importance_pct REAL,
        m4_gbdt_importance_pct REAL)""",
    """CREATE TABLE IF NOT EXISTS ads_policy_calibration (
        calib_id INTEGER PRIMARY KEY AUTOINCREMENT, item TEXT NOT NULL, value REAL, note TEXT)""",
]


# 明细层 2026-10-20 新增的列（各语言评测占比、ITAD 价格与慈善包特征）。旧库用 ALTER TABLE 补上，不用删库重建
DWD_EXTRA_COLS = [(f"share_{l}", "REAL") for l in LANG_FEATURES] + [
    ("itad_covered", "INTEGER"), ("price_at_cutoff_usd", "REAL"), ("launch_price_usd", "REAL"),
    ("n_discounts", "INTEGER"), ("max_cut", "INTEGER"), ("frac_time_discounted", "REAL"),
    ("frac_time_cut75", "REAL"), ("n_bundles", "INTEGER"),
    ("ln_regional_cn", "REAL"), ("ln_regional_tr", "REAL"), ("ln_regional_br", "REAL"),
]


def init_dw_tables(conn):
    with conn:
        for stmt in DDL:
            conn.execute(stmt)
        have = {r[1] for r in conn.execute("PRAGMA table_info(dwd_game_sample)")}
        for col, typ in DWD_EXTRA_COLS:
            if col not in have:
                conn.execute(f"ALTER TABLE dwd_game_sample ADD COLUMN {col} {typ}")


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_etl_job(conn, job_name, t0, extracted, loaded, rejected, status, msg):
    t1 = time.time()
    with conn:
        conn.execute(
            """INSERT INTO etl_job_log (job_name, start_time, end_time, rows_extracted, rows_loaded,
               rows_rejected, status, duration_ms, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
             extracted, loaded, rejected, status, int((t1 - t0) * 1000), msg))


def none_if_nan(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_policy():
    with open(RAW_DIR / "policy_change_detected.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# 步骤 1：原始层全量装载
# ------------------------------------------------------------------------------
def load_table_from_csv(conn, table, csv_path, columns, transform=None):
    df = pd.read_csv(csv_path)
    rows = []
    for rec in df.to_dict("records"):
        if transform:
            rec = transform(rec)
        rows.append(tuple(none_if_nan(rec.get(c)) for c in columns))
    with conn:
        conn.execute(f"DELETE FROM {table};")
        conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", rows)
    return len(rows)


def load_df(conn, table, df, columns):
    """和 load_table_from_csv 一样（先清空再写入），但输入是已经拼好的 DataFrame。"""
    rows = [tuple(none_if_nan(rec.get(c)) for c in columns) for rec in df.to_dict("records")]
    with conn:
        conn.execute(f"DELETE FROM {table};")
        conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", rows)
    return len(rows)


def read_with_v2(name, keys):
    """主表 + 新真值集单独导出的同结构表（*_truth_v2.csv）拼在一起，按主键去重（同一款游戏两边都有时保留主表那条）。"""
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    p2 = RAW_DIR / f"{name}_truth_v2.csv"
    if p2.exists():
        df = pd.concat([df, pd.read_csv(p2)], ignore_index=True)
    return df.drop_duplicates(subset=keys, keep="first")


def load_ods(conn):
    t0 = time.time()
    n = 0
    ars_row_no = iter(range(1, 1000000))
    n += load_table_from_csv(conn, "ods_ars_players_2018", RAW_DIR / "ars_players_2018.csv",
                             ["row_no", "appid", "title", "players", "source_url"],
                             lambda r: {**r, "row_no": next(ars_row_no), "source_url": ARS_SOURCE_URL})
    n += load_df(conn, "ods_steam_appdetails", read_with_v2("steam_appdetails", ["appid"]),
                             ["appid", "success", "name", "app_type", "is_free", "release_date_text", "release_date",
                              "currency", "initial_price_usd", "final_price_usd", "discount_percent", "genres",
                              "categories", "developers", "publishers", "metacritic", "achievements_total",
                              "supports_schinese", "fetched_at"])
    n += load_df(conn, "ods_steam_reviews_at_cutoff",
                 read_with_v2("steam_reviews_at_cutoff", ["appid", "cohort", "cutoff_date"]),
                             ["appid", "cohort", "cutoff_date", "cutoff_ts", "total_reviews", "total_positive",
                              "total_negative", "schinese_reviews", "steam_purchase_reviews", "reviews_before_policy",
                              "sample_n", "sample_median_playtime_at_review_hrs",
                              "sample_mean_playtime_at_review_hrs", "first_review_date", "fetched_at"])
    n += load_df(conn, "ods_steamcharts_monthly", read_with_v2("steamcharts_monthly", ["appid", "month"]),
                 ["appid", "month", "avg_players", "peak_players"])
    # 真值表本身没有主键（同一款游戏可能有两个公布日期，比如 Missing Hiker），按文件里的行号编 truth_id
    row_no = iter(range(1, 100000))
    n += load_table_from_csv(conn, "ods_recent_sales_truth", RAW_DIR / "recent_sales_truth.csv",
                             ["truth_id", "appid", "game", "report_date", "reported_sales", "reported_sales_text",
                              "source_url", "gamalytic_estimate", "benchmark_source"],
                             lambda r: {**r, "truth_id": next(row_no)})
    if (RAW_DIR / "price_history_wayback.csv").exists():
        n += load_table_from_csv(conn, "ods_price_history_wayback", RAW_DIR / "price_history_wayback.csv",
                                 ["appid", "name", "observed_date", "wayback_timestamp", "base_price_usd",
                                  "final_price_usd", "discount_percent", "archive_url"])
    n += load_table_from_csv(conn, "ods_review_monthly_policy_scan", RAW_DIR / "review_monthly_policy_scan.csv",
                             ["appid", "month", "new_reviews"])
    # ---- 2026-10-20 新增 ----
    n += load_table_from_csv(conn, "ods_official_sales_truth_v2", RAW_DIR / "official_sales_truth_v2.csv",
                             ["truth_v2_id", "appid", "game", "report_date", "as_of_date", "as_of_rule",
                              "reported_number", "number_text", "steam_truth", "tier", "platform_scope",
                              "is_lower_bound", "source_type", "source_url", "collect_round", "quote_found_on_page"])
    lang = pd.read_csv(RAW_DIR / "review_language_counts.csv").drop_duplicates(
        subset=["appid", "cohort", "cutoff_date", "language"])
    n += load_df(conn, "ods_review_language_counts", lang, ["appid", "cohort", "cutoff_date", "language", "total_reviews"])
    hist = pd.read_csv(RAW_DIR / "itad_price_history.csv").rename(columns={"timestamp": "ts"})
    n += load_df(conn, "ods_itad_price_history", hist, ["appid", "country", "ts", "currency", "price", "regular", "cut"])
    bund = pd.read_csv(RAW_DIR / "itad_bundles.csv").drop_duplicates(subset=["appid", "bundle_id"])
    n += load_df(conn, "ods_itad_bundles", bund, ["appid", "bundle_id", "title", "page", "publish"])
    log_etl_job(conn, "ODS_FULL_LOAD", t0, n, n, 0, "SUCCESS",
                "原始层 11 张表全量装载（先清空再写入，同一事务）；评测/商店/SteamCharts 含新真值集的样本")


# ------------------------------------------------------------------------------
# 步骤 2：静态维度
# ------------------------------------------------------------------------------
def load_static_dims(conn, policy):
    t0 = time.time()
    policy_year = int(policy["policy_month"][:4])
    with conn:
        # 事实表外键指向这些维度，重建维度前先清空事实表（models.py 会重新写入）
        conn.execute("DELETE FROM fact_sales_estimation;")
        conn.execute("DELETE FROM dim_game;")
        conn.execute("DELETE FROM dim_genre;")
        conn.executemany("INSERT INTO dim_genre VALUES (?, ?, ?)", GENRE_META)

        # 文献 k 值（M1 用）：设计文档 3.5 节整理的业界经验值，2017 年前 74、2019 年 51、2020 年后 38；
        # 2017-2018 文档没有给，取 74 和 51 之间的 60，并在报告里注明是插值
        conn.execute("DELETE FROM dim_release_date;")
        for yr in range(1990, 2027):  # Steam 上有 1990 年代的老游戏（如 Half-Life）
            if yr <= 2016:
                row = ("ERA_EARLY", "早期(<=2016)", 0, 74.0)
            elif yr <= 2018:
                row = ("ERA_PRE_POLICY", "提示上线前(2017-2018)", 0, 60.0)
            elif yr == policy_year:
                row = ("ERA_TRANSITION", f"提示上线当年({yr})", 1, 51.0)
            else:
                row = ("ERA_POST_POLICY", f"提示上线后(>={policy_year + 1})", 1, 38.0)
            conn.execute("INSERT INTO dim_release_date VALUES (?, ?, ?, ?, ?, ?)", (yr, yr) + row)

        conn.execute("DELETE FROM dim_estimation_method;")
        methods = [
            (1, "M0", "M0 固定倍数 k=30", "评测", "SQL", 0, "销量 = 评测数 × 30"),
            (2, "M1", "M1 按发售年份取文献 k", "评测", "SQL", 0, "<=2016:74, 2017-18:60, 2019:51, >=2020:38"),
            (3, "M2", "M2 对数线性回归", "评测", "Python", 1,
             "2018 数据学 ln k；估近年销量时再用真值集把系数重估一遍（留一法）"),
            (4, "M3", "M3 分位数回归(中位数+80%区间)", "评测", "Python", 1, "2018 拟合 10%/50%/90% 分位；近年样本中位数沿用 M2 校准"),
            (5, "M4", "M4 随机森林+梯度提升树", "评测", "Python", 1, "手写 CART 集成；近年校准同 M2"),
            (6, "Signal_B", "信号B 在线人数×时长", "在线", "Python", 0,
             "SteamCharts 累计游玩小时 ÷ 人均小时（人均小时用 2018 真值校准评测者时长）"),
            (7, "Fusion_AB", "融合 A(M4)+B", "评测+在线", "Python", 1, "对数空间加权；校准与权重都用嵌套留一法"),
        ]
        conn.executemany("INSERT INTO dim_estimation_method VALUES (?, ?, ?, ?, ?, ?, ?)", methods)
    log_etl_job(conn, "STATIC_DIM_LOAD", t0, 0, len(GENRE_META), 0, "SUCCESS", "类型/年份/估算方法三张静态维度表")


# ------------------------------------------------------------------------------
# 步骤 3：清洗 ODS -> DWD（分批提交 + 检查点断点续传）
# ------------------------------------------------------------------------------
def classify_price_tier(p):
    if p < 10:
        return "1_<$10"
    if p < 20:
        return "2_$10-20"
    if p < 40:
        return "3_$20-40"
    return "4_>=$40"


def classify_pos_rate_tier(r):
    if r < 0.60:
        return "1_<60%"
    if r < 0.70:
        return "2_60-70%"
    if r < 0.80:
        return "3_70-80%"
    if r < 0.90:
        return "4_80-90%"
    if r < 0.95:
        return "5_90-95%"
    return "6_>=95%"


def main_genre(genres_text):
    tags = [g for g in str(genres_text or "").split("|") if g]
    for g in GENRE_PRIORITY:
        if g in tags:
            return "Sports & Racing" if g in ("Racing", "Sports") else g
    return "Indie Only" if "Indie" in tags else "Other"


def charts_hours_before(month_rows, cutoff_date, release_date=None):
    """
    SteamCharts 月均在线 × 当月小时数，累加到截止日所在月份之前（不算不完整的当月）。
    坑：上架当月的"月均在线"是按上架后那几天平均的，第一版却乘了整月的小时数，
    上架越晚的游戏首月被放大得越多（红色沙漠 3 月 19 日上架，放大约 2.4 倍）。现在上架当月只按上架后的天数算。
    """
    cutoff_month = cutoff_date[:7]
    total = 0.0
    for month, avg in month_rows:
        if month >= cutoff_month:
            continue
        y, m = map(int, month.split("-"))
        days = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.date(y, m, 1)).days
        if release_date and month == release_date[:7]:
            days = days - int(release_date[8:10]) + 1
        total += avg * days * 24
    return total


def build_candidates(conn, policy):
    """把三个来源拼成"候选样本"：2018 训练样本来自 Ars 泄露表，真值样本来自 Gamalytic 测试表。"""
    apps = {r[0]: r for r in conn.execute(
        "SELECT appid, success, name, app_type, is_free, release_date, initial_price_usd, genres FROM ods_steam_appdetails")}
    reviews = {(r[0], r[1], r[2]): r for r in conn.execute(
        """SELECT appid, cohort, cutoff_date, total_reviews, total_positive, schinese_reviews, steam_purchase_reviews,
                  reviews_before_policy, sample_n, sample_median_playtime_at_review_hrs, first_review_date
           FROM ods_steam_reviews_at_cutoff""")}
    charts = {}
    for appid, month, avg in conn.execute("SELECT appid, month, avg_players FROM ods_steamcharts_monthly ORDER BY month"):
        charts.setdefault(appid, []).append((month, avg))

    cands = []
    for appid, title, players, n_rows in conn.execute(
            """SELECT appid, MAX(title), MAX(players), COUNT(*) FROM ods_ars_players_2018
               WHERE players >= ? GROUP BY appid ORDER BY appid""", (MIN_PLAYERS_2018,)):
        if appid not in apps:
            continue  # 还没爬到（采集只跑了一部分时会出现），不算清洗问题
        cands.append({"key": f"A{appid:08d}", "appid": appid, "cohort": "train_2018", "cutoff_date": CUTOFF_2018,
                      "truth_value": players, "truth_type": "players_2018_leak", "source_url": ARS_SOURCE_URL,
                      "fallback_name": title, "dup_rows": n_rows})
    for truth_id, appid, game, report_date, sales, src in conn.execute(
            "SELECT truth_id, appid, game, report_date, reported_sales, source_url FROM ods_recent_sales_truth ORDER BY truth_id"):
        cands.append({"key": f"T{truth_id:08d}", "appid": appid, "cohort": "truth", "cutoff_date": report_date,
                      "truth_value": sales, "truth_type": "reported_sales", "source_url": src, "fallback_name": game})
    # 新真值集：截止日取 as_of_date（"首日/首周卖了多少"已换算成 发售日 + 天数），真值取 Steam 口径的数字
    tiers = TRUTH_V2_TIERS_FIT + TRUTH_V2_TIERS_REF
    for tid, appid, game, as_of, value, tier, src in conn.execute(
            f"""SELECT truth_v2_id, appid, game, as_of_date, steam_truth, tier, source_url FROM ods_official_sales_truth_v2
                WHERE steam_truth IS NOT NULL AND tier IN ({','.join('?' * len(tiers))}) ORDER BY truth_v2_id""", tiers):
        cands.append({"key": f"W{tid:08d}", "appid": appid, "cohort": "truth_v2", "cutoff_date": as_of,
                      "truth_value": value, "truth_type": tier, "source_url": src, "fallback_name": game})
    cands.sort(key=lambda c: c["key"])
    return cands, apps, reviews, charts


def clean_one(c, apps, reviews, charts, policy):
    """
    返回 (dwd 行 或 None, [(规则码, 说明, 处理方式), ...])。
    处理方式 FILTERED_OUT = 不进明细层；FLAGGED = 进明细层但不参与建模（model_eligible=0）。
    """
    issues = []
    if c.get("dup_rows", 1) > 1:
        issues.append(("DUPLICATE_SOURCE_ROW", f"原始 CSV 中重复 {c['dup_rows']} 次，按 appid 去重保留一条", "DEDUPLICATED"))
    a = apps.get(c["appid"])
    if a is None or not a[1]:
        return None, issues + [("DELISTED_OR_UNAVAILABLE", "appdetails success=false（已下架或锁区）", "FILTERED_OUT")]
    _, _, name, app_type, is_free, release_date, price, genres = a
    if app_type != "game":
        return None, issues + [("NON_GAME", f"app_type={app_type}", "FILTERED_OUT")]
    if is_free:
        return None, issues + [("FREE_GAME", "is_free=1，免费游戏 k 约为付费的两倍，不在研究范围", "FILTERED_OUT")]
    if price is None or price <= 0:
        return None, issues + [("NO_PRICE", "商店没有美元价格（可能停售）", "FILTERED_OUT")]
    rv = reviews.get((c["appid"], c["cohort"], c["cutoff_date"]))
    if rv is None:
        return None, issues + [("NO_REVIEW_DATA", "评测接口未取到数据", "FILTERED_OUT")]
    _, _, _, tot, pos, sch, steam_p, before_pol, sample_n, pt_med, first_review = rv
    if not tot or tot <= 0:
        return None, issues + [("ZERO_REVIEWS", f"截至 {c['cutoff_date']} 评测数为 0", "FILTERED_OUT")]

    # 实际上架日期：商店发售日、第一条评测日期、SteamCharts 第一个月，三者取最早。
    # 坑：抢先体验游戏转正后，商店显示的是正式版日期（比如 Core Keeper 显示 2024 年），
    # 第一版清洗把这些游戏全当成"发售日晚于截止日"丢掉了，真值集一下少了 14 款。
    month_rows = charts.get(c["appid"], [])
    options = []
    if release_date:
        options.append((release_date, "store"))
    if first_review:
        options.append((first_review, "first_review"))
    # SteamCharts 只有月份，补成当月 1 号会比商店日期早几天；只有首月早于商店发售所在月时才采用，避免误判
    if month_rows and (not release_date or month_rows[0][0] < release_date[:7]):
        options.append((month_rows[0][0] + "-01", "steamcharts_first_month"))
    if not options:
        return None, issues + [("RELEASE_DATE_UNKNOWN", "商店发售日无法解析，也没有其他日期来源", "FILTERED_OUT")]
    release_date, release_src = min(options)
    if release_src != "store":
        issues.append(("RELEASE_DATE_ADJUSTED", f"商店发售日 {a[5]}，改用 {release_src} = {release_date}", "FLAGGED"))
    if release_date > c["cutoff_date"]:
        return None, issues + [("RELEASED_AFTER_CUTOFF",
                       f"上架日期 {release_date} 晚于截止日 {c['cutoff_date']}", "FILTERED_OUT")]

    policy_date = policy["policy_month"] + "-01"
    if c["cohort"] == "train_2018" or c["cutoff_date"] < policy_date:
        era_group = "pre_policy"
        before = tot
    elif release_date >= policy_date:
        era_group = "post_policy"
        before = 0
    else:
        era_group = "cross_era"
        before = int(before_pol) if before_pol is not None and not (isinstance(before_pol, float) and math.isnan(before_pol)) else 0
    before = min(before, tot)

    rel = dt.date.fromisoformat(release_date)
    cut = dt.date.fromisoformat(c["cutoff_date"])
    age_years = max((cut - rel).days / 365.25, 1 / 12)

    coverage = 0
    hours = None
    if month_rows:
        first_month = month_rows[0][0]
        # 只有 SteamCharts 从发售后两个月内就开始记录，累计小时才算完整；2012 年 7 月以前发售的游戏都不完整
        rel_plus2 = (rel.replace(day=1) + dt.timedelta(days=62)).strftime("%Y-%m")
        # 截止日距"确定的上架日"（商店日期或第一条评测，不用 SteamCharts 月份）不足 45 天时，还没有一个完整月的在线数据，
        # 信号 B 只能算到上架前的零星记录（SCUM、《史丹利的寓言》按首周销量截止时，B 只有真值的万分之一）。这种情况不算 B
        firm = [d for d, src in options if src != "steamcharts_first_month"]
        firm_rel = dt.date.fromisoformat(min(firm)) if firm else rel
        if first_month <= rel_plus2 and first_month < c["cutoff_date"][:7] and (cut - firm_rel).days >= 45:
            coverage = 1
            hours = charts_hours_before(month_rows, c["cutoff_date"], release_date)
            if hours <= 0:
                coverage, hours = 0, None

    eligible, note = 1, ""
    if c["cohort"] == "train_2018" and tot < MIN_REVIEWS_2018:
        eligible, note = 0, f"评测数 {tot} < {MIN_REVIEWS_2018}，受抽样门槛截断影响，不参与 2018 建模"
        issues.append(("TRUNCATION_RISK", note, "FLAGGED"))

    k = c["truth_value"] / tot
    row = {
        "appid": c["appid"], "name": name or c["fallback_name"], "cohort": c["cohort"], "era_group": era_group,
        "cutoff_date": c["cutoff_date"], "release_date": release_date, "release_date_source": release_src,
        "release_year": rel.year,
        "game_age_years": round(age_years, 4), "genre": main_genre(genres),
        "is_indie": int("Indie" in str(genres or "").split("|")), "price_usd": float(price),
        "price_tier": classify_price_tier(float(price)), "total_reviews": int(tot),
        "reviews_before_policy": int(before), "reviews_after_policy": int(tot - before),
        "positive_rate": round(pos / tot, 4), "positive_rate_tier": classify_pos_rate_tier(pos / tot),
        "schinese_ratio": round((sch or 0) / tot, 4), "steam_purchase_ratio": round((steam_p or 0) / tot, 4),
        "playtime_review_median_hrs": none_if_nan(pt_med), "review_sample_n": int(sample_n or 0),
        "charts_total_hours": round(hours, 1) if hours else None, "charts_coverage": coverage,
        "truth_value": int(c["truth_value"]), "truth_type": c["truth_type"], "k_ratio": round(k, 4),
        "log_k": round(math.log(k), 6), "log_reviews": round(math.log(tot), 6),
        "log_price": round(math.log(float(price)), 6), "model_eligible": eligible, "eligibility_note": note,
        "source_url": c["source_url"],
    }
    return row, issues


DWD_COLS = ["sample_id", "appid", "name", "cohort", "era_group", "cutoff_date", "release_date", "release_date_source",
            "release_year",
            "game_age_years", "genre", "is_indie", "price_usd", "price_tier", "total_reviews", "reviews_before_policy",
            "reviews_after_policy", "positive_rate", "positive_rate_tier", "schinese_ratio", "steam_purchase_ratio",
            "playtime_review_median_hrs", "review_sample_n", "charts_total_hours", "charts_coverage", "truth_value",
            "truth_type", "k_ratio", "log_k", "log_reviews", "log_price", "model_eligible", "eligibility_note",
            "source_url"] + [c for c, _ in DWD_EXTRA_COLS]


def load_extra_lookups(conn):
    """补充特征用的查找表：各语言评测数、ITAD 价格历史（按游戏和地区）、慈善包发布时间。"""
    langs = {}
    for appid, cohort, cut, lang, n in conn.execute(
            "SELECT appid, cohort, cutoff_date, language, total_reviews FROM ods_review_language_counts"):
        langs.setdefault((appid, cohort, cut), {})[lang] = n or 0
    hist = pd.read_sql_query("SELECT appid, country, ts, regular, cut FROM ods_itad_price_history", conn)
    hist["t"] = pd.to_datetime(hist["ts"], utc=True, format="ISO8601").astype("int64") // 10 ** 9
    itad = {}
    for (appid, country), g in hist.sort_values("t").groupby(["appid", "country"]):
        itad[(appid, country)] = (g["t"].values, g["regular"].values.astype(float), g["cut"].fillna(0).values.astype(float))
    bund = pd.read_sql_query("SELECT appid, publish FROM ods_itad_bundles", conn)
    bund["t"] = pd.to_datetime(bund["publish"], utc=True, errors="coerce", format="ISO8601")
    bundles = {a: np.sort(g["t"].dropna().astype("int64").values // 10 ** 9) for a, g in bund.groupby("appid")}
    return {"langs": langs, "itad": itad, "bundles": bundles}


def extra_features(row, lk):
    """
    截至截止日的补充特征（全部只用截止日当天及以前的数据，不看真值）：
    - 各语言评测占比：现有模型只有简中一种语言；德语、波兰语玩家写评测的意愿明显不同
    - ITAD 美区价格历史：截止日当时的原价（替代 2026 年抓取时的商店价）、首发价、打折次数、最深折扣、打折时间占比
    - 中国/土耳其/巴西区原价相对美区的比值（取对数）：低价区定价策略
    - 截止日前进过几次慈善包
    """
    out = {c: None for c, _ in DWD_EXTRA_COLS}
    cut_ts = int(dt.datetime.fromisoformat(row["cutoff_date"]).replace(tzinfo=dt.timezone.utc).timestamp()) + 86399
    n = row["total_reviews"]
    lc = lk["langs"].get((row["appid"], row["cohort"], row["cutoff_date"]))
    if lc and n:
        for l in LANG_FEATURES:
            v = lc.get("spanish", 0) + lc.get("latam", 0) if l == "spanish_all" else lc.get(l, 0)
            out[f"share_{l}"] = round(min(v / n, 1.0), 5)
    us = lk["itad"].get((row["appid"], "US"))
    out["itad_covered"] = 0
    if us is not None:
        t, reg, cut = us
        m = t <= cut_ts
        if m.sum() > 0 and reg[m][-1] > 0:
            t, reg, cut = t[m], reg[m], cut[m]
            out["itad_covered"] = 1
            out["price_at_cutoff_usd"] = float(reg[-1])
            out["launch_price_usd"] = float(reg[0])
            started = (cut > 0) & (np.r_[0, cut[:-1]] == 0)
            out["n_discounts"] = int(started.sum())
            out["max_cut"] = int(cut.max())
            dur = np.clip(np.r_[t[1:], cut_ts] - t, 0, None).astype(float)
            tot = dur.sum()
            if tot > 0:
                out["frac_time_discounted"] = round(float(dur[cut > 0].sum() / tot), 5)
                out["frac_time_cut75"] = round(float(dur[cut >= 75].sum() / tot), 5)
            for country in ("CN", "TR", "BR"):
                h = lk["itad"].get((row["appid"], country))
                if h is not None:
                    mm = h[0] <= cut_ts
                    if mm.sum() > 0 and h[1][mm][-1] > 0:
                        out[f"ln_regional_{country.lower()}"] = round(math.log(h[1][mm][-1] / reg[-1]), 5)
    b = lk["bundles"].get(row["appid"])
    out["n_bundles"] = int((b <= cut_ts).sum()) if b is not None else 0
    return out


def truth_eligibility(row):
    """真值样本里不参与校准的：评测太少的、只是 PC 全商店口径（上界）的。返回要写进清洗日志的记录。"""
    if row["cohort"] not in ("truth", "truth_v2"):
        return []
    issues = []
    if row["total_reviews"] < MIN_REVIEWS_TRUTH:
        row["model_eligible"], row["eligibility_note"] = 0, f"评测数 {row['total_reviews']} < {MIN_REVIEWS_TRUTH}，k 的随机误差太大"
        issues.append(("TRUTH_FEW_REVIEWS", row["eligibility_note"], "FLAGGED"))
    if row["cutoff_date"] < "2013-11-01":
        row["model_eligible"], row["eligibility_note"] = 0, "Steam 评测系统 2013-11 才上线，截止日在此之前，评测数不可比"
        issues.append(("TRUTH_BEFORE_REVIEW_SYSTEM", row["eligibility_note"], "FLAGGED"))
    elif row["k_ratio"] < 2:
        # 每 2 份销量就有 1 条评测，远超 Steam 上见过的评测率；多半是把"首月/首周"的数字配上了多年后的评测数
        row["model_eligible"], row["eligibility_note"] = 0, f"k = {row['k_ratio']:.2f} < 2，不合常理，多半是时间点配错"
        issues.append(("TRUTH_IMPLAUSIBLE_K", row["eligibility_note"], "FLAGGED"))
    if row["truth_type"] == "P_pc":
        row["model_eligible"], row["eligibility_note"] = 0, "PC 全商店口径，真实 Steam 销量不超过该数，只作参考"
        issues.append(("TRUTH_PC_UPPER_BOUND", row["eligibility_note"], "FLAGGED"))
    elif row["truth_type"] in TRUTH_V2_TIERS_REF:
        row["model_eligible"], row["eligibility_note"] = 0, "标为 Steam 口径但原句没提平台，口径存疑，只作参考"
        issues.append(("TRUTH_SCOPE_UNVERIFIED", row["eligibility_note"], "FLAGGED"))
    return issues


class SimulatedCrash(Exception):
    pass


def clean_ods_to_dwd(conn, policy, crash_after_batches=None):
    """
    逐批清洗：每处理 BATCH_SIZE 条就提交一次事务，并把这一批最后一条的样本键写进 etl_checkpoint。
    如果进程中途挂掉，下次运行时先读检查点，只处理键比它大的样本，已经提交的批次不会重复写。
    crash_after_batches 用来在答辩时演示"中断 -> 续跑"：处理到第 N 批后故意抛异常。
    """
    job = "CLEAN_ODS_TO_DWD"
    t0 = time.time()
    cp = conn.execute("SELECT last_processed_key, status FROM etl_checkpoint WHERE job_name = ?", (job,)).fetchone()
    if cp is None or cp[1] == "COMPLETED":
        # 上次已正常结束（或从没跑过）：本次是全量重建
        with conn:
            conn.execute("DELETE FROM dwd_game_sample;")
            conn.execute("DELETE FROM etl_clean_error_log WHERE job_name = ?;", (job,))
            conn.execute("INSERT OR REPLACE INTO etl_checkpoint VALUES (?, '', NULL, 'RUNNING', ?)", (job, now_str()))
        resume_key = ""
    else:
        resume_key = cp[0] or ""
        print(f"[ETL] 检测到上次未完成（状态 {cp[1]}），从样本键 > {resume_key} 继续", flush=True)

    cands, apps, reviews, charts = build_candidates(conn, policy)
    lk = load_extra_lookups(conn)
    todo = [c for c in cands if c["key"] > resume_key]
    next_id = (conn.execute("SELECT COALESCE(MAX(sample_id), 0) FROM dwd_game_sample").fetchone()[0]) + 1
    loaded = rejected = 0
    for b in range(0, len(todo), BATCH_SIZE):
        batch = todo[b:b + BATCH_SIZE]
        with conn:  # 一批一个事务：要么整批写入并推进检查点，要么整批都不生效
            for c in batch:
                row, issues = clean_one(c, apps, reviews, charts, policy)
                if row is not None:
                    row.update(extra_features(row, lk))
                    issues = issues + truth_eligibility(row)
                for code, desc, action in issues:
                    conn.execute(
                        """INSERT INTO etl_clean_error_log (job_name, appid, cohort, game_name, rule_code,
                           raw_value_desc, clean_action, logged_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (job, c["appid"], c["cohort"], c["fallback_name"], code, desc, action, now_str()))
                if row is None:
                    rejected += 1
                    continue
                row["sample_id"] = next_id
                next_id += 1
                conn.execute(f"INSERT INTO dwd_game_sample ({', '.join(DWD_COLS)}) VALUES ({', '.join('?' * len(DWD_COLS))})",
                             tuple(row[col] for col in DWD_COLS))
                loaded += 1
            conn.execute("UPDATE etl_checkpoint SET last_processed_key = ?, status = 'RUNNING', updated_at = ? WHERE job_name = ?",
                         (batch[-1]["key"], now_str(), job))
        if crash_after_batches is not None and (b // BATCH_SIZE + 1) >= crash_after_batches:
            with conn:
                conn.execute("UPDATE etl_checkpoint SET status = 'INTERRUPTED', updated_at = ? WHERE job_name = ?", (now_str(), job))
            log_etl_job(conn, job + "_PART1", t0, b + len(batch), loaded, rejected, "INTERRUPTED",
                        f"演示：处理 {crash_after_batches} 批后中断，检查点停在 {batch[-1]['key']}")
            raise SimulatedCrash(batch[-1]["key"])
    with conn:
        conn.execute("UPDATE etl_checkpoint SET status = 'COMPLETED', updated_at = ? WHERE job_name = ?", (now_str(), job))
    log_etl_job(conn, job + ("_RESUME" if resume_key else ""), t0, len(todo), loaded, rejected, "SUCCESS",
                f"从样本键 > '{resume_key}' 开始处理 {len(todo)} 条候选，入明细层 {loaded} 条，拦截 {rejected} 条")


def build_dim_game(conn):
    t0 = time.time()
    genre_map = dict(conn.execute("SELECT genre_name, genre_id FROM dim_genre").fetchall())
    rows = conn.execute(
        """SELECT appid, MAX(name), MAX(genre), MAX(release_date), MAX(release_year), MAX(price_usd), MAX(price_tier), MAX(is_indie)
           FROM dwd_game_sample GROUP BY appid ORDER BY appid""").fetchall()
    with conn:
        conn.execute("DELETE FROM dim_game;")
        for key, (appid, name, genre, rel, yr, price, tier, indie) in enumerate(rows, start=1):
            conn.execute("INSERT INTO dim_game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (key, appid, name, genre_map.get(genre, 10), yr, rel, price, tier, indie))
    log_etl_job(conn, "DIM_GAME_BUILD", t0, len(rows), len(rows), 0, "SUCCESS", "按 appid 去重生成游戏维度")


# ------------------------------------------------------------------------------
# 步骤 4：价格维度 SCD Type 2（真实观测：档案馆历史存档 + 本次采集时的商店价格）
# ------------------------------------------------------------------------------
def apply_scd2_price_history(conn):
    """
    按时间顺序回放价格观测：
    - 这款游戏还没有当前版本 -> 插入 version 1；
    - 价格或折扣和当前版本不同 -> 把当前版本的 effective_end 设为本次观测日、is_current=0，再插入新版本；
    - 一样 -> 什么都不做（拉链表只在"变化"时开新版本）。
    """
    t0 = time.time()
    obs = []
    for appid, name, d, base, final, disc, url in conn.execute(
            """SELECT appid, name, observed_date, base_price_usd, final_price_usd, discount_percent, archive_url
               FROM ods_price_history_wayback ORDER BY appid, observed_date"""):
        obs.append((appid, name, d, base, final, int(disc), "wayback_archive", url))
    # 每款游戏再追加一条"本次采集时"的商店价格，作为最新观测
    tracked = {o[0] for o in obs}
    for appid, name, init_p, final_p, disc, fetched in conn.execute(
            "SELECT appid, name, initial_price_usd, final_price_usd, discount_percent, fetched_at FROM ods_steam_appdetails"):
        if appid in tracked and init_p:
            obs.append((appid, name, fetched[:10], init_p, final_p, int(disc or 0), "store_api",
                        f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=us"))
    obs.sort(key=lambda o: (o[0], o[2]))
    loaded = 0
    with conn:
        conn.execute("DELETE FROM dim_price_scd2;")
        for appid, name, d, base, final, disc, src, ref in obs:
            cur = conn.execute(
                "SELECT price_scd_key, base_price_usd, final_price_usd, discount_percent, version_no FROM dim_price_scd2 WHERE appid = ? AND is_current = 1",
                (appid,)).fetchone()
            if cur is None:
                conn.execute(
                    """INSERT INTO dim_price_scd2 (appid, name, base_price_usd, discount_percent, final_price_usd,
                       effective_start, effective_end, is_current, version_no, observation_source, source_ref)
                       VALUES (?, ?, ?, ?, ?, ?, '9999-12-31', 1, 1, ?, ?)""", (appid, name, base, disc, final, d, src, ref))
                loaded += 1
            elif abs(cur[1] - base) > 1e-6 or abs(cur[2] - final) > 1e-6 or cur[3] != disc:
                conn.execute("UPDATE dim_price_scd2 SET effective_end = ?, is_current = 0 WHERE price_scd_key = ?", (d, cur[0]))
                conn.execute(
                    """INSERT INTO dim_price_scd2 (appid, name, base_price_usd, discount_percent, final_price_usd,
                       effective_start, effective_end, is_current, version_no, observation_source, source_ref)
                       VALUES (?, ?, ?, ?, ?, ?, '9999-12-31', 1, ?, ?, ?)""",
                    (appid, name, base, disc, final, d, cur[4] + 1, src, ref))
                loaded += 1
    log_etl_job(conn, "SCD2_PRICE_DIMENSION", t0, len(obs), loaded, 0, "SUCCESS",
                f"{len(obs)} 条真实价格观测回放成 {loaded} 个价格版本")


# ------------------------------------------------------------------------------
# 步骤 5：周期快照增量抽取（水位线 = 已入库的最大 fetched_at）
# ------------------------------------------------------------------------------
def incremental_snapshot_load(conn):
    job = "INCR_SNAPSHOT"
    t0 = time.time()
    snap_file = RAW_DIR / "snapshots.csv"
    if not snap_file.exists():
        log_etl_job(conn, job, t0, 0, 0, 0, "SKIPPED", "data/raw/snapshots.csv 不存在")
        return
    cp = conn.execute("SELECT last_update_time FROM etl_checkpoint WHERE job_name = ?", (job,)).fetchone()
    watermark = cp[0] if cp and cp[0] else "1970-01-01 00:00:00"
    df = pd.read_csv(snap_file)
    new = df[df["fetched_at"] > watermark].sort_values(["appid", "fetched_at"])
    loaded = 0
    with conn:
        for r in new.to_dict("records"):
            prev = conn.execute(
                "SELECT total_reviews FROM fact_snapshot WHERE appid = ? AND fetched_at < ? ORDER BY fetched_at DESC LIMIT 1",
                (int(r["appid"]), r["fetched_at"])).fetchone()
            tot = none_if_nan(r["total_reviews"])
            delta = (int(tot) - prev[0]) if (prev and prev[0] is not None and tot is not None) else None
            cur = conn.execute(
                """INSERT OR IGNORE INTO fact_snapshot (fetched_at, round_no, appid, name, current_players, total_reviews,
                   new_reviews_since_prev, total_positive, final_price_usd, discount_percent, loaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["fetched_at"], int(r["round_no"]), int(r["appid"]), r["name"], none_if_nan(r["current_players"]),
                 None if tot is None else int(tot), delta, none_if_nan(r["total_positive"]),
                 none_if_nan(r["final_price"]), none_if_nan(r["discount_percent"]), now_str()))
            loaded += cur.rowcount
        new_wm = new["fetched_at"].max() if len(new) else watermark
        conn.execute("INSERT OR REPLACE INTO etl_checkpoint VALUES (?, NULL, ?, 'INCREMENTAL_SYNCED', ?)", (job, new_wm, now_str()))
    log_etl_job(conn, job, t0, len(new), loaded, 0, "SUCCESS", f"水位线 {watermark} -> {new_wm}，新增 {loaded} 行")


# ------------------------------------------------------------------------------
# 步骤 6：评测月度指数（政策上线判定结果入库）与 SQL 视图
# ------------------------------------------------------------------------------
def load_review_monthly_index(conn, policy):
    t0 = time.time()
    df = pd.read_csv(RAW_DIR / "review_monthly_index.csv")
    with conn:
        conn.execute("DELETE FROM ads_review_monthly_index;")
        for r in df.to_dict("records"):
            conn.execute("INSERT INTO ads_review_monthly_index VALUES (?, ?, ?, ?, ?, ?)",
                         (r["month"], r["median_index"], r["q25"], r["q75"], int(r["n_games"]),
                          int(r["month"] == policy["policy_month"])))
    log_etl_job(conn, "REVIEW_MONTHLY_INDEX", t0, len(df), len(df), 0, "SUCCESS",
                f"判定政策月份 {policy['policy_month']}，前后评测指数 {policy['median_index_before']} -> {policy['median_index_after']}")


def create_sql_views(conn):
    sql_text = (BASE_DIR / "sql" / "02_views_and_models_mysql.sql").read_text(encoding="utf-8")
    with conn:
        conn.executescript(sql_text)


def run_pipeline(demo_resume=False):
    conn = get_connection()
    init_dw_tables(conn)
    policy = load_policy()
    load_ods(conn)
    load_static_dims(conn, policy)
    if demo_resume:
        try:
            clean_ods_to_dwd(conn, policy, crash_after_batches=3)
        except SimulatedCrash as e:
            print(f"[ETL] 演示中断：检查点停在 {e}，重新调用清洗步骤续跑……", flush=True)
    clean_ods_to_dwd(conn, policy)
    build_dim_game(conn)
    apply_scd2_price_history(conn)
    incremental_snapshot_load(conn)
    load_review_monthly_index(conn, policy)
    create_sql_views(conn)
    print("=== ETL 完成 ===")
    for tbl in ["ods_ars_players_2018", "ods_steam_appdetails", "ods_steam_reviews_at_cutoff", "ods_steamcharts_monthly",
                "ods_recent_sales_truth", "ods_official_sales_truth_v2", "ods_review_language_counts",
                "ods_itad_price_history", "ods_itad_bundles", "ods_price_history_wayback", "dwd_game_sample", "dim_game", "dim_price_scd2",
                "fact_snapshot", "etl_clean_error_log", "etl_job_log"]:
        print(f"  {tbl:30s} {conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:7d} 行")
    for cohort, era, n, n_ok in conn.execute(
            """SELECT cohort, era_group, COUNT(*), SUM(model_eligible) FROM dwd_game_sample GROUP BY cohort, era_group"""):
        print(f"  样本 {cohort:10s} {era:12s} {n:5d} 条（可建模 {n_ok}）")
    conn.close()


if __name__ == "__main__":
    run_pipeline(demo_resume="--demo-resume" in sys.argv)
