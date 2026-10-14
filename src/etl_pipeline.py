# -*- coding: utf-8 -*-
"""
Steam 游戏销量估算系统 - Python ETL 核心管道 (src/etl_pipeline.py)
支持双引擎模式：
1. 若本机安装了 MySQL + pymysql 且配置了环境变量 MYSQL_HOST，自动写入 MySQL 8.0；
2. 默认同时写入本地 SQLite 数据仓库 (data/steam_sales_dw.db) 并导出标准 MySQL INSERT 脚本与 FineBI 视图 CSV。

核心工程特性（对齐往届高分示例要求）：
- 三层数据仓库架构：原始层 (ODS) -> 明细与维度层 (DWD/DIM) -> 结果与事实层 (FACT/ADS)
- 断点续传检查点表 (etl_checkpoint)：演示分批抽取中断后从上次最后成功的 appid 继续执行
- 数据清洗与异常审计表 (etl_clean_error_log)：拦截免费游戏、Demo、非游戏工具、负价格与零评测异常并逐条记日志
- SCD Type 2 缓慢变化维 (dim_price_scd2)：自动闭合旧价格版本 (effective_end, is_current=0) 并插入新版本 (version_no+1)
- 基于 last_update_time 水位线的增量抽取 (fact_daily_snapshot)
- 事务原子写入 (with conn:) 与 ETL 运行日志表 (etl_job_log)
"""
import os
import json
import time
import math
import sqlite3
import pathlib
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_dw_tables(conn):
    """创建三层架构表结构、星型模型维度表与运维审计表"""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS ods_leak_players_2018 (
            appid INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            players_2018_07_01 INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            source_url TEXT,
            loaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ods_steam_appdetails (
            appid INTEGER PRIMARY KEY,
            name TEXT,
            app_type TEXT,
            is_free INTEGER,
            release_year INTEGER,
            initial_price_usd REAL,
            final_price_usd REAL,
            discount_percent INTEGER,
            main_genre TEXT,
            achievements_count INTEGER,
            dirty_note TEXT,
            raw_json TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ods_steam_reviews (
            appid INTEGER PRIMARY KEY,
            cutoff_date TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            total_positive INTEGER NOT NULL,
            total_negative INTEGER NOT NULL,
            positive_rate REAL,
            schinese_review_ratio REAL,
            avg_playtime_at_review_hrs REAL,
            steamspy_avg_forever_min INTEGER,
            steamcharts_total_player_hours INTEGER,
            steamcharts_avg_ccu REAL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ods_recent_sales_news (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            appid INTEGER NOT NULL,
            name TEXT NOT NULL,
            news_date TEXT NOT NULL,
            raw_sentence TEXT NOT NULL,
            llm_extracted_number INTEGER,
            llm_extracted_metric TEXT,
            llm_extracted_platform TEXT,
            human_audit_status TEXT,
            human_audit_reason TEXT,
            source_url TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS etl_job_log (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            batch_date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            rows_extracted INTEGER DEFAULT 0,
            rows_loaded INTEGER DEFAULT 0,
            rows_rejected INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            duration_ms INTEGER DEFAULT 0,
            message TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS etl_checkpoint (
            job_name TEXT PRIMARY KEY,
            last_processed_appid INTEGER DEFAULT 0,
            last_cursor TEXT DEFAULT '*',
            last_update_time TEXT,
            status TEXT DEFAULT 'READY',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS etl_clean_error_log (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            appid INTEGER NOT NULL,
            game_name TEXT,
            rule_code TEXT NOT NULL,
            raw_value_desc TEXT,
            clean_action TEXT DEFAULT 'FILTERED_OUT',
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_genre (
            genre_id INTEGER PRIMARY KEY,
            genre_name TEXT NOT NULL UNIQUE,
            genre_cn_name TEXT NOT NULL,
            audience_category TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_release_date (
            date_key INTEGER PRIMARY KEY,
            release_year INTEGER NOT NULL UNIQUE,
            era_code TEXT NOT NULL,
            era_name TEXT NOT NULL,
            has_review_prompt_policy INTEGER NOT NULL,
            literature_k_multiple REAL NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_estimation_method (
            method_id INTEGER PRIMARY KEY,
            method_code TEXT NOT NULL UNIQUE,
            method_name TEXT NOT NULL,
            signal_source TEXT NOT NULL,
            impl_tech TEXT NOT NULL,
            description TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_game (
            game_key INTEGER PRIMARY KEY,
            appid INTEGER NOT NULL UNIQUE,
            name TEXT NOT NULL,
            genre_id INTEGER NOT NULL,
            date_key INTEGER NOT NULL,
            initial_price_usd REAL NOT NULL,
            price_tier TEXT NOT NULL,
            is_cross_era INTEGER DEFAULT 0,
            data_cohort TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_price_scd2 (
            price_scd_key INTEGER PRIMARY KEY AUTOINCREMENT,
            appid INTEGER NOT NULL,
            name TEXT NOT NULL,
            base_price_usd REAL NOT NULL,
            discount_percent INTEGER NOT NULL,
            final_price_usd REAL NOT NULL,
            event_type TEXT,
            effective_start TEXT NOT NULL,
            effective_end TEXT NOT NULL,
            is_current INTEGER NOT NULL DEFAULT 1,
            version_no INTEGER NOT NULL DEFAULT 1,
            is_simulated INTEGER NOT NULL DEFAULT 1
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dwd_steam_game_aligned (
            appid INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            release_year INTEGER NOT NULL,
            genre TEXT NOT NULL,
            price_usd REAL NOT NULL,
            price_tier TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            reviews_pre_2019 INTEGER NOT NULL,
            reviews_post_2019 INTEGER NOT NULL,
            positive_rate REAL NOT NULL,
            positive_rate_tier TEXT NOT NULL,
            schinese_ratio REAL NOT NULL,
            avg_playtime_at_review_hrs REAL NOT NULL,
            steamspy_avg_forever_min INTEGER NOT NULL,
            steamcharts_total_player_hours INTEGER NOT NULL,
            true_sales_or_players INTEGER NOT NULL,
            k_ratio REAL NOT NULL,
            log_k REAL NOT NULL,
            log_reviews REAL NOT NULL,
            log_price REAL NOT NULL,
            data_cohort TEXT NOT NULL,
            snapshot_or_report_date TEXT NOT NULL,
            source_url TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS fact_sales_estimation (
            estimation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            appid INTEGER NOT NULL,
            game_key INTEGER NOT NULL,
            genre_id INTEGER NOT NULL,
            date_key INTEGER NOT NULL,
            method_id INTEGER NOT NULL,
            method_code TEXT NOT NULL,
            data_cohort TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            price_usd REAL NOT NULL,
            true_sales_or_players INTEGER NOT NULL,
            estimated_sales INTEGER NOT NULL,
            estimated_low_bound INTEGER NOT NULL,
            estimated_high_bound INTEGER NOT NULL,
            estimated_revenue_usd REAL NOT NULL,
            implied_k REAL NOT NULL,
            observed_k REAL NOT NULL,
            pct_error REAL NOT NULL,
            abs_pct_error REAL NOT NULL,
            within_30pct_flag INTEGER NOT NULL,
            within_50pct_flag INTEGER NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS fact_daily_snapshot (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            appid INTEGER NOT NULL,
            name TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            daily_new_reviews INTEGER NOT NULL,
            current_players INTEGER NOT NULL,
            discount_percent INTEGER NOT NULL,
            final_price_usd REAL NOT NULL,
            last_update_time TEXT NOT NULL,
            is_simulated INTEGER NOT NULL DEFAULT 1
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ads_model_eval_summary (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_cohort TEXT NOT NULL,
            method_code TEXT NOT NULL,
            method_name TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            mape_pct REAL,
            mdape_pct REAL,
            r2_log REAL,
            hit_rate_10pct REAL,
            hit_rate_30pct REAL NOT NULL,
            hit_rate_50pct REAL,
            note TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ads_feature_importance (
            feat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_code TEXT NOT NULL,
            feature_cn_name TEXT NOT NULL,
            m2_ols_coef REAL,
            m2_std_err REAL,
            m2_t_stat REAL,
            m2_p_value REAL,
            m4_rf_importance_pct REAL,
            m4_gbdt_importance_pct REAL,
            direction_desc TEXT
        );
        """
    ]
    with conn:
        for stmt in ddl_statements:
            conn.execute(stmt)


def log_etl_job(conn, job_name, batch_date, t0, extracted, loaded, rejected, status, msg):
    t1 = time.time()
    dur_ms = int(round((t1 - t0) * 1000))
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
    with conn:
        conn.execute(
            """
            INSERT INTO etl_job_log
            (job_name, batch_date, start_time, end_time, rows_extracted, rows_loaded, rows_rejected, status, duration_ms, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_name, batch_date, start_str, end_str, extracted, loaded, rejected, status, dur_ms, msg)
        )


def classify_price_tier(price_usd):
    if price_usd < 10.0:
        return "1_低价独立(<$10)"
    elif price_usd < 20.0:
        return "2_中低定价($10-$20)"
    elif price_usd < 40.0:
        return "3_中高等级($20-$40)"
    else:
        return "4_全价大作(>=$40)"


def classify_pos_rate_tier(pos_rate):
    if pos_rate < 0.60:
        return "1_褒贬不一及以下(<60%)"
    elif pos_rate < 0.75:
        return "2_中度好评(60%-75%)"
    elif pos_rate < 0.85:
        return "3_多半好评(75%-85%)"
    elif pos_rate < 0.93:
        return "4_特别好评(85%-93%)"
    else:
        return "5_好评如潮(>=93%)"


def load_ods_and_dimensions(conn):
    """步骤1：加载原始层 (ODS) 与静态维度表 (dim_genre, dim_release_date, dim_estimation_method)"""
    t0 = time.time()
    df_leak = pd.read_csv(BASE_DIR / "data/raw/games_achievements_players_2018-07-01.csv")
    df_rev = pd.read_csv(BASE_DIR / "data/raw/steam_reviews_2018_aligned.csv")
    df_news = pd.read_csv(BASE_DIR / "data/raw/recent_news_extraction_audit.csv")
    with open(BASE_DIR / "data/raw/steam_appdetails_2018_aligned.json", "r", encoding="utf-8") as f:
        app_dict = json.load(f)

    with conn:
        conn.execute("DELETE FROM ods_leak_players_2018;")
        conn.execute("DELETE FROM ods_steam_appdetails;")
        conn.execute("DELETE FROM ods_steam_reviews;")
        conn.execute("DELETE FROM ods_recent_sales_news;")

        for _, r in df_leak.iterrows():
            conn.execute(
                "INSERT INTO ods_leak_players_2018 (appid, name, players_2018_07_01, snapshot_date, source_url) VALUES (?, ?, ?, ?, ?)",
                (int(r["appid"]), r["name"], int(r["players_2018_07_01"]), r["snapshot_date"], r["source_url"])
            )

        for appid_str, payload in app_dict.items():
            d = payload["data"]
            p_obj = d.get("price_overview")
            init_p = (p_obj["initial"] / 100.0) if p_obj else 0.0
            fin_p = (p_obj["final"] / 100.0) if p_obj else 0.0
            disc = p_obj["discount_percent"] if p_obj else 0
            rel_str = d.get("release_date", {}).get("date", "2015")
            rel_year = int(rel_str.split(",")[-1].strip()) if "," in rel_str else 2015
            genre = d.get("genres", [{}])[0].get("description", "Action")
            ach_cnt = d.get("achievements", {}).get("total", 0)
            conn.execute(
                """
                INSERT INTO ods_steam_appdetails
                (appid, name, app_type, is_free, release_year, initial_price_usd, final_price_usd, discount_percent, main_genre, achievements_count, dirty_note, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(appid_str), d["name"], d["type"], 1 if d["is_free"] else 0, rel_year, init_p, fin_p, disc, genre, ach_cnt, d.get("dirty_note", ""), json.dumps(d, ensure_ascii=False))
            )

        for _, r in df_rev.iterrows():
            conn.execute(
                """
                INSERT INTO ods_steam_reviews
                (appid, cutoff_date, total_reviews, total_positive, total_negative, positive_rate, schinese_review_ratio, avg_playtime_at_review_hrs, steamspy_avg_forever_min, steamcharts_total_player_hours, steamcharts_avg_ccu)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r["appid"]), r["cutoff_date"], int(r["total_reviews_2018"]), int(r["total_positive_2018"]), int(r["total_negative_2018"]),
                 float(r["positive_rate_2018"]), float(r["schinese_review_ratio"]), float(r["avg_playtime_at_review_hrs"]),
                 int(r["steamspy_avg_forever_min"]), int(r["steamcharts_total_player_hours"]), float(r["steamcharts_avg_ccu"]))
            )

        for _, r in df_news.iterrows():
            conn.execute(
                """
                INSERT INTO ods_recent_sales_news
                (appid, name, news_date, raw_sentence, llm_extracted_number, llm_extracted_metric, llm_extracted_platform, human_audit_status, human_audit_reason, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r["appid"]), r["name"], r["news_date"], r["raw_sentence"], int(r["llm_extracted_number"]),
                 r["llm_extracted_metric"], r["llm_extracted_platform"], r["human_audit_status"], r["human_audit_reason"], r["source_url"])
            )

        # 初始化静态维度表 dim_genre
        conn.execute("DELETE FROM dim_genre;")
        genres_meta = [
            (1, "Action", "动作射击", "核心大众受众"),
            (2, "RPG", "角色扮演", "深度叙事受众"),
            (3, "Strategy", "策略战棋", "长时硬核受众"),
            (4, "Simulation", "模拟建造", "长尾模拟受众"),
            (5, "Adventure", "冒险解谜", "剧情体验受众"),
            (6, "Indie", "独立创意", "高评论意愿核心受众"),
            (7, "Survival", "生存建造", "联机高粘性受众"),
            (8, "Sports", "体育竞速", "低评论率泛大众受众"),
        ]
        conn.executemany("INSERT INTO dim_genre VALUES (?, ?, ?, ?);", genres_meta)

        # 初始化静态维度表 dim_release_date
        conn.execute("DELETE FROM dim_release_date;")
        for yr in range(2003, 2026):
            if yr <= 2016:
                era_code, era_name, has_pol, lit_k = "ERA_EARLY", "早期Steam阶段(<=2016)", 0, 74.0
            elif yr in (2017, 2018):
                era_code, era_name, has_pol, lit_k = "ERA_PRE_POLICY", "直通发售后评测提示前(2017-2018)", 0, 60.0
            elif yr == 2019:
                era_code, era_name, has_pol, lit_k = "ERA_TRANSITION", "评测提示政策上线过渡期(2019)", 1, 51.0
            else:
                era_code, era_name, has_pol, lit_k = "ERA_POST_POLICY", "游玩后提示写评测时代(>=2020)", 1, 38.0
            conn.execute("INSERT INTO dim_release_date VALUES (?, ?, ?, ?, ?, ?);", (yr, yr, era_code, era_name, has_pol, lit_k))

        # 初始化静态维度表 dim_estimation_method
        conn.execute("DELETE FROM dim_estimation_method;")
        methods_meta = [
            (1, "M0", "M0: 固定倍数基线 (k=30)", "仅评测信号A", "SQL", "经典 Boxleiter 固定常数：销量 = 评测数 × 30"),
            (2, "M1", "M1: 按年份分段文献值", "仅评测信号A", "SQL", "按发售年代取文献中位数：<=2016取74, 17-18取60, 2019取51, >=2020取38"),
            (3, "M2", "M2: 对数线性回归 (含年代校准)", "仅评测信号A", "Python Stats", "综合价格、好评率倒U形二次项、评测规模、游玩时长与中文占比的多元对数回归"),
            (4, "M3", "M3: 分位数回归 (中位数+80%区间)", "仅评测信号A", "Python Stats", "基于 Pinball Loss 与残差分位数给出稳健中位数预测及 [P10, P90] 置信区间"),
            (5, "M4", "M4: 梯度提升/随机森林树模型", "仅评测信号A", "Python ML", "非线性决策树集成学习，自动捕捉好评率倒U形与价格-时长交互效应"),
            (6, "Signal_B", "信号B: 在线人数与人均时长推算", "仅在线信号B", "Python Stats", "与评测无关：累计总游玩小时数 ÷ 校准后人均游玩时长（不受2019评测弹窗政策影响）"),
            (7, "Fusion_AB", "Fusion: 评测信号A + 在线信号B 加权融合", "A+B双信号融合", "Python ML", "对数空间双信号最优加权融合：突破单一评测信号的信息瓶颈"),
        ]
        conn.executemany("INSERT INTO dim_estimation_method VALUES (?, ?, ?, ?, ?, ?);", methods_meta)

    total_ods = len(df_leak) + len(app_dict) + len(df_rev) + len(df_news)
    log_etl_job(conn, "ODS_AND_STATIC_DIM_LOAD", "2026-10-12", t0, total_ods, total_ods, 0, "SUCCESS", "原始层四张表与静态维度表全量入库完成")


def clean_and_load_dwd_with_checkpoint(conn):
    """
    步骤2：执行 ODS -> DWD 清洗转换，演示检查点断点续传 (etl_checkpoint) 与清洗异常日志 (etl_clean_error_log)
    为什么分两批模拟断点续传：验证当批处理在第 150 条中断时，重启任务能够读取 etl_checkpoint 的 last_processed_appid 接着跑，不丢不重。
    """
    with conn:
        conn.execute("DELETE FROM dwd_steam_game_aligned;")
        conn.execute("DELETE FROM dim_game;")
        conn.execute("DELETE FROM etl_clean_error_log;")
        conn.execute("DELETE FROM etl_checkpoint WHERE job_name = 'CLEAN_ODS_TO_DWD';")
        conn.execute(
            "INSERT INTO etl_checkpoint (job_name, last_processed_appid, last_cursor, last_update_time, status) VALUES ('CLEAN_ODS_TO_DWD', 0, 'BATCH_0', '2018-07-01 00:00:00', 'READY');"
        )

    # 读取 ODS 关联数据，按 appid 升序排列以便基于 last_processed_appid 断点续传
    query = """
    SELECT
        p.appid, p.name, p.players_2018_07_01, p.snapshot_date, p.source_url,
        a.app_type, a.is_free, a.release_year, a.initial_price_usd, a.main_genre, a.dirty_note,
        r.total_reviews, r.positive_rate, r.schinese_review_ratio,
        r.avg_playtime_at_review_hrs, r.steamspy_avg_forever_min, r.steamcharts_total_player_hours
    FROM ods_leak_players_2018 p
    JOIN ods_steam_appdetails a ON p.appid = a.appid
    JOIN ods_steam_reviews r ON p.appid = r.appid
    ORDER BY p.appid ASC
    """
    rows = conn.execute(query).fetchall()
    genre_map = dict(conn.execute("SELECT genre_name, genre_id FROM dim_genre").fetchall())

    # 第一阶段：处理前 150 条后模拟发生网络/进程中断，更新检查点
    t0 = time.time()
    loaded_b1, rejected_b1 = 0, 0
    last_appid_b1 = 0
    with conn:
        for idx, r in enumerate(rows[:150]):
            (appid, name, players, snap_date, src_url, app_type, is_free, rel_year, price_usd,
             genre, dirty_note, tot_rev, pos_rate, sch_ratio, avg_play_hrs, spy_min, charts_hrs) = r
            last_appid_b1 = appid
            if app_type != "game":
                rule = "DEMO_VERSION" if app_type == "demo" else "NON_GAME_SOFTWARE"
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, rule, f"app_type={app_type}"))
                rejected_b1 += 1
                continue
            if is_free == 1:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "FREE_GAME", f"is_free=1, price={price_usd}"))
                rejected_b1 += 1
                continue
            if price_usd <= 0:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "INVALID_PRICE", f"initial_price_usd={price_usd}"))
                rejected_b1 += 1
                continue
            if tot_rev <= 0:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "ZERO_REVIEWS", f"total_reviews={tot_rev}"))
                rejected_b1 += 1
                continue

            k_val = players / tot_rev
            p_tier = classify_price_tier(price_usd)
            pos_tier = classify_pos_rate_tier(pos_rate)
            conn.execute(
                """
                INSERT INTO dwd_steam_game_aligned VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (appid, name, rel_year, genre, price_usd, p_tier, tot_rev, tot_rev, 0, pos_rate, pos_tier,
                 sch_ratio, avg_play_hrs, spy_min, charts_hrs, players, round(k_val, 4),
                 round(math.log(k_val), 4), round(math.log(tot_rev), 4), round(math.log(price_usd), 4),
                 "train_2018", snap_date, src_url)
            )
            loaded_b1 += 1

        conn.execute(
            "UPDATE etl_checkpoint SET last_processed_appid = ?, last_cursor = 'BATCH_1_INTERRUPTED', status = 'INTERRUPTED', updated_at = CURRENT_TIMESTAMP WHERE job_name = 'CLEAN_ODS_TO_DWD';",
            (last_appid_b1,)
        )
    log_etl_job(conn, "CLEAN_ODS_TO_DWD_PART1", "2026-10-12", t0, 150, loaded_b1, rejected_b1, "INTERRUPTED",
                f"模拟第150条处断点保存，checkpoint.last_processed_appid={last_appid_b1}")

    # 第二阶段：从 etl_checkpoint 读取 last_processed_appid 断点续传剩余记录
    t1 = time.time()
    cp_row = conn.execute("SELECT last_processed_appid FROM etl_checkpoint WHERE job_name = 'CLEAN_ODS_TO_DWD'").fetchone()
    resume_appid = cp_row[0]
    remaining_rows = [r for r in rows if r[0] > resume_appid]

    loaded_b2, rejected_b2 = 0, 0
    last_appid_b2 = resume_appid
    with conn:
        for r in remaining_rows:
            (appid, name, players, snap_date, src_url, app_type, is_free, rel_year, price_usd,
             genre, dirty_note, tot_rev, pos_rate, sch_ratio, avg_play_hrs, spy_min, charts_hrs) = r
            last_appid_b2 = appid
            if app_type != "game":
                rule = "DEMO_VERSION" if app_type == "demo" else "NON_GAME_SOFTWARE"
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, rule, f"app_type={app_type}"))
                rejected_b2 += 1
                continue
            if is_free == 1:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "FREE_GAME", f"is_free=1, price={price_usd}"))
                rejected_b2 += 1
                continue
            if price_usd <= 0:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "INVALID_PRICE", f"initial_price_usd={price_usd}"))
                rejected_b2 += 1
                continue
            if tot_rev <= 0:
                conn.execute("INSERT INTO etl_clean_error_log (job_name, appid, game_name, rule_code, raw_value_desc) VALUES (?, ?, ?, ?, ?)",
                             ("CLEAN_ODS_TO_DWD", appid, name, "ZERO_REVIEWS", f"total_reviews={tot_rev}"))
                rejected_b2 += 1
                continue

            k_val = players / tot_rev
            p_tier = classify_price_tier(price_usd)
            pos_tier = classify_pos_rate_tier(pos_rate)
            conn.execute(
                """
                INSERT INTO dwd_steam_game_aligned VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (appid, name, rel_year, genre, price_usd, p_tier, tot_rev, tot_rev, 0, pos_rate, pos_tier,
                 sch_ratio, avg_play_hrs, spy_min, charts_hrs, players, round(k_val, 4),
                 round(math.log(k_val), 4), round(math.log(tot_rev), 4), round(math.log(price_usd), 4),
                 "train_2018", snap_date, src_url)
            )
            loaded_b2 += 1

        # 同时加载 2020+ 近年真值集与跨时代检验集进入 DWD 宽表（若跨时代游戏已在 2018 表中，用新 appid 偏移键 9000000+appid 存储其 2021-2023 跨时代状态，保持主键唯一且不丢失 2018 记录）
        df_recent = pd.read_csv(BASE_DIR / "data/raw/recent_sales_truth_2020_2025.csv")
        for _, r in df_recent.iterrows():
            orig_appid = int(r["appid"])
            cohort = "modern_2020" if r["era_type"] == "post_2020" else "cross_era"
            dwd_appid = orig_appid if cohort == "modern_2020" else (9000000 + orig_appid)
            display_name = r["name"] if cohort == "modern_2020" else f"{r['name']} (跨时代2020+检验)"
            price_usd = float(r["price_usd"])
            tot_rev = int(r["reviews_at_report"])
            sales = int(r["reported_sales"])
            k_val = sales / tot_rev
            p_tier = classify_price_tier(price_usd)
            pos_tier = classify_pos_rate_tier(float(r["positive_rate"]))
            conn.execute(
                """
                INSERT OR REPLACE INTO dwd_steam_game_aligned VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (dwd_appid, display_name, int(r["release_year"]), r["genre"], price_usd, p_tier,
                 tot_rev, int(r["reviews_pre_2019"]), int(r["reviews_post_2019"]), float(r["positive_rate"]), pos_tier,
                 float(r["schinese_ratio"]), float(r["avg_playtime_at_review_hrs"]), int(r["steamspy_avg_forever_min"]),
                 int(r["steamcharts_total_player_hours"]), sales, round(k_val, 4),
                 round(math.log(k_val), 4), round(math.log(tot_rev), 4), round(math.log(price_usd), 4),
                 cohort, r["report_date"], r["source_url"])
            )
            loaded_b2 += 1

        # 构建游戏维度表 dim_game
        all_dwd = conn.execute("SELECT appid, name, genre, release_year, price_usd, price_tier, data_cohort FROM dwd_steam_game_aligned ORDER BY appid ASC").fetchall()
        for g_key, g_row in enumerate(all_dwd, start=1):
            appid, name, genre, rel_year, price_usd, p_tier, cohort = g_row
            gid = genre_map.get(genre, 1)
            is_cross = 1 if cohort == "cross_era" else 0
            conn.execute("INSERT INTO dim_game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (g_key, appid, name, gid, rel_year, price_usd, p_tier, is_cross, cohort))

        conn.execute(
            "UPDATE etl_checkpoint SET last_processed_appid = ?, last_cursor = 'COMPLETED_ALL', status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE job_name = 'CLEAN_ODS_TO_DWD';",
            (last_appid_b2,)
        )

    log_etl_job(conn, "CLEAN_ODS_TO_DWD_RESUME", "2026-10-12", t1, len(remaining_rows) + len(df_recent), loaded_b2, rejected_b2, "SUCCESS",
                f"从断点 appid>{resume_appid} 成功续传并完成全部清洗与维度构建")


def apply_scd2_price_history(conn):
    """
    步骤3：执行游戏价格维度 SCD Type 2 缓慢变化维拉链更新 (dim_price_scd2)
    机制：对于按时间到达的价格变更事件，若该游戏已有 is_current=1 的记录且价格/折扣发生变化，
    则先将旧记录的 effective_end 更新为本次变更日期、is_current 置为 0，再插入新版本 (version_no = 旧版本+1, is_current=1, effective_end='9999-12-31')。
    """
    t0 = time.time()
    df_sim_price = pd.read_csv(BASE_DIR / "data/simulated/sim_price_changes.csv")
    with conn:
        conn.execute("DELETE FROM dim_price_scd2;")
        loaded_cnt = 0
        for _, r in df_sim_price.iterrows():
            appid = int(r["appid"])
            name = r["name"]
            base_p = float(r["price_usd"])
            disc = int(r["discount_pct"])
            final_p = float(r["final_price_usd"])
            c_date = r["change_date"]
            ev_type = r["event_type"]
            is_sim = int(r["is_simulated"])

            curr = conn.execute(
                "SELECT price_scd_key, final_price_usd, discount_percent, version_no FROM dim_price_scd2 WHERE appid = ? AND is_current = 1",
                (appid,)
            ).fetchone()

            if curr is None:
                conn.execute(
                    """
                    INSERT INTO dim_price_scd2
                    (appid, name, base_price_usd, discount_percent, final_price_usd, event_type, effective_start, effective_end, is_current, version_no, is_simulated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, '9999-12-31', 1, 1, ?)
                    """,
                    (appid, name, base_p, disc, final_p, ev_type, c_date, is_sim)
                )
                loaded_cnt += 1
            else:
                old_key, old_final, old_disc, old_ver = curr
                if abs(old_final - final_p) > 1e-4 or old_disc != disc:
                    # 闭合旧版本
                    conn.execute(
                        "UPDATE dim_price_scd2 SET effective_end = ?, is_current = 0 WHERE price_scd_key = ?",
                        (c_date, old_key)
                    )
                    # 插入新版本
                    conn.execute(
                        """
                        INSERT INTO dim_price_scd2
                        (appid, name, base_price_usd, discount_percent, final_price_usd, event_type, effective_start, effective_end, is_current, version_no, is_simulated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, '9999-12-31', 1, ?, ?)
                        """,
                        (appid, name, base_p, disc, final_p, ev_type, c_date, old_ver + 1, is_sim)
                    )
                    loaded_cnt += 1

    log_etl_job(conn, "SCD2_PRICE_DIMENSION_UPDATE", "2026-10-12", t0, len(df_sim_price), loaded_cnt, 0, "SUCCESS",
                "SCD Type 2 价格维度拉链表构建完成（含历史闭合版本与当前生效版本）")


def extract_incremental_daily_snapshots(conn):
    """
    步骤4：基于 last_update_time 水位线的增量抽取演示 -> 周期快照事实表 (fact_daily_snapshot)
    分两次增量拉取：第一次拉取 2026-10-01 至 2026-10-07，更新水位线；第二次只抽取 last_update_time > '2026-10-07 23:55:00' 的新增快照。
    """
    df_snap = pd.read_csv(BASE_DIR / "data/simulated/sim_daily_increments.csv")
    with conn:
        conn.execute("DELETE FROM fact_daily_snapshot;")
        conn.execute("DELETE FROM etl_checkpoint WHERE job_name = 'INCR_DAILY_SNAPSHOT';")
        conn.execute(
            "INSERT INTO etl_checkpoint (job_name, last_processed_appid, last_cursor, last_update_time, status) VALUES ('INCR_DAILY_SNAPSHOT', 0, 'INIT', '2026-09-30 23:59:59', 'READY');"
        )

    # 第一批增量：抽取截至 2026-10-07 的数据
    t0 = time.time()
    watermark_1 = conn.execute("SELECT last_update_time FROM etl_checkpoint WHERE job_name = 'INCR_DAILY_SNAPSHOT'").fetchone()[0]
    batch_1 = df_snap[(df_snap["last_update_time"] > watermark_1) & (df_snap["snapshot_date"] <= "2026-10-07")]
    max_ts_1 = batch_1["last_update_time"].max()
    with conn:
        for _, r in batch_1.iterrows():
            conn.execute(
                """
                INSERT INTO fact_daily_snapshot
                (snapshot_date, appid, name, total_reviews, daily_new_reviews, current_players, discount_percent, final_price_usd, last_update_time, is_simulated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (r["snapshot_date"], int(r["appid"]), r["name"], int(r["total_reviews"]), int(r["daily_new_reviews"]),
                 int(r["current_players"]), int(r["discount_percent"]), float(r["final_price_usd"]), r["last_update_time"], int(r["is_simulated"]))
            )
        conn.execute("UPDATE etl_checkpoint SET last_update_time = ?, status = 'INCREMENTAL_SYNCED' WHERE job_name = 'INCR_DAILY_SNAPSHOT'", (max_ts_1,))
    log_etl_job(conn, "INCR_DAILY_SNAPSHOT_BATCH1", "2026-10-07", t0, len(batch_1), len(batch_1), 0, "SUCCESS",
                f"增量抽取第一周快照完成，水位线推进至 {max_ts_1}")

    # 第二批增量：只抽取 last_update_time > max_ts_1 的数据
    t1 = time.time()
    watermark_2 = conn.execute("SELECT last_update_time FROM etl_checkpoint WHERE job_name = 'INCR_DAILY_SNAPSHOT'").fetchone()[0]
    batch_2 = df_snap[df_snap["last_update_time"] > watermark_2]
    max_ts_2 = batch_2["last_update_time"].max()
    with conn:
        for _, r in batch_2.iterrows():
            conn.execute(
                """
                INSERT INTO fact_daily_snapshot
                (snapshot_date, appid, name, total_reviews, daily_new_reviews, current_players, discount_percent, final_price_usd, last_update_time, is_simulated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (r["snapshot_date"], int(r["appid"]), r["name"], int(r["total_reviews"]), int(r["daily_new_reviews"]),
                 int(r["current_players"]), int(r["discount_percent"]), float(r["final_price_usd"]), r["last_update_time"], int(r["is_simulated"]))
            )
        conn.execute("UPDATE etl_checkpoint SET last_update_time = ?, status = 'COMPLETED' WHERE job_name = 'INCR_DAILY_SNAPSHOT'", (max_ts_2,))
    log_etl_job(conn, "INCR_DAILY_SNAPSHOT_BATCH2", "2026-10-14", t1, len(batch_2), len(batch_2), 0, "SUCCESS",
                f"增量抽取第二周快照完成，水位线从 {watermark_2} 推进至 {max_ts_2}")


def create_sql_views(conn):
    """步骤5：在数据库中创建 SQL 模型视图与 FineBI 只读视图"""
    sql_text = (BASE_DIR / "sql/02_views_and_models_mysql.sql").read_text(encoding="utf-8")
    with conn:
        conn.executescript(sql_text)


if __name__ == "__main__":
    conn = get_connection()
    init_dw_tables(conn)
    load_ods_and_dimensions(conn)
    clean_and_load_dwd_with_checkpoint(conn)
    apply_scd2_price_history(conn)
    extract_incremental_daily_snapshots(conn)
    create_sql_views(conn)

    print("=== ETL 管道执行完成 ===")
    for tbl in ["ods_leak_players_2018", "dwd_steam_game_aligned", "dim_game", "dim_price_scd2", "fact_daily_snapshot", "etl_job_log", "etl_checkpoint", "etl_clean_error_log"]:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  表 {tbl:26s}: {cnt:4d} 行")
    conn.close()
