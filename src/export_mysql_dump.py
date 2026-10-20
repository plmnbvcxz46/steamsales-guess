# -*- coding: utf-8 -*-
"""
把 SQLite 数据仓库导出成 MySQL 的 INSERT 脚本 (src/export_mysql_dump.py) -> sql/04_full_data_dump_mysql.sql

导入顺序：01 建表 -> 04 数据 -> 02 视图 -> 03 索引与 EXPLAIN
为什么要单独写导出：本机没装 MySQL 时，整条流程在 SQLite 上跑；交作业、连 FineBI 时再一键灌进 MySQL。
"""
import sqlite3
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"
OUT = BASE_DIR / "sql" / "04_full_data_dump_mysql.sql"

# 按外键依赖排好的顺序：先维度、后事实
TABLES = [
    "ods_ars_players_2018", "ods_steam_appdetails", "ods_steam_reviews_at_cutoff", "ods_steamcharts_monthly",
    "ods_recent_sales_truth", "ods_price_history_wayback", "ods_review_monthly_policy_scan",
    # ods_itad_price_history 有 30 多万行，放进转储会让文件翻倍；原始 CSV 在 data/raw/ 里，需要时用 LOAD DATA 导入
    "ods_official_sales_truth_v2", "ods_review_language_counts", "ods_itad_bundles",
    "etl_job_log", "etl_checkpoint", "etl_clean_error_log",
    "dim_genre", "dim_release_date", "dim_estimation_method", "dim_game", "dim_price_scd2",
    "dwd_game_sample", "fact_sales_estimation", "fact_snapshot",
    "ads_review_monthly_index", "ads_model_eval_summary", "ads_feature_importance", "ads_policy_calibration",
]
BATCH = 500


def sql_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{s}'"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    total = 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("-- 由 src/export_mysql_dump.py 从 data/steam_sales_dw.db 自动生成，请勿手改\n")
        f.write("USE steam_sales_dw;\nSET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS = 0;\n\n")
        for t in TABLES:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            rows = conn.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
            f.write(f"-- {t}: {len(rows)} 行\nDELETE FROM {t};\n")
            for i in range(0, len(rows), BATCH):
                chunk = rows[i:i + BATCH]
                values = ",\n".join("(" + ", ".join(sql_value(v) for v in r) + ")" for r in chunk)
                f.write(f"INSERT INTO {t} ({', '.join(cols)}) VALUES\n{values};\n")
            f.write("\n")
            total += len(rows)
        f.write("SET FOREIGN_KEY_CHECKS = 1;\n")
    conn.close()
    print(f"[dump] {len(TABLES)} 张表共 {total} 行 -> sql/04_full_data_dump_mysql.sql")


if __name__ == "__main__":
    main()
