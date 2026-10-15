-- =============================================================================
-- 索引与 EXPLAIN 对比（MySQL 8.0）
-- 先看没有索引时的执行计划，再建复合索引，再看一次。SQLite 上的实测对比见 data/cleaned/explain_comparison_report.txt
-- 说明：04 转储脚本只导数据、不带二级索引，所以第一段 EXPLAIN 看到的就是"无索引"的计划
-- =============================================================================
USE steam_sales_dw;

EXPLAIN
SELECT dm.method_name, dge.genre_cn_name, COUNT(*) AS game_cnt,
       ROUND(AVG(f.abs_pct_error) * 100, 2) AS mean_ape_pct,
       ROUND(SUM(f.within_30pct_flag) * 100.0 / COUNT(*), 2) AS hit_rate_30pct
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_estimation_method dm ON f.method_id = dm.method_id
WHERE f.cohort = 'truth' AND f.method_code IN ('M0', 'M2', 'M4', 'Fusion_AB')
GROUP BY dm.method_name, dge.genre_cn_name;

-- 过滤条件 cohort + method_code 打头，genre_id 放第三列，分组时也能用上
CREATE INDEX idx_fact_cohort_method ON fact_sales_estimation (cohort, method_code, genre_id);
CREATE INDEX idx_fact_game_key ON fact_sales_estimation (game_key);
CREATE INDEX idx_dwd_cohort_year ON dwd_game_sample (cohort, release_year);
CREATE INDEX idx_scd2_appid_dates ON dim_price_scd2 (appid, effective_start, effective_end);
CREATE INDEX idx_snapshot_appid_time ON fact_snapshot (appid, fetched_at);

EXPLAIN
SELECT dm.method_name, dge.genre_cn_name, COUNT(*) AS game_cnt,
       ROUND(AVG(f.abs_pct_error) * 100, 2) AS mean_ape_pct,
       ROUND(SUM(f.within_30pct_flag) * 100.0 / COUNT(*), 2) AS hit_rate_30pct
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_estimation_method dm ON f.method_id = dm.method_id
WHERE f.cohort = 'truth' AND f.method_code IN ('M0', 'M2', 'M4', 'Fusion_AB')
GROUP BY dm.method_name, dge.genre_cn_name;
