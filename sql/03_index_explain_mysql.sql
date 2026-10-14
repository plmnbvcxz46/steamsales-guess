-- 索引优化与 EXPLAIN 执行计划对比脚本
EXPLAIN
SELECT dm.method_name, dge.genre_cn_name, COUNT(*) AS game_cnt, AVG(f.abs_pct_error) AS mean_ape
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_estimation_method dm ON f.method_id = dm.method_id
WHERE f.data_cohort = 'modern_2020' AND f.method_code IN ('M0', 'M2', 'M4', 'Fusion_AB')
GROUP BY dm.method_name, dge.genre_cn_name;

CREATE INDEX idx_fact_cohort_method ON fact_sales_estimation(data_cohort, method_code, genre_id);
CREATE INDEX idx_fact_game_key ON fact_sales_estimation(game_key);
CREATE INDEX idx_dwd_cohort_genre_year ON dwd_steam_game_aligned(data_cohort, genre, release_year);
CREATE INDEX idx_scd2_appid_dates ON dim_price_scd2(appid, effective_start, effective_end, is_current);
CREATE INDEX idx_snap_appid_date ON fact_daily_snapshot(appid, snapshot_date);

EXPLAIN
SELECT dm.method_name, dge.genre_cn_name, COUNT(*) AS game_cnt, AVG(f.abs_pct_error) AS mean_ape
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_estimation_method dm ON f.method_id = dm.method_id
WHERE f.data_cohort = 'modern_2020' AND f.method_code IN ('M0', 'M2', 'M4', 'Fusion_AB')
GROUP BY dm.method_name, dge.genre_cn_name;
