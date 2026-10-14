-- SQL 实现 M0/M1、窗口函数计算 k 中位数与排名、以及 FineBI 只读视图定义
DROP VIEW IF EXISTS vw_sql_m0_m1_estimation;
CREATE VIEW vw_sql_m0_m1_estimation AS
SELECT
    d.appid,
    d.name,
    d.release_year,
    d.genre,
    d.price_usd,
    d.price_tier,
    d.total_reviews,
    d.positive_rate,
    d.true_sales_or_players,
    d.k_ratio AS observed_k,
    30.0 AS m0_k,
    CAST(ROUND(d.total_reviews * 30.0) AS INTEGER) AS m0_estimated_sales,
    ROUND(ABS(d.total_reviews * 30.0 - d.true_sales_or_players) * 1.0 / d.true_sales_or_players, 4) AS m0_ape,
    CASE WHEN ABS(d.total_reviews * 30.0 - d.true_sales_or_players) * 1.0 / d.true_sales_or_players <= 0.30 THEN 1 ELSE 0 END AS m0_within_30pct,
    CASE
        WHEN d.release_year <= 2013 THEN 68.0
        WHEN d.release_year BETWEEN 2014 AND 2016 THEN 56.0
        WHEN d.release_year IN (2017, 2018) THEN 46.0
        WHEN d.release_year = 2019 THEN 40.0
        ELSE 35.0
    END AS m1_k,
    CAST(ROUND(d.total_reviews * (
        CASE
            WHEN d.release_year <= 2013 THEN 68.0
            WHEN d.release_year BETWEEN 2014 AND 2016 THEN 56.0
            WHEN d.release_year IN (2017, 2018) THEN 46.0
            WHEN d.release_year = 2019 THEN 40.0
            ELSE 35.0
        END
    )) AS INTEGER) AS m1_estimated_sales,
    d.data_cohort
FROM dwd_steam_game_aligned d;

DROP VIEW IF EXISTS vw_finebi_k_median_by_year;
CREATE VIEW vw_finebi_k_median_by_year AS
WITH ranked_k AS (
    SELECT
        release_year,
        k_ratio,
        ROW_NUMBER() OVER (PARTITION BY release_year ORDER BY k_ratio ASC) AS rn,
        COUNT(*) OVER (PARTITION BY release_year) AS cnt
    FROM dwd_steam_game_aligned
    WHERE data_cohort IN ('train_2018', 'modern_2020')
),
year_median AS (
    SELECT
        release_year,
        ROUND(AVG(k_ratio), 2) AS median_k
    FROM ranked_k
    WHERE rn * 2 IN (cnt, cnt + 1, cnt + 2)
    GROUP BY release_year
),
year_stats AS (
    SELECT
        release_year,
        COUNT(*) AS game_count,
        ROUND(AVG(k_ratio), 2) AS mean_k,
        ROUND(MIN(k_ratio), 2) AS min_k,
        ROUND(MAX(k_ratio), 2) AS max_k
    FROM dwd_steam_game_aligned
    WHERE data_cohort IN ('train_2018', 'modern_2020')
    GROUP BY release_year
)
SELECT
    s.release_year,
    r.era_name,
    r.has_review_prompt_policy,
    s.game_count,
    m.median_k,
    s.mean_k,
    s.min_k,
    s.max_k,
    r.literature_k_multiple
FROM year_stats s
JOIN year_median m ON s.release_year = m.release_year
LEFT JOIN dim_release_date r ON s.release_year = r.release_year
ORDER BY s.release_year ASC;

DROP VIEW IF EXISTS vw_finebi_k_median_by_genre_price;
CREATE VIEW vw_finebi_k_median_by_genre_price AS
WITH genre_ranked AS (
    SELECT
        genre AS dimension_value,
        '游戏类型' AS dimension_type,
        k_ratio,
        positive_rate,
        price_usd,
        ROW_NUMBER() OVER (PARTITION BY genre ORDER BY k_ratio ASC) AS rn,
        COUNT(*) OVER (PARTITION BY genre) AS cnt
    FROM dwd_steam_game_aligned
    WHERE data_cohort = 'train_2018'
),
price_ranked AS (
    SELECT
        price_tier AS dimension_value,
        '价格分档' AS dimension_type,
        k_ratio,
        positive_rate,
        price_usd,
        ROW_NUMBER() OVER (PARTITION BY price_tier ORDER BY k_ratio ASC) AS rn,
        COUNT(*) OVER (PARTITION BY price_tier) AS cnt
    FROM dwd_steam_game_aligned
    WHERE data_cohort = 'train_2018'
),
pos_ranked AS (
    SELECT
        positive_rate_tier AS dimension_value,
        '好评率区间' AS dimension_type,
        k_ratio,
        positive_rate,
        price_usd,
        ROW_NUMBER() OVER (PARTITION BY positive_rate_tier ORDER BY k_ratio ASC) AS rn,
        COUNT(*) OVER (PARTITION BY positive_rate_tier) AS cnt
    FROM dwd_steam_game_aligned
    WHERE data_cohort = 'train_2018'
),
combined AS (
    SELECT * FROM genre_ranked
    UNION ALL
    SELECT * FROM price_ranked
    UNION ALL
    SELECT * FROM pos_ranked
)
SELECT
    dimension_type,
    dimension_value,
    MAX(cnt) AS sample_count,
    ROUND(AVG(CASE WHEN rn * 2 IN (cnt, cnt + 1, cnt + 2) THEN k_ratio END), 2) AS median_k,
    ROUND(AVG(k_ratio), 2) AS mean_k,
    ROUND(AVG(positive_rate) * 100, 1) AS avg_positive_pct,
    ROUND(AVG(price_usd), 2) AS avg_price_usd
FROM combined
GROUP BY dimension_type, dimension_value;

DROP VIEW IF EXISTS vw_finebi_overview_dist;
CREATE VIEW vw_finebi_overview_dist AS
SELECT
    d.appid,
    d.name,
    d.release_year,
    d.genre,
    g.genre_cn_name,
    g.audience_category,
    d.price_usd,
    d.price_tier,
    d.total_reviews,
    d.positive_rate,
    ROUND(d.positive_rate * 100, 1) AS positive_pct,
    d.positive_rate_tier,
    ROUND(d.schinese_ratio * 100, 1) AS schinese_pct,
    d.avg_playtime_at_review_hrs,
    d.true_sales_or_players,
    d.k_ratio,
    d.log_reviews,
    d.log_k,
    d.data_cohort,
    RANK() OVER (PARTITION BY d.data_cohort, d.genre ORDER BY d.true_sales_or_players DESC) AS rank_in_genre,
    ROUND(d.true_sales_or_players * 100.0 / SUM(d.true_sales_or_players) OVER (PARTITION BY d.data_cohort, d.genre), 2) AS share_in_genre_pct
FROM dwd_steam_game_aligned d
LEFT JOIN dim_genre g ON d.genre = g.genre_name;

DROP VIEW IF EXISTS vw_finebi_star_sales_switch;
CREATE VIEW vw_finebi_star_sales_switch AS
SELECT
    f.estimation_id,
    f.appid,
    dg.name AS game_name,
    dg.price_tier,
    dge.genre_name,
    dge.genre_cn_name,
    dge.audience_category,
    dr.release_year,
    dr.era_name,
    dm.method_code,
    dm.method_name,
    dm.signal_source,
    dm.impl_tech,
    f.data_cohort,
    f.total_reviews,
    f.price_usd,
    f.true_sales_or_players,
    f.estimated_sales,
    f.estimated_low_bound,
    f.estimated_high_bound,
    f.estimated_revenue_usd,
    f.implied_k,
    f.observed_k,
    ROUND(f.pct_error * 100, 2) AS pct_error_val,
    ROUND(f.abs_pct_error * 100, 2) AS ape_pct,
    f.within_30pct_flag,
    f.within_50pct_flag,
    RANK() OVER (PARTITION BY f.data_cohort, f.method_code ORDER BY f.estimated_sales DESC) AS sales_rank_in_method
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_release_date dr ON f.date_key = dr.date_key
JOIN dim_estimation_method dm ON f.method_id = dm.method_id;

DROP VIEW IF EXISTS vw_finebi_price_scd2_timeline;
CREATE VIEW vw_finebi_price_scd2_timeline AS
SELECT
    s.snapshot_date,
    s.appid,
    s.name,
    s.total_reviews,
    s.daily_new_reviews,
    s.current_players,
    p.version_no AS price_version_no,
    p.base_price_usd,
    p.discount_percent,
    p.final_price_usd,
    p.event_type AS price_event_type,
    p.effective_start,
    p.effective_end,
    p.is_current,
    s.is_simulated
FROM fact_daily_snapshot s
LEFT JOIN dim_price_scd2 p
    ON s.appid = p.appid
    AND s.snapshot_date >= p.effective_start
    AND s.snapshot_date <= p.effective_end;
