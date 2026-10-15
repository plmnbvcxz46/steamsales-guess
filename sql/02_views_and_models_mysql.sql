-- =============================================================================
-- SQL 实现 M0/M1、窗口函数求 k 的中位数、FineBI 只读视图
-- 同一份脚本在 SQLite 3.25+（Python ETL 用）和 MySQL 8.0（FineBI 连接）上都能执行：
--   只用两边都支持的语法：CTE、ROW_NUMBER/RANK/COUNT/SUM OVER、CASE WHEN、ROUND、SUBSTR
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. M0 / M1：纯 SQL 估算
--    M0：销量 = 评测数 × 30
--    M1：按发售年份取文献 k（<=2016 取 74，2017-2018 取 60，2019 取 51，2020 起取 38）
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_sql_m0_m1_estimation;
CREATE VIEW vw_sql_m0_m1_estimation AS
SELECT
    d.sample_id,
    d.appid,
    d.name,
    d.cohort,
    d.era_group,
    d.release_year,
    d.total_reviews,
    d.truth_value,
    d.k_ratio AS observed_k,
    30.0 AS m0_k,
    ROUND(d.total_reviews * 30.0, 0) AS m0_estimated_sales,
    ROUND(ABS(d.total_reviews * 30.0 - d.truth_value) / d.truth_value, 4) AS m0_ape,
    CASE WHEN ABS(d.total_reviews * 30.0 - d.truth_value) / d.truth_value <= 0.30 THEN 1 ELSE 0 END AS m0_within_30pct,
    CASE
        WHEN d.release_year <= 2016 THEN 74.0
        WHEN d.release_year <= 2018 THEN 60.0
        WHEN d.release_year = 2019 THEN 51.0
        ELSE 38.0
    END AS m1_k,
    ROUND(d.total_reviews * (CASE
        WHEN d.release_year <= 2016 THEN 74.0
        WHEN d.release_year <= 2018 THEN 60.0
        WHEN d.release_year = 2019 THEN 51.0
        ELSE 38.0
    END), 0) AS m1_estimated_sales
FROM dwd_game_sample d;

-- -----------------------------------------------------------------------------
-- 2. 各发售年份的 k 中位数（MySQL 和 SQLite 都没有 MEDIAN()，用窗口函数求）
--    思路：组内按 k 排序编号 rn，组内总数 cnt；
--          cnt 为奇数时取 rn = (cnt+1)/2 那一条，为偶数时取 rn = cnt/2 和 cnt/2+1 两条求平均。
--          两种情况合起来正好是 rn*2 IN (cnt, cnt+1, cnt+2)。
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_k_median_by_year;
CREATE VIEW vw_finebi_k_median_by_year AS
WITH ranked AS (
    SELECT
        cohort,
        release_year,
        k_ratio,
        ROW_NUMBER() OVER (PARTITION BY cohort, release_year ORDER BY k_ratio) AS rn,
        COUNT(*) OVER (PARTITION BY cohort, release_year) AS cnt
    FROM dwd_game_sample
    WHERE model_eligible = 1
)
SELECT
    r.cohort,
    r.release_year,
    MAX(r.cnt) AS game_count,
    ROUND(AVG(CASE WHEN r.rn * 2 IN (r.cnt, r.cnt + 1, r.cnt + 2) THEN r.k_ratio END), 2) AS median_k,
    ROUND(AVG(r.k_ratio), 2) AS mean_k,
    MAX(y.era_name) AS era_name,
    MAX(y.literature_k_multiple) AS literature_k
FROM ranked r
LEFT JOIN dim_release_date y ON y.release_year = r.release_year
GROUP BY r.cohort, r.release_year;

-- -----------------------------------------------------------------------------
-- 3. 2018 训练集里 k 按各因素分组的中位数（价格档、好评率档、主类型、是否独立）
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_k_median_by_factor;
CREATE VIEW vw_finebi_k_median_by_factor AS
WITH base AS (
    SELECT price_tier, positive_rate_tier, genre,
           CASE WHEN is_indie = 1 THEN '独立' ELSE '非独立' END AS indie_flag,
           k_ratio
    FROM dwd_game_sample
    WHERE cohort = 'train_2018' AND model_eligible = 1
),
long_form AS (
    SELECT '价格档' AS factor, price_tier AS factor_value, k_ratio FROM base
    UNION ALL SELECT '好评率档', positive_rate_tier, k_ratio FROM base
    UNION ALL SELECT '主类型', genre, k_ratio FROM base
    UNION ALL SELECT '是否独立', indie_flag, k_ratio FROM base
),
ranked AS (
    SELECT factor, factor_value, k_ratio,
           ROW_NUMBER() OVER (PARTITION BY factor, factor_value ORDER BY k_ratio) AS rn,
           COUNT(*) OVER (PARTITION BY factor, factor_value) AS cnt
    FROM long_form
)
SELECT
    factor,
    factor_value,
    MAX(cnt) AS game_count,
    ROUND(AVG(CASE WHEN rn * 2 IN (cnt, cnt + 1, cnt + 2) THEN k_ratio END), 2) AS median_k
FROM ranked
GROUP BY factor, factor_value;

-- -----------------------------------------------------------------------------
-- 4. 样本明细（数据概览大屏）：带类型内排名与份额两个窗口函数
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_overview_dist;
CREATE VIEW vw_finebi_overview_dist AS
SELECT
    d.sample_id,
    d.appid,
    d.name,
    d.cohort,
    d.era_group,
    d.release_year,
    d.genre,
    g.genre_cn_name,
    d.price_usd,
    d.price_tier,
    d.total_reviews,
    ROUND(d.positive_rate * 100, 1) AS positive_pct,
    d.positive_rate_tier,
    ROUND(d.schinese_ratio * 100, 1) AS schinese_pct,
    ROUND(d.steam_purchase_ratio * 100, 1) AS steam_purchase_pct,
    d.playtime_review_median_hrs,
    d.truth_value,
    d.truth_type,
    d.k_ratio,
    d.model_eligible,
    RANK() OVER (PARTITION BY d.cohort, d.genre ORDER BY d.truth_value DESC) AS rank_in_genre,
    ROUND(d.truth_value * 100.0 / SUM(d.truth_value) OVER (PARTITION BY d.cohort, d.genre), 2) AS share_in_genre_pct
FROM dwd_game_sample d
LEFT JOIN dim_genre g ON g.genre_name = d.genre;

-- -----------------------------------------------------------------------------
-- 5. 星型模型大屏视图：事实表 + 四个维度。FineBI 用 method_name 做切片器，整屏随估算方法联动
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_star_sales_switch;
CREATE VIEW vw_finebi_star_sales_switch AS
SELECT
    f.estimation_id,
    f.sample_id,
    f.appid,
    dg.name AS game_name,
    dg.price_tier,
    dge.genre_cn_name,
    dr.release_year,
    dr.era_name,
    dm.method_code,
    dm.method_name,
    dm.signal_source,
    f.cohort,
    f.era_group,
    f.total_reviews,
    f.price_usd,
    f.true_value,
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
    RANK() OVER (PARTITION BY f.cohort, f.method_code ORDER BY f.estimated_sales DESC) AS sales_rank_in_method
FROM fact_sales_estimation f
JOIN dim_game dg ON dg.game_key = f.game_key
JOIN dim_genre dge ON dge.genre_id = f.genre_id
JOIN dim_release_date dr ON dr.date_key = f.date_key
JOIN dim_estimation_method dm ON dm.method_id = f.method_id;

-- -----------------------------------------------------------------------------
-- 6. 价格拉链 + 快照：每条快照关联"快照当天生效"的那个价格版本
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_price_scd2_timeline;
CREATE VIEW vw_finebi_price_scd2_timeline AS
SELECT
    p.appid,
    p.name,
    p.version_no,
    p.base_price_usd,
    p.discount_percent,
    p.final_price_usd,
    p.effective_start,
    p.effective_end,
    p.is_current,
    p.observation_source,
    s.fetched_at AS snapshot_time,
    s.current_players,
    s.total_reviews,
    s.new_reviews_since_prev
FROM dim_price_scd2 p
LEFT JOIN fact_snapshot s
    ON s.appid = p.appid
    AND SUBSTR(s.fetched_at, 1, 10) >= p.effective_start
    AND SUBSTR(s.fetched_at, 1, 10) < p.effective_end;

-- -----------------------------------------------------------------------------
-- 7. 评测月度指数（"游玩后提示写评测"上线前后对比）
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_review_policy_index;
CREATE VIEW vw_finebi_review_policy_index AS
SELECT month, median_index, q25, q75, n_games, is_policy_month
FROM ads_review_monthly_index;

-- -----------------------------------------------------------------------------
-- 8. 各方法准确率（含 Gamalytic 在同一批游戏上的估算作对照）
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_finebi_method_accuracy;
CREATE VIEW vw_finebi_method_accuracy AS
SELECT eval_cohort, method_code, method_name, sample_size, mape_pct, mdape_pct,
       hit_rate_10pct, hit_rate_30pct, hit_rate_50pct, note
FROM ads_model_eval_summary;
