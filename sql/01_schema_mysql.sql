-- =============================================================================
-- 基于 ETL 的 Steam 游戏销量估算系统：三层数据仓库 + 星型模型 DDL（MySQL 8.0）
-- 与 src/etl_pipeline.py 里的 SQLite 表结构一一对应。导入顺序：
--   01_schema_mysql.sql -> 04_full_data_dump_mysql.sql -> 02_views_and_models_mysql.sql -> 03_index_explain_mysql.sql
-- =============================================================================
CREATE DATABASE IF NOT EXISTS steam_sales_dw DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE steam_sales_dw;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS fact_sales_estimation, fact_snapshot, dwd_game_sample, dim_game, dim_price_scd2,
    dim_estimation_method, dim_release_date, dim_genre, ads_review_monthly_index, ads_model_eval_summary,
    ads_feature_importance, ads_policy_calibration, etl_job_log, etl_checkpoint, etl_clean_error_log,
    ods_ars_players_2018, ods_steam_appdetails, ods_steam_reviews_at_cutoff, ods_steamcharts_monthly,
    ods_recent_sales_truth, ods_price_history_wayback, ods_review_monthly_policy_scan,
    ods_official_sales_truth_v2, ods_review_language_counts, ods_itad_price_history, ods_itad_bundles,
    dim_signal, fact_signal_estimate,
    stg_kettle_train_clean, dim_price_scd2_kettle;
SET FOREIGN_KEY_CHECKS = 1;

-- ----------------------------------------------------------------------------- 原始层
CREATE TABLE ods_ars_players_2018 (
    row_no INT PRIMARY KEY COMMENT '原始 CSV 行号（原文件有 6 组重复行，原样保留）',
    appid INT NOT NULL,
    title VARCHAR(255) COMMENT '原文件游戏名（编码混乱，仅供参考）',
    players BIGINT NOT NULL COMMENT '2018-07-01 成就接口反推的玩家数',
    source_url VARCHAR(500),
    KEY idx_ods_ars_appid (appid)
) ENGINE=InnoDB COMMENT='原始层：Ars Technica 2018-07-01 泄露玩家数';

CREATE TABLE ods_steam_appdetails (
    appid INT PRIMARY KEY,
    success TINYINT COMMENT '接口是否返回成功（0 = 已下架或锁区）',
    name VARCHAR(255), app_type VARCHAR(50), is_free TINYINT,
    release_date_text VARCHAR(100), release_date DATE,
    currency VARCHAR(10), initial_price_usd DECIMAL(10,2), final_price_usd DECIMAL(10,2), discount_percent INT,
    genres VARCHAR(500), categories TEXT, developers VARCHAR(500), publishers VARCHAR(500),
    metacritic INT, achievements_total INT, supports_schinese TINYINT, fetched_at DATETIME
) ENGINE=InnoDB COMMENT='原始层：商店详情 appdetails（cc=us）';

CREATE TABLE ods_steam_reviews_at_cutoff (
    appid INT NOT NULL, cohort VARCHAR(20) NOT NULL, cutoff_date DATE NOT NULL, cutoff_ts BIGINT,
    total_reviews INT, total_positive INT, total_negative INT,
    schinese_reviews INT COMMENT '简体中文评测数',
    steam_purchase_reviews INT COMMENT '在 Steam 内购买的玩家写的评测数',
    reviews_before_policy INT COMMENT '评测提示政策上线前的评测数（跨时代样本才有）',
    sample_n INT, sample_median_playtime_at_review_hrs DECIMAL(12,3), sample_mean_playtime_at_review_hrs DECIMAL(12,3),
    first_review_date DATE COMMENT '第一条评测日期（只对商店发售日晚于截止日的游戏补查）',
    fetched_at DATETIME,
    PRIMARY KEY (appid, cohort, cutoff_date)
) ENGINE=InnoDB COMMENT='原始层：截至截止日的评测汇总（GetAppReviews, input_json 传参）';

CREATE TABLE ods_steamcharts_monthly (
    appid INT NOT NULL, month CHAR(7) NOT NULL, avg_players DECIMAL(14,2), peak_players INT,
    PRIMARY KEY (appid, month)
) ENGINE=InnoDB COMMENT='原始层：SteamCharts 月度平均/峰值在线';

CREATE TABLE ods_recent_sales_truth (
    truth_id INT PRIMARY KEY, appid INT NOT NULL, game VARCHAR(255), report_date DATE NOT NULL,
    reported_sales BIGINT NOT NULL, reported_sales_text VARCHAR(20), source_url VARCHAR(1000),
    gamalytic_estimate BIGINT COMMENT 'Gamalytic 对同一游戏的估算（对照用）', benchmark_source VARCHAR(500)
) ENGINE=InnoDB COMMENT='原始层：近年公开销量真值（来自 Gamalytic 测试表，每条附原始出处）';

CREATE TABLE ods_price_history_wayback (
    appid INT NOT NULL, name VARCHAR(255), observed_date DATE NOT NULL, wayback_timestamp CHAR(14) NOT NULL,
    base_price_usd DECIMAL(10,2), final_price_usd DECIMAL(10,2), discount_percent INT, archive_url VARCHAR(500),
    PRIMARY KEY (appid, wayback_timestamp)
) ENGINE=InnoDB COMMENT='原始层：互联网档案馆商店页存档中的历史价格';

CREATE TABLE ods_review_monthly_policy_scan (
    appid INT NOT NULL, month CHAR(7) NOT NULL, new_reviews INT,
    PRIMARY KEY (appid, month)
) ENGINE=InnoDB COMMENT='原始层：40 款老游戏每月新增评测数（判定评测提示上线月份）';

-- 2026-10-20 新增：第二批真值与补充特征的原始数据
CREATE TABLE ods_official_sales_truth_v2 (
    truth_v2_id INT PRIMARY KEY, appid INT NOT NULL, game VARCHAR(255), report_date DATE NOT NULL,
    as_of_date DATE NOT NULL COMMENT '数字对应的时间点（"首日/首周"已换算成 发售日 + 天数）', as_of_rule VARCHAR(30),
    reported_number DOUBLE NOT NULL, number_text VARCHAR(100), steam_truth DOUBLE COMMENT 'Steam 口径的真值（Q 层 = 总数 × Steam 占比）',
    tier VARCHAR(30) NOT NULL COMMENT 'S1 原句写 Steam / S2 标了 Steam 但原句没说 / Q 拆分 / P PC 全商店 / X 排除',
    platform_scope VARCHAR(30), is_lower_bound VARCHAR(10), source_type VARCHAR(30), source_url VARCHAR(1000),
    collect_round INT, quote_found_on_page VARCHAR(20)
) ENGINE=InnoDB COMMENT='原始层：开发商/发行商公布的销量（两轮 AI 收集 + 规则分层，见 src/truth_v2_merge.py）';

CREATE TABLE ods_review_language_counts (
    appid INT NOT NULL, cohort VARCHAR(20) NOT NULL, cutoff_date DATE NOT NULL, language VARCHAR(20) NOT NULL,
    total_reviews INT, PRIMARY KEY (appid, cohort, cutoff_date, language)
) ENGINE=InnoDB COMMENT='原始层：截止日前各语言的评测数';

CREATE TABLE ods_itad_price_history (
    appid INT NOT NULL, country CHAR(2) NOT NULL, ts VARCHAR(30) NOT NULL, currency CHAR(3),
    price DECIMAL(12,2), regular DECIMAL(12,2), cut INT,
    KEY idx_itad_appid_country (appid, country)
) ENGINE=InnoDB COMMENT='原始层：IsThereAnyDeal 的 Steam 价格历史（美/中/土/巴四个区，只保留到截止日）';

CREATE TABLE ods_itad_bundles (
    appid INT NOT NULL, bundle_id INT NOT NULL, title VARCHAR(500), page VARCHAR(100), publish VARCHAR(30),
    PRIMARY KEY (appid, bundle_id)
) ENGINE=InnoDB COMMENT='原始层：游戏进过的慈善包（IsThereAnyDeal）';

-- ----------------------------------------------------------------------------- 运维表
CREATE TABLE etl_job_log (
    job_id INT AUTO_INCREMENT PRIMARY KEY, job_name VARCHAR(100) NOT NULL,
    start_time DATETIME NOT NULL, end_time DATETIME NOT NULL,
    rows_extracted INT DEFAULT 0, rows_loaded INT DEFAULT 0, rows_rejected INT DEFAULT 0,
    status VARCHAR(30) NOT NULL, duration_ms INT DEFAULT 0, message VARCHAR(1000)
) ENGINE=InnoDB COMMENT='ETL 运行日志';

CREATE TABLE etl_checkpoint (
    job_name VARCHAR(100) PRIMARY KEY,
    last_processed_key VARCHAR(50) COMMENT '已提交的最后一个样本键（断点续传）',
    last_update_time DATETIME COMMENT '增量水位线',
    status VARCHAR(30) DEFAULT 'READY', updated_at DATETIME
) ENGINE=InnoDB COMMENT='检查点：断点续传与增量水位线';

CREATE TABLE etl_clean_error_log (
    error_id INT AUTO_INCREMENT PRIMARY KEY, job_name VARCHAR(100) NOT NULL, appid INT NOT NULL,
    cohort VARCHAR(20), game_name VARCHAR(255), rule_code VARCHAR(50) NOT NULL, raw_value_desc VARCHAR(500),
    clean_action VARCHAR(30) NOT NULL COMMENT 'FILTERED_OUT / FLAGGED / DEDUPLICATED', logged_at DATETIME,
    KEY idx_err_rule (rule_code)
) ENGINE=InnoDB COMMENT='清洗审计：被拦截或标记的记录';

-- ----------------------------------------------------------------------------- 维度表
CREATE TABLE dim_genre (
    genre_id INT PRIMARY KEY, genre_name VARCHAR(50) NOT NULL UNIQUE, genre_cn_name VARCHAR(50) NOT NULL
) ENGINE=InnoDB COMMENT='维度：主类型';

CREATE TABLE dim_release_date (
    date_key INT PRIMARY KEY, release_year INT NOT NULL UNIQUE, era_code VARCHAR(30) NOT NULL,
    era_name VARCHAR(50) NOT NULL, has_review_prompt_policy TINYINT NOT NULL, literature_k_multiple DECIMAL(6,2) NOT NULL
) ENGINE=InnoDB COMMENT='维度：发售年份（含评测提示政策分期与文献 k）';

CREATE TABLE dim_estimation_method (
    method_id INT PRIMARY KEY, method_code VARCHAR(30) NOT NULL UNIQUE, method_name VARCHAR(100) NOT NULL,
    signal_source VARCHAR(30) NOT NULL, impl_tech VARCHAR(30) NOT NULL,
    uses_truth_for_fitting TINYINT NOT NULL COMMENT '是否在真值集上拟合参数（是则一律留一法）',
    description VARCHAR(500) NOT NULL
) ENGINE=InnoDB COMMENT='维度：估算方法（FineBI 切片器）';

CREATE TABLE dim_game (
    game_key INT PRIMARY KEY, appid INT NOT NULL UNIQUE, name VARCHAR(255) NOT NULL,
    genre_id INT NOT NULL, date_key INT NOT NULL, release_date DATE,
    price_usd DECIMAL(10,2) NOT NULL, price_tier VARCHAR(20) NOT NULL, is_indie TINYINT NOT NULL,
    CONSTRAINT fk_game_genre FOREIGN KEY (genre_id) REFERENCES dim_genre(genre_id),
    CONSTRAINT fk_game_date FOREIGN KEY (date_key) REFERENCES dim_release_date(date_key)
) ENGINE=InnoDB COMMENT='维度：游戏';

CREATE TABLE dim_price_scd2 (
    price_scd_key INT AUTO_INCREMENT PRIMARY KEY, appid INT NOT NULL, name VARCHAR(255) NOT NULL,
    base_price_usd DECIMAL(10,2) NOT NULL, discount_percent INT NOT NULL, final_price_usd DECIMAL(10,2) NOT NULL,
    effective_start DATE NOT NULL, effective_end DATE NOT NULL COMMENT '当前版本为 9999-12-31',
    is_current TINYINT NOT NULL, version_no INT NOT NULL,
    observation_source VARCHAR(30) NOT NULL COMMENT 'wayback_archive / store_api', source_ref VARCHAR(500)
) ENGINE=InnoDB COMMENT='维度：价格缓慢变化维 SCD Type 2（真实观测）';

-- ----------------------------------------------------------------------------- 明细层
CREATE TABLE dwd_game_sample (
    sample_id INT PRIMARY KEY, appid INT NOT NULL, name VARCHAR(255) NOT NULL,
    cohort VARCHAR(20) NOT NULL COMMENT 'train_2018 / truth / truth_v2',
    era_group VARCHAR(20) NOT NULL COMMENT 'pre_policy / cross_era / post_policy',
    cutoff_date DATE NOT NULL, release_date DATE NOT NULL COMMENT '实际上架日期（三个来源取最早）',
    release_date_source VARCHAR(30) NOT NULL COMMENT 'store / first_review / steamcharts_first_month',
    release_year INT NOT NULL, game_age_years DECIMAL(8,4) NOT NULL,
    genre VARCHAR(50) NOT NULL, is_indie TINYINT NOT NULL, price_usd DECIMAL(10,2) NOT NULL, price_tier VARCHAR(20) NOT NULL,
    total_reviews INT NOT NULL, reviews_before_policy INT NOT NULL, reviews_after_policy INT NOT NULL,
    positive_rate DECIMAL(6,4) NOT NULL, positive_rate_tier VARCHAR(20) NOT NULL,
    schinese_ratio DECIMAL(6,4) NOT NULL, steam_purchase_ratio DECIMAL(6,4) NOT NULL,
    playtime_review_median_hrs DECIMAL(12,3), review_sample_n INT,
    charts_total_hours DECIMAL(18,1), charts_coverage TINYINT NOT NULL,
    truth_value BIGINT NOT NULL, truth_type VARCHAR(30) NOT NULL,
    k_ratio DECIMAL(14,4) NOT NULL, log_k DECIMAL(12,6) NOT NULL, log_reviews DECIMAL(12,6) NOT NULL,
    log_price DECIMAL(12,6) NOT NULL, model_eligible TINYINT NOT NULL, eligibility_note VARCHAR(500),
    source_url VARCHAR(1000),
    -- 2026-10-20 新增的补充特征（全部只用截止日及以前的数据）
    share_english DECIMAL(8,5), share_russian DECIMAL(8,5), share_german DECIMAL(8,5), share_french DECIMAL(8,5),
    share_polish DECIMAL(8,5), share_brazilian DECIMAL(8,5), share_spanish_all DECIMAL(8,5), share_japanese DECIMAL(8,5),
    share_koreana DECIMAL(8,5), share_tchinese DECIMAL(8,5), share_turkish DECIMAL(8,5),
    itad_covered TINYINT, price_at_cutoff_usd DECIMAL(10,2) COMMENT '截止日当时的原价（ITAD）',
    launch_price_usd DECIMAL(10,2), n_discounts INT, max_cut INT, frac_time_discounted DECIMAL(8,5),
    frac_time_cut75 DECIMAL(8,5), n_bundles INT,
    ln_regional_cn DECIMAL(10,5), ln_regional_tr DECIMAL(10,5), ln_regional_br DECIMAL(10,5)
) ENGINE=InnoDB COMMENT='明细层：样本宽表（一行 = 一款游戏在一个截止日）';

-- ----------------------------------------------------------------------------- 结果层
CREATE TABLE fact_sales_estimation (
    estimation_id INT AUTO_INCREMENT PRIMARY KEY, sample_id INT NOT NULL, appid INT NOT NULL,
    game_key INT NOT NULL, genre_id INT NOT NULL, date_key INT NOT NULL, method_id INT NOT NULL,
    method_code VARCHAR(30) NOT NULL, cohort VARCHAR(20) NOT NULL, era_group VARCHAR(20) NOT NULL,
    total_reviews INT NOT NULL, price_usd DECIMAL(10,2) NOT NULL, true_value BIGINT NOT NULL,
    estimated_sales BIGINT NOT NULL, estimated_low_bound BIGINT NOT NULL, estimated_high_bound BIGINT NOT NULL,
    estimated_revenue_usd DECIMAL(16,2) NOT NULL COMMENT '估算销量 × 当前标价（毛收入上限）',
    implied_k DECIMAL(14,3) NOT NULL, observed_k DECIMAL(14,3) NOT NULL,
    pct_error DECIMAL(12,5) NOT NULL, abs_pct_error DECIMAL(12,5) NOT NULL,
    within_30pct_flag TINYINT NOT NULL, within_50pct_flag TINYINT NOT NULL,
    CONSTRAINT fk_fact_game FOREIGN KEY (game_key) REFERENCES dim_game(game_key),
    CONSTRAINT fk_fact_genre FOREIGN KEY (genre_id) REFERENCES dim_genre(genre_id),
    CONSTRAINT fk_fact_date FOREIGN KEY (date_key) REFERENCES dim_release_date(date_key),
    CONSTRAINT fk_fact_method FOREIGN KEY (method_id) REFERENCES dim_estimation_method(method_id)
) ENGINE=InnoDB COMMENT='事实表：样本 × 估算方法';

CREATE TABLE fact_snapshot (
    snapshot_id INT AUTO_INCREMENT PRIMARY KEY, fetched_at DATETIME NOT NULL, round_no INT NOT NULL,
    appid INT NOT NULL, name VARCHAR(255), current_players INT, total_reviews INT, new_reviews_since_prev INT,
    total_positive INT, final_price_usd DECIMAL(10,2), discount_percent INT, loaded_at DATETIME NOT NULL,
    UNIQUE KEY uk_snapshot (appid, fetched_at)
) ENGINE=InnoDB COMMENT='周期快照事实表（每小时；按 fetched_at 水位线增量装载）';

CREATE TABLE ads_review_monthly_index (
    month CHAR(7) PRIMARY KEY, median_index DECIMAL(10,4), q25 DECIMAL(10,4), q75 DECIMAL(10,4),
    n_games INT, is_policy_month TINYINT
) ENGINE=InnoDB COMMENT='结果：老游戏月度评测指数';

CREATE TABLE ads_model_eval_summary (
    eval_id INT AUTO_INCREMENT PRIMARY KEY, eval_cohort VARCHAR(50) NOT NULL, method_code VARCHAR(30) NOT NULL,
    method_name VARCHAR(100) NOT NULL, sample_size INT NOT NULL, mape_pct DECIMAL(10,2), mdape_pct DECIMAL(10,2),
    r2_log DECIMAL(10,4), hit_rate_10pct DECIMAL(6,2), hit_rate_30pct DECIMAL(6,2), hit_rate_50pct DECIMAL(6,2),
    note VARCHAR(500)
) ENGINE=InnoDB COMMENT='结果：各方法准确率';

CREATE TABLE ads_feature_importance (
    feat_id INT AUTO_INCREMENT PRIMARY KEY, feature_code VARCHAR(50) NOT NULL, feature_cn_name VARCHAR(100) NOT NULL,
    m2_ols_coef DECIMAL(14,5), m2_std_err DECIMAL(14,5), m2_t_stat DECIMAL(12,3), m2_p_value DOUBLE,
    m4_rf_importance_pct DECIMAL(8,2), m4_gbdt_importance_pct DECIMAL(8,2)
) ENGINE=InnoDB COMMENT='结果：回归系数与树模型特征重要性';

CREATE TABLE dim_signal (
    signal_code VARCHAR(10) PRIMARY KEY, signal_name VARCHAR(100) NOT NULL,
    signal_kind VARCHAR(20) NOT NULL COMMENT 'estimate = 估算值 / lower_bound = 下限 / upper_bound = 上限',
    quality_feature VARCHAR(200) COMMENT '决定这路信号可信度的质量指标'
) ENGINE=InnoDB COMMENT='维度：参与融合的信号（以后加新信号只需在这里登记）';

CREATE TABLE fact_signal_estimate (
    sample_id INT NOT NULL, signal_code VARCHAR(10) NOT NULL, available TINYINT NOT NULL,
    ln_estimate DOUBLE, sigma DOUBLE COMMENT '这路信号在这款游戏上的预计误差（ln 尺度）',
    weight_share DOUBLE NOT NULL COMMENT '这路信号在融合里的权重占比',
    PRIMARY KEY (sample_id, signal_code),
    CONSTRAINT fk_sig_signal FOREIGN KEY (signal_code) REFERENCES dim_signal(signal_code)
) ENGINE=InnoDB COMMENT='结果层：每款真值游戏 × 每路信号的估算与权重（融合怎么做的决定）';

CREATE TABLE ads_policy_calibration (
    calib_id INT AUTO_INCREMENT PRIMARY KEY, item VARCHAR(60) NOT NULL, value DOUBLE, note VARCHAR(500)
) ENGINE=InnoDB COMMENT='结果：政策校准系数、信号 B 参数、融合权重、残差检验';

-- ----------------------------------------------------------------------------- Kettle 转换的输出表
-- 和 Python 写的表分开，跑完可以直接对比两边结果是否一致
CREATE TABLE stg_kettle_train_clean (
    appid INT, name VARCHAR(255), release_date VARCHAR(20), price_usd DECIMAL(10,2), players BIGINT,
    total_reviews INT, positive_rate DOUBLE, k_ratio DOUBLE
) ENGINE=InnoDB COMMENT='Kettle 清洗转换输出（对照 dwd_game_sample 的 train_2018 部分）';

CREATE TABLE dim_price_scd2_kettle (
    price_scd_key INT AUTO_INCREMENT PRIMARY KEY, version_no INT, effective_start DATETIME, effective_end DATETIME,
    appid INT, name VARCHAR(255), base_price_usd DOUBLE, final_price_usd DOUBLE, discount_percent INT,
    source_ref VARCHAR(500), is_current CHAR(1),
    KEY idx_kettle_scd2 (appid, effective_start)
) ENGINE=InnoDB COMMENT='Kettle 维度查询/更新步骤维护的价格拉链（对照 dim_price_scd2）';
