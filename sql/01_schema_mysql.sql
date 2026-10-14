-- 基于 ETL 的 Steam 游戏销量估算与智能分析系统 - 三层架构与星型模型 DDL (MySQL 8.0+)
CREATE DATABASE IF NOT EXISTS steam_sales_dw DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE steam_sales_dw;

DROP TABLE IF EXISTS ods_leak_players_2018;
CREATE TABLE ods_leak_players_2018 (
    appid INT PRIMARY KEY COMMENT 'Steam 应用唯一编号',
    name VARCHAR(255) NOT NULL COMMENT '游戏/应用原始名称',
    players_2018_07_01 BIGINT NOT NULL COMMENT '2018-07-01 成就接口反推真实玩家数',
    snapshot_date DATE NOT NULL COMMENT '快照日期 (2018-07-01)',
    source_url VARCHAR(500) COMMENT '数据来源归档链接',
    loaded_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间'
) ENGINE=InnoDB COMMENT='原始层：2018年Ars Technica成就泄露玩家数快照表';

DROP TABLE IF EXISTS ods_steam_appdetails;
CREATE TABLE ods_steam_appdetails (
    appid INT PRIMARY KEY COMMENT 'Steam 应用编号',
    name VARCHAR(255) COMMENT '商店显示名称',
    app_type VARCHAR(50) COMMENT '应用类型：game / software / demo / dlc',
    is_free TINYINT(1) COMMENT '是否免费：1=免费，0=付费',
    release_year INT COMMENT '发售年份',
    initial_price_usd DECIMAL(10,2) COMMENT '美区原价（美元）',
    final_price_usd DECIMAL(10,2) COMMENT '当前折后价（美元）',
    discount_percent INT COMMENT '当前折扣比例 (0-100)',
    main_genre VARCHAR(100) COMMENT '主游戏类型',
    achievements_count INT COMMENT '成就总数',
    dirty_note VARCHAR(100) COMMENT '原始异常标记',
    raw_json TEXT COMMENT '接口返回原始 JSON 备份'
) ENGINE=InnoDB COMMENT='原始层：Steam 商店详情 appdetails 原始表';

DROP TABLE IF EXISTS ods_steam_reviews;
CREATE TABLE ods_steam_reviews (
    appid INT PRIMARY KEY COMMENT 'Steam 应用编号',
    cutoff_date DATE NOT NULL COMMENT '评测统计截断日期',
    total_reviews INT NOT NULL COMMENT '截断日期前累计评测总数',
    total_positive INT NOT NULL COMMENT '累计好评数',
    total_negative INT NOT NULL COMMENT '累计差评数',
    positive_rate DECIMAL(6,4) COMMENT '好评率 (0~1)',
    schinese_review_ratio DECIMAL(6,4) COMMENT '简体中文评测占比 (0~1)',
    avg_playtime_at_review_hrs DECIMAL(10,2) COMMENT '评测作者写评测时平均游玩小时数',
    steamspy_avg_forever_min INT COMMENT 'SteamSpy 全量玩家平均总时长（分钟）',
    steamcharts_total_player_hours BIGINT COMMENT 'SteamCharts 累计总游玩小时数',
    steamcharts_avg_ccu DECIMAL(12,2) COMMENT 'SteamCharts 历史月均同时在线人数'
) ENGINE=InnoDB COMMENT='原始层：Steam 评测与在线时长原始采集表';

DROP TABLE IF EXISTS ods_recent_sales_news;
CREATE TABLE ods_recent_sales_news (
    audit_id INT AUTO_INCREMENT PRIMARY KEY,
    appid INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    news_date DATE NOT NULL,
    raw_sentence TEXT NOT NULL,
    llm_extracted_number BIGINT,
    llm_extracted_metric VARCHAR(50),
    llm_extracted_platform VARCHAR(100),
    human_audit_status VARCHAR(50),
    human_audit_reason VARCHAR(500),
    source_url VARCHAR(500) NOT NULL
) ENGINE=InnoDB COMMENT='原始层：GetNewsForApp 销量里程碑公告扫描与人工核对表';

DROP TABLE IF EXISTS etl_job_log;
CREATE TABLE etl_job_log (
    job_id INT AUTO_INCREMENT PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    batch_date VARCHAR(20) NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    rows_extracted INT DEFAULT 0,
    rows_loaded INT DEFAULT 0,
    rows_rejected INT DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    duration_ms INT DEFAULT 0,
    message VARCHAR(500)
) ENGINE=InnoDB COMMENT='运维层：ETL 作业运行日志表';

DROP TABLE IF EXISTS etl_checkpoint;
CREATE TABLE etl_checkpoint (
    job_name VARCHAR(100) PRIMARY KEY,
    last_processed_appid INT DEFAULT 0,
    last_cursor VARCHAR(255) DEFAULT '*',
    last_update_time DATETIME,
    status VARCHAR(30) DEFAULT 'READY',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB COMMENT='运维层：ETL 断点续传与增量水位线检查点表';

DROP TABLE IF EXISTS etl_clean_error_log;
CREATE TABLE etl_clean_error_log (
    error_id INT AUTO_INCREMENT PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    appid INT NOT NULL,
    game_name VARCHAR(255),
    rule_code VARCHAR(50) NOT NULL,
    raw_value_desc VARCHAR(255),
    clean_action VARCHAR(50) DEFAULT 'FILTERED_OUT',
    logged_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB COMMENT='运维层：数据清洗异常与过滤审计日志表';

DROP TABLE IF EXISTS dim_genre;
CREATE TABLE dim_genre (
    genre_id INT PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE,
    genre_cn_name VARCHAR(50) NOT NULL,
    audience_category VARCHAR(50) NOT NULL
) ENGINE=InnoDB COMMENT='维度表：游戏类型维';

DROP TABLE IF EXISTS dim_release_date;
CREATE TABLE dim_release_date (
    date_key INT PRIMARY KEY,
    release_year INT NOT NULL UNIQUE,
    era_code VARCHAR(30) NOT NULL,
    era_name VARCHAR(100) NOT NULL,
    has_review_prompt_policy TINYINT(1) NOT NULL,
    literature_k_multiple DECIMAL(8,2) NOT NULL
) ENGINE=InnoDB COMMENT='维度表：发售时间与政策时代维';

DROP TABLE IF EXISTS dim_estimation_method;
CREATE TABLE dim_estimation_method (
    method_id INT PRIMARY KEY,
    method_code VARCHAR(30) NOT NULL UNIQUE,
    method_name VARCHAR(100) NOT NULL,
    signal_source VARCHAR(50) NOT NULL,
    impl_tech VARCHAR(50) NOT NULL,
    description VARCHAR(500) NOT NULL
) ENGINE=InnoDB COMMENT='维度表：销量估算方法维';

DROP TABLE IF EXISTS dim_game;
CREATE TABLE dim_game (
    game_key INT PRIMARY KEY,
    appid INT NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    genre_id INT NOT NULL,
    date_key INT NOT NULL,
    initial_price_usd DECIMAL(10,2) NOT NULL,
    price_tier VARCHAR(50) NOT NULL,
    is_cross_era TINYINT(1) DEFAULT 0,
    data_cohort VARCHAR(30) NOT NULL,
    CONSTRAINT fk_game_genre FOREIGN KEY (genre_id) REFERENCES dim_genre(genre_id),
    CONSTRAINT fk_game_date FOREIGN KEY (date_key) REFERENCES dim_release_date(date_key)
) ENGINE=InnoDB COMMENT='维度表：游戏核心信息维';

DROP TABLE IF EXISTS dim_price_scd2;
CREATE TABLE dim_price_scd2 (
    price_scd_key INT AUTO_INCREMENT PRIMARY KEY,
    appid INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    base_price_usd DECIMAL(10,2) NOT NULL,
    discount_percent INT NOT NULL,
    final_price_usd DECIMAL(10,2) NOT NULL,
    event_type VARCHAR(100),
    effective_start DATE NOT NULL,
    effective_end DATE NOT NULL,
    is_current TINYINT(1) NOT NULL DEFAULT 1,
    version_no INT NOT NULL DEFAULT 1,
    is_simulated TINYINT(1) NOT NULL DEFAULT 1
) ENGINE=InnoDB COMMENT='维度表：游戏价格缓慢变化维 (SCD Type 2 拉链表)';

DROP TABLE IF EXISTS dwd_steam_game_aligned;
CREATE TABLE dwd_steam_game_aligned (
    appid INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    release_year INT NOT NULL,
    genre VARCHAR(50) NOT NULL,
    price_usd DECIMAL(10,2) NOT NULL,
    price_tier VARCHAR(50) NOT NULL,
    total_reviews INT NOT NULL,
    reviews_pre_2019 INT NOT NULL,
    reviews_post_2019 INT NOT NULL,
    positive_rate DECIMAL(6,4) NOT NULL,
    positive_rate_tier VARCHAR(50) NOT NULL,
    schinese_ratio DECIMAL(6,4) NOT NULL,
    avg_playtime_at_review_hrs DECIMAL(10,2) NOT NULL,
    steamspy_avg_forever_min INT NOT NULL,
    steamcharts_total_player_hours BIGINT NOT NULL,
    true_sales_or_players BIGINT NOT NULL,
    k_ratio DECIMAL(10,4) NOT NULL,
    log_k DECIMAL(10,4) NOT NULL,
    log_reviews DECIMAL(10,4) NOT NULL,
    log_price DECIMAL(10,4) NOT NULL,
    data_cohort VARCHAR(30) NOT NULL,
    snapshot_or_report_date DATE NOT NULL,
    source_url VARCHAR(500),
    steam_purchase_ratio DECIMAL(8,4) DEFAULT 0.8500,
    max_discount_pct INT DEFAULT 50,
    achievement_first_rate DECIMAL(8,4) DEFAULT 0.8200,
    launch_30d_review_share DECIMAL(8,4) DEFAULT 0.4500,
    peak_ccu INT DEFAULT 1000,
    has_preorder TINYINT(1) DEFAULT 0,
    ccu_tail_ratio DECIMAL(8,4) DEFAULT 1.5000,
    steam_followers INT DEFAULT 5000,
    top_seller_weeks INT DEFAULT 0,
    audience_cluster_idx DECIMAL(8,4) DEFAULT 0.0000
) ENGINE=InnoDB COMMENT='明细层：清洗对齐后的买断制游戏分析宽表（含H1~H6全维度特征）';

DROP TABLE IF EXISTS fact_sales_estimation;
CREATE TABLE fact_sales_estimation (
    estimation_id INT AUTO_INCREMENT PRIMARY KEY,
    appid INT NOT NULL,
    game_key INT NOT NULL,
    genre_id INT NOT NULL,
    date_key INT NOT NULL,
    method_id INT NOT NULL,
    method_code VARCHAR(30) NOT NULL,
    data_cohort VARCHAR(30) NOT NULL,
    total_reviews INT NOT NULL,
    price_usd DECIMAL(10,2) NOT NULL,
    true_sales_or_players BIGINT NOT NULL,
    estimated_sales BIGINT NOT NULL,
    estimated_low_bound BIGINT NOT NULL,
    estimated_high_bound BIGINT NOT NULL,
    estimated_revenue_usd DECIMAL(16,2) NOT NULL,
    implied_k DECIMAL(10,2) NOT NULL,
    observed_k DECIMAL(10,2) NOT NULL,
    pct_error DECIMAL(10,4) NOT NULL,
    abs_pct_error DECIMAL(10,4) NOT NULL,
    within_30pct_flag TINYINT(1) NOT NULL,
    within_50pct_flag TINYINT(1) NOT NULL,
    CONSTRAINT fk_fact_game FOREIGN KEY (game_key) REFERENCES dim_game(game_key),
    CONSTRAINT fk_fact_genre FOREIGN KEY (genre_id) REFERENCES dim_genre(genre_id),
    CONSTRAINT fk_fact_date FOREIGN KEY (date_key) REFERENCES dim_release_date(date_key),
    CONSTRAINT fk_fact_method FOREIGN KEY (method_id) REFERENCES dim_estimation_method(method_id)
) ENGINE=InnoDB COMMENT='事实表：星型模型多方法销量估算事实表';

DROP TABLE IF EXISTS fact_daily_snapshot;
CREATE TABLE fact_daily_snapshot (
    snapshot_id INT AUTO_INCREMENT PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    appid INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    total_reviews INT NOT NULL,
    daily_new_reviews INT NOT NULL,
    current_players INT NOT NULL,
    discount_percent INT NOT NULL,
    final_price_usd DECIMAL(10,2) NOT NULL,
    last_update_time DATETIME NOT NULL,
    is_simulated TINYINT(1) NOT NULL DEFAULT 1
) ENGINE=InnoDB COMMENT='事实表：每日评测与在线人数周期快照事实表';

DROP TABLE IF EXISTS ads_model_eval_summary;
CREATE TABLE ads_model_eval_summary (
    eval_id INT AUTO_INCREMENT PRIMARY KEY,
    eval_cohort VARCHAR(50) NOT NULL,
    method_code VARCHAR(30) NOT NULL,
    method_name VARCHAR(100) NOT NULL,
    sample_size INT NOT NULL,
    mape_pct DECIMAL(8,2),
    mdape_pct DECIMAL(8,2),
    r2_log DECIMAL(8,4),
    hit_rate_10pct DECIMAL(8,2),
    hit_rate_30pct DECIMAL(8,2) NOT NULL,
    hit_rate_50pct DECIMAL(8,2),
    note VARCHAR(255)
) ENGINE=InnoDB COMMENT='结果层：多模型与商业基准准确率对比汇总表';

DROP TABLE IF EXISTS ads_feature_importance;
CREATE TABLE ads_feature_importance (
    feat_id INT AUTO_INCREMENT PRIMARY KEY,
    feature_code VARCHAR(50) NOT NULL,
    feature_cn_name VARCHAR(100) NOT NULL,
    m2_ols_coef DECIMAL(10,4),
    m2_std_err DECIMAL(10,4),
    m2_t_stat DECIMAL(10,4),
    m2_p_value DECIMAL(10,6),
    m4_rf_importance_pct DECIMAL(8,2),
    m4_gbdt_importance_pct DECIMAL(8,2),
    direction_desc VARCHAR(255)
) ENGINE=InnoDB COMMENT='结果层：M2回归系数与M4树模型特征重要性汇总表';

DROP TABLE IF EXISTS ads_hypothesis_ablation;
CREATE TABLE ads_hypothesis_ablation (
    step_code VARCHAR(40) PRIMARY KEY,
    step_name VARCHAR(120) NOT NULL,
    mape_pct DECIMAL(8,2) NOT NULL,
    mdape_pct DECIMAL(8,2) NOT NULL,
    r2_log DECIMAL(8,4) NOT NULL,
    hit_rate_10pct DECIMAL(8,2) NOT NULL,
    hit_rate_30pct DECIMAL(8,2) NOT NULL,
    hit_rate_50pct DECIMAL(8,2) NOT NULL,
    mechanism_desc VARCHAR(255)
) ENGINE=InnoDB COMMENT='结果层：6大科学假设(H1~H6)逐步消融实验统计汇总表';
