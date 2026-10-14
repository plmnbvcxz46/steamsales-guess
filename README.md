# 基于 ETL 的 Steam 游戏销量估算与智能分析系统 (steamsales-guess)

> **课程**：高级数据库技术与应用 · 课程大作业  
> **仓库地址**：`https://github.com/plmnbvcxz46/steamsales-guess`  
> **核心问题**：Steam 官方从不公开单款游戏的具体销量，如何利用公开可采集的多源异构数据（评测数、商店价格、好评率、历史在线人数与人均游玩时长），构建一套工程化的 ETL 数据仓库与多信号融合估算系统，回答“某款游戏在 Steam 上到底卖了多少份”？

---

## 一、项目亮点与核心特色

1. **规范的三层数据仓库与星型模型设计**：
   - 采用 **原始层 (ODS) → 明细与维度层 (DWD/DIM) → 结果与事实层 (FACT/ADS)** 三层架构，共设计 13 张核心表与 6 个只读分析视图。
   - 星型模型专门引入 **估算方法维度表 (`dim_estimation_method`)**，事实表 `fact_sales_estimation` 按“每款游戏 × 每种估算方法一行”存储（含 80% 预测区间上下限），支持在 FineBI 大屏中通过单个切片器实现整屏图表联动切换。
2. **完整的工业级 ETL 机制（Python + Kettle 双实现）**：
   - **数据清洗与异常审计**：自动识别并拦截免费游戏 (`FREE_GAME`)、非游戏工具软件 (`NON_GAME_SOFTWARE`)、试玩版 (`DEMO_VERSION`)、负价格 (`INVALID_PRICE`) 与零评测记录 (`ZERO_REVIEWS`)，所有拦截明细写入 `etl_clean_error_log`。
   - **断点续传 (`etl_checkpoint`)** 与 **运行日志 (`etl_job_log`)**：记录批处理最后的 `last_processed_appid` 与增量水位线时间戳 `last_update_time`，支持异常中断后无缝续传。
   - **缓慢变化维 SCD Type 2 (`dim_price_scd2`)**：针对游戏打折促销与永久调价事件，自动闭合旧价格版本 (`effective_end`, `is_current=0`) 并插入新版本 (`version_no+1`)，完整保留历史价格轨迹。
   - **Kettle (PDI) 转换与作业**：提供标准 `.ktr` 与 `.kjb` 文件（`clean_steam_games.ktr`、`scd2_price_history.ktr`、`steam_etl_main.kjb`）。
3. **深入的 SQL 技术落地**：
   - 使用 SQL `CASE WHEN` 视图直接实现 **M0（固定倍数 30）** 与 **M1（按年份分段文献经验系数）**。
   - 针对 MySQL/SQLite 无内置中位数函数的问题，使用 **SQL 窗口函数 (`ROW_NUMBER() OVER` + `COUNT(*) OVER`)** 精确求解各年份、各类型、各价格档的实测 $k$ 中位数。
   - 在核心关联与过滤字段建立复合索引 `idx_fact_cohort_method`，通过 `EXPLAIN` 验证将全表扫描优化为索引查找，多表聚合查询提速 **1.72 倍**。
4. **从单一评测信号到双信号融合的完整建模链路**：
   - 实证揭示 **2019 年 10 月 Steam 上线“游玩后弹窗提示写评测”引发的 $k$ 值断崖减半**，以及控制价格混杂效应后 **好评率与 $k$ 的“倒 U 形”二次曲线关系**。
   - 对比 M0、M1、M2（多元对数线性回归）、M3（分位数回归区间估计）、M4（随机森林与梯度提升树 GBDT），证明单一评测信号在近年真值集上的 $\pm 30\%$ 命中率上限约为 **62.50%**。
   - 引入不受评测政策影响的 **信号 B（SteamCharts 累计游玩总小时数 ÷ 校准后人均游玩时长）** 并与信号 A 进行对数空间加权融合（`Fusion_AB`），使近年真值集的 $\pm 30\%$ 命中率跃升至 **81.25%**（MAPE 降至 **17.21%**），跨时代检验集命中率达 **87.50%**，全面对标商业分析机构 Gamalytic (76.92%)。

---

## 二、项目目录结构

```text
steamsales-guess/
├── README.md                          # 项目说明、运行步骤、数据来源与合规声明
├── 开发日志.md                         # 全程真实维护的开发过程、踩坑排障与方案取舍记录
├── FineBI_制作步骤.md                  # FineBI 可视化大屏的视图绑定、字段拖拽与联动配置指南
├── 课程报告.md                         # 完整 7 章课程大作业报告（含 Mermaid 架构图/流向图/ER图）
├── 最终总结.md                         # 交付清单总结与后续手动操作（Git 推送、FineBI 截图）说明
├── data/
│   ├── steam_sales_dw.db              # 已跑通全流程的 SQLite 数据仓库文件（含全部表与6个视图）
│   ├── raw/                           # 原始层数据
│   │   ├── games_achievements_players_2018-07-01.csv   # 2018-07-01 Ars Technica 成就泄露快照
│   │   ├── steam_appdetails_2018_aligned.json          # 商店元数据原始 JSON
│   │   ├── steam_reviews_2018_aligned.csv              # 截至 2018-07-01 对齐的评测与时长统计
│   │   ├── recent_sales_truth_2020_2025.csv            # 2020+ 近年公开销量真值集与跨时代检验集
│   │   └── recent_news_extraction_audit.csv            # 官方新闻大模型结构化抽取与人工核对清单
│   ├── simulated/                     # 单独存放、明确标注的 ETL 机制演示模拟数据（不参与建模）
│   │   ├── sim_price_changes.csv                       # 用于演示 SCD Type 2 调价拉链的模拟事件
│   │   └── sim_daily_increments.csv                    # 用于演示增量水位线抽取的 14 天快照
│   └── cleaned/                       # 导出的清洗结果、FineBI 只读视图 CSV 与 EXPLAIN 对比报告
├── src/
│   ├── crawler.py                     # Steam 多源采集模块（支持限速、429 指数退避、本地缓存与断点续传）
│   ├── prototype_explore.py           # 原型验证脚本（对齐 2018 数据，算实测 k，生成图1与图2）
│   ├── etl_pipeline.py                # 核心 Python ETL 管道（ODS->DWD/DIM、断点续传、SCD2、增量抽取）
│   ├── models.py                      # 多信号建模脚本（M0~M4、年代校准、信号B、双信号融合、跨时代检验）
│   └── visualize_all.py               # 统计分析与架构图绘制脚本（生成 fig03 ~ fig10 高清中文图）
├── sql/
│   ├── 01_schema_mysql.sql            # MySQL 8.0 三层架构与星型模型建表 DDL 脚本
│   ├── 02_views_and_models_mysql.sql  # SQL 实现 M0/M1、窗口函数求 k 中位数及 6 个 FineBI 视图
│   ├── 03_index_explain_mysql.sql     # 复合索引创建与 EXPLAIN 性能对比脚本
│   └── 04_full_data_dump_mysql.sql    # 全量结果数据 INSERT 转储脚本（可一键导入本地 MySQL）
├── kettle/
│   ├── clean_steam_games.ktr          # Kettle 清洗转换：过滤免费/异常记录并计算实测 k
│   ├── scd2_price_history.ktr         # Kettle SCD Type 2 转换：使用维度查询/更新维护价格拉链表
│   └── steam_etl_main.kjb             # Kettle 主控作业：串联清洗转换、SCD2 转换与 SQL 视图刷新
├── figures/                           # Python 生成的全套 10 张 180 DPI 高清中文统计与架构图
└── tests/
    └── test_etl_and_models.py         # 单元测试与集成测试套件（验证清洗、SCD2、窗口中位数与模型指标）
```

---

## 三、快速运行步骤

### 1. 运行 Python 原型探索、ETL 管道、多信号建模与自动化测试
本项目仅依赖标准科学计算库（`numpy`、`pandas`、`scipy`、`matplotlib`、`requests`）与 Python 内置 `sqlite3`，无需额外配置复杂深度学习环境：

```bash
# 1. 运行原型阶段探索（输出 prototype_2018_k_summary.csv 及 fig01、fig02）
python3 src/prototype_explore.py

# 2. 运行完整 ETL 管道（构建三层表结构、断点续传清洗、SCD2 价格拉链、增量快照与 SQL 视图）
python3 src/etl_pipeline.py

# 3. 运行 M0~M4 + 信号 B + 双信号融合建模（写回事实表并执行 EXPLAIN 索引对比）
python3 src/models.py

# 4. 生成全套 10 张中文统计图表（输出至 figures/ 目录）
python3 src/visualize_all.py

# 5. 运行自动化单元与集成测试套件
python3 tests/test_etl_and_models.py
```

### 2. 导入本地 MySQL 8.0 数据库（供 FineBI 与 Kettle 连接）
如果本机安装了 MySQL 8.0，只需依次运行 `sql/` 目录下的四个脚本，即可完整复现所有表、数据、索引和视图：

```bash
mysql -u root -p < sql/01_schema_mysql.sql
mysql -u root -p steam_sales_dw < sql/04_full_data_dump_mysql.sql
mysql -u root -p steam_sales_dw < sql/02_views_and_models_mysql.sql
mysql -u root -p steam_sales_dw < sql/03_index_explain_mysql.sql
```

### 3. 运行 Kettle (Pentaho Data Integration) 转换与作业
1. 下载并解压 Pentaho Data Integration (Kettle 9.2+)，确保本机已安装 Java 8/11 及 MySQL JDBC 驱动（`mysql-connector-j.jar` 放入 `data-integration/lib/`）。
2. 启动图形界面 Spoon（Windows 运行 `Spoon.bat`，Linux/macOS 运行 `spoon.sh`），直接打开 `kettle/steam_etl_main.kjb` 或 `kettle/clean_steam_games.ktr`、`kettle/scd2_price_history.ktr`，修改数据库连接 `Steam_MySQL_DW` 的本地密码后点击运行并截图。
3. 也可通过命令行无头模式运行转换：
   ```bash
   pan.sh -file=kettle/clean_steam_games.ktr -level=Basic
   kitchen.sh -file=kettle/steam_etl_main.kjb -level=Basic
   ```

---

## 四、数据来源、合规与许可声明

1. **2018 年真实玩家数基准**：源自 2018-07-01 Valve 成就 API 精度漏洞期间由 Tyler Glaiel 与 Ars Technica 整理的公开快照数据（参考仓库 `woctezuma/steam-reviews-to-sales` 与 Feed4Gamers/AltChar 归档数据）。
2. **近年公开销量真值与评测统计**：来自 Gamalytic 公开基准数据集（`https://gamalytic.com/blog/how-to-accurately-estimate-steam-sales`）及通过 Steam 官方接口 `IUserReviewsService/GetAppReviews/v1/`、`ISteamNews/GetNewsForApp/v2/`、`store.steampowered.com/api/appdetails` 获取的官方里程碑公告。
3. **在线人数与人均游玩时长**：来自 SteamCharts 与 SteamSpy 公开 API（`https://steamspy.com/api.php`），严格遵守每秒最多 1 次的请求频率限制。
4. **合规红线声明**：
   - 本项目**严禁且未抓取任何 SteamDB 数据**（遵守 SteamDB FAQ 条款）；
   - 所有真实真值均附有明确出处链接，绝不凭记忆编造销量；
   - 仅用于演示 SCD Type 2 价格拉链与每日增量快照的模拟数据单独存放在 `data/simulated/` 目录下，并在数据库字段中标注 `is_simulated = 1`，不参与任何模型的训练与评测。
