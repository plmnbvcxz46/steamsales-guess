# 基于 ETL 的 Steam 游戏销量估算与智能分析系统 (steamsales-guess)

> 课程：高级数据库技术与应用 · 大作业
> 核心问题：Steam 从不公布单款游戏的销量。只用公开可采集的数据（评测、价格、在线人数），一款游戏在 Steam 上到底卖了多少份？

**全部数据为真实采集**：模型输入来自 Steam 官方接口和 SteamCharts，真值来自 2018 年 Ars Technica 公开的泄露数据与有出处的公开销量。没有模拟数据，也没有由真值反推出来的特征。

---

## 一、结果一览

| 测试集 | 方法 | ±30% 命中率 | 说明 |
|---|---|---|---|
| 2018 训练集 1,795 款（真值 = 泄露玩家数） | M0 固定 30 倍 | 9.7% | 五折交叉验证 |
| | M4 树模型 | 54.6% | ln k 的 R² 0.491（只用原有特征时 0.447） |
| 旧真值集 107 款（Gamalytic 测试表） | M0 固定 30 倍 | 31.8% | |
| | M1 按年份文献 k | 40.2% | |
| | M4 树模型 + 近年校准 | 50.5% | 校准系数按游戏留一法 |
| | 融合 A+B | **53.3%** | 嵌套留一法 |
| 新真值集 57 条（开发商原话写明 Steam） | M4 / 融合 | 50.9% / 47.4% | 同上，与旧真值合并校准 |
| 新旧合并 164 条 / 157 款 | 融合 A+B | 51.2%（±50% 内 76.8%） | |
| 合并真值中信号 B 可用的 81 条 | 融合 A+B | **58.0%**（±50% 内 85.2%） | |
| 外部检验：只用旧真值校准，估新真值集 | 融合（原有特征 / 加新特征） | 45.6% / 49.1% | 两批游戏没有重复，模型完全没见过 |
| 对照：旧真值集同一批 107 款 | Gamalytic 自己的估算 | 73.8% | 来自 Gamalytic 测试表 |
| 对照：Gamalytic 原文（约 120 款） | 评测 × 固定 / 修正倍数 | 42.7% / 50.4% | 原文只公布了这一列 |

注：真值集样本少，命中率的 95% 置信区间大约是 ±8~13 个百分点。2026-10-20 加入新特征和新真值后，旧真值集上融合从 48.6% 提到 53.3%，但配对自助法的区间是 [−1.9, +11.2] 个百分点，还不能说显著。

几条主要发现：
1. **"游玩后提示写评测"是 2019 年 11 月上线的。** 这是从 40 款老游戏的月度评测量测出来的：此后月新增评测持续是之前的 3.5 倍（`fig04`）。在近年真值上独立校准出的政策系数，折算成 k 的倍数是 0.503，和"k 减半"的说法一致。
2. **2018 年的"玩家数"和近年的"销量"不是同一个口径。** 直接把 2018 年模型迁移过去，残差会随游戏年龄、Steam 内购买评测占比系统性变化，所以这些系数必须用近年真值重估（设计文档 3.6 节的第二种情况）。
3. **只靠评测这一路信号，命中率大约 40%~46%**，与 Gamalytic 公布的同类方法一致；加入在线人数信号后提升到 50% 以上。和 Gamalytic 仍有差距，主要因为对方还有"抽样公开玩家资料"这一路信号，我们拿不到。
4. **补充特征里最有用的是"深度折扣"。** 截止日前打 75% 以上折扣的时间占比，在 2018 数据的树模型里重要性排第二，回归系数也最显著：深度折扣带来的买家很少写评测，k 明显偏高。价格也改成了截止日当时的原价（IsThereAnyDeal 价格历史），不再用 2026 年抓取时的商店价。
5. **真值本身要严格把关。** 第二批真值由 AI 收集，原始 518 条里只有 57 条最终参与校准：维基百科抄来的、原句没写 Steam 的、"首日卖了多少"却配了几年后评测数的，都会把模型带偏（见 `src/truth_v2_merge.py` 和开发日志）。
6. **实测《红色沙漠》**（`src/case_study.py`，`figures/case_crimson_desert.png`）：只看评测的方法对这款 70 美元的 3A 大作高估了 50%~85%，信号 B 和融合更接近由官方数字推出的参照区间。真值集里几乎都是独立游戏，大作的 k 只能靠外推。

---

## 二、目录

```text
steamsales-guess/
├── run_all.py                    # 一条命令重跑离线部分：ETL -> 建模 -> 作图 -> 转储 -> 测试
├── src/
│   ├── crawler.py                # 采集：appdetails、评测(input_json)、SteamCharts、公告；缓存 + 检查点续传
│   ├── crawler_extra.py          # 补充采集：各语言评测数、IsThereAnyDeal 价格历史/区域价/慈善包、新真值集
│   ├── truth_v2_merge.py         # 合并两轮 AI 收集的官方销量，按口径分层、换算"首日/首周"的时间点
│   ├── detect_policy_change.py   # 用 40 款老游戏的月度评测量判定评测提示政策的上线月份
│   ├── price_history_wayback.py  # 从互联网档案馆存档提取历史价格（SCD2 用）
│   ├── snapshot_collector.py     # 每小时快照：在线人数、评测数、价格（增量抽取用）
│   ├── prototype_explore.py      # 原型：200 款游戏，k 的分布与对数散点
│   ├── etl_pipeline.py           # ETL：原始层 -> 明细层/维度 -> 结果层，分批事务 + 检查点、SCD2、增量快照
│   ├── models.py                 # M0~M4、信号 B、融合；交叉验证 / 留一法；索引 EXPLAIN 对比
│   ├── news_extract_audit.py     # 官方公告销量句子：抽取、规则对照、与真值交叉核对、人工核对清单
│   ├── case_study.py             # 单款游戏实测：给 appid 和日期，采集并用全部方法估算（例：红色沙漠）
│   ├── visualize_all.py          # 报告用统计图 fig03~fig10
│   ├── plot_style.py             # 作图公共样式
│   └── export_mysql_dump.py      # SQLite -> MySQL INSERT 脚本
├── sql/
│   ├── 01_schema_mysql.sql       # 建表（原始层 7 张、维度 5 张、明细 1 张、结果 6 张、运维 3 张、Kettle 输出 2 张）
│   ├── 02_views_and_models_mysql.sql  # M0/M1、窗口函数中位数、8 个 FineBI 视图（SQLite 和 MySQL 通用）
│   ├── 03_index_explain_mysql.sql     # 复合索引与 EXPLAIN
│   └── 04_full_data_dump_mysql.sql    # 全量数据（由 export_mysql_dump.py 生成）
├── kettle/                       # clean_steam_games.ktr、scd2_price_history.ktr、steam_etl_main.kjb
├── data/
│   ├── raw/                      # 采集结果（见第四节）
│   │   └── source/               # 两份原始外部文件：Ars 泄露表原件、Gamalytic 测试表转录件
│   ├── cleaned/                  # 视图与结果表导出的 CSV、EXPLAIN 报告、公告抽取审核表
│   ├── cache/                    # 采集缓存（不入库，只保留 crawl_log.csv 与检查点）
│   └── steam_sales_dw.db         # SQLite 数据仓库
├── figures/                      # fig01~fig10
├── tests/test_etl_and_models.py  # 9 项测试（含"不泄露"测试）
├── 课程报告.md / 开发日志.md / FineBI_制作步骤.md / 最终总结.md
```

---

## 三、运行

依赖：Python 3.10+，`numpy pandas scipy matplotlib requests`（`pip install numpy pandas scipy matplotlib requests`）。不需要 sklearn，树模型是手写的。

### 1. 只重跑离线部分（推荐，不联网，1 分钟左右）
```bash
python run_all.py --fresh
```

### 2. 从头重新采集（联网，单进程约 5 小时，可多开进程分片）
```bash
python src/crawler.py policy_scan                 # 40 款老游戏的月度评测（约 10 分钟）
python src/detect_policy_change.py                # 判定政策月份 -> data/raw/policy_change_detected.json
python src/crawler.py train2018 --shard 0/2       # 2018 训练集第一轮（玩家数 >= 2 万），可开两个进程 0/2、1/2
python src/crawler.py train2018 --extend          # 扩展轮（玩家数 5000~2 万），可再开一个 --reverse 从另一头抓
python src/crawler.py truth                       # 118 条公开销量对应的评测（截到公布日）
python src/crawler.py first_review                # 抢先体验游戏补查第一条评测日期
python src/crawler.py news_scan                   # 官方公告里的销量句子
python src/crawler.py export                      # 缓存 -> data/raw/*.csv
python src/truth_v2_merge.py                      # 合并新真值集（data/raw/source/ 下的两轮收集结果）
python src/crawler_extra.py jobs                  # 补充采集清单（需要先跑过一次 ETL）
python src/crawler_extra.py truth_v2              # 新真值集的评测、商店、SteamCharts
python src/crawler_extra.py langs                 # 各语言评测数（约 2.7 万次请求，0.3 秒一次约 2 小时）
python src/crawler_extra.py itad                  # 价格历史/慈善包，需要 .env 里的 ITAD_API_KEY（账号要验证邮箱）
python src/crawler_extra.py export                # 缓存 -> data/raw/ 下的补充 CSV
python src/price_history_wayback.py               # 档案馆价格（档案馆较慢，可多次运行，靠缓存续跑）
python src/snapshot_collector.py --rounds 3       # 每小时快照，可用 Windows 任务计划程序定期运行
python run_all.py
```
中途断掉直接重跑同一条命令即可，已完成的请求会从缓存读取。

### 3. 导入 MySQL 8.0（给 FineBI 和 Kettle 用）
```bash
mysql -u root -p < sql/01_schema_mysql.sql
mysql -u root -p steam_sales_dw < sql/04_full_data_dump_mysql.sql
mysql -u root -p steam_sales_dw < sql/02_views_and_models_mysql.sql
mysql -u root -p steam_sales_dw < sql/03_index_explain_mysql.sql
```
这四个脚本已在 MariaDB 10.11 上完整执行过一遍，8 个视图的行数、k 中位数与 SQLite 版逐行一致。

### 4. Kettle
用 Spoon（PDI 9.x）打开 `kettle/steam_etl_main.kjb`，在数据库连接 `Steam_MySQL_DW` 里填本机 MySQL 密码后运行。需要把 `mysql-connector-j.jar` 放进 `data-integration/lib/`。
**注意：开发环境里没有装 Kettle，这三个文件只校验过 XML 格式，没有在 Spoon 里实际跑过**，第一次打开时如果报错，按提示调整即可。

---

## 四、数据来源

| 数据 | 来源 | 文件 |
|---|---|---|
| 2018-07-01 玩家数（真值） | Ars Technica 公开的成就接口泄露表 `arstechnica.com/wp-content/uploads/2018/07/games_achievements_players_2018-07-01.csv`，13,281 行 | `data/raw/source/…_ars_original.csv`、`ars_players_2018.csv` |
| 近年公开销量（真值） | Gamalytic《How to accurately estimate Steam sales》测试表，118 行，每行附原始出处 | `data/raw/source/gamalytic_benchmark_118.csv`、`recent_sales_truth.csv` |
| 商店信息 | `store.steampowered.com/api/appdetails`（cc=us） | `steam_appdetails.csv` |
| 评测 | `api.steampowered.com/IUserReviewsService/GetAppReviews/v1/`，`input_json` 传参，按截止日截断 | `steam_reviews_at_cutoff.csv` |
| 月度在线 | `steamcharts.com/app/<appid>` | `steamcharts_monthly.csv` |
| 官方公告 | `api.steampowered.com/ISteamNews/GetNewsForApp/v2/` | `news_sales_sentences.csv` |
| 历史价格 | 互联网档案馆 `web.archive.org` 的商店页存档 | `price_history_wayback.csv` |
| 实时快照 | `ISteamUserStats/GetNumberOfCurrentPlayers` + 评测汇总 + appdetails | `snapshots.csv` |

合规：没有访问 SteamDB（其 FAQ 禁止爬取）；所有接口都有限速（商店 1.5 秒/次、API 0.8 秒/次、SteamCharts 1.5 秒/次、档案馆每线程 2 秒/次），遇到 429 会退避；Gamalytic 的数据只用于学术对照，并注明出处。
