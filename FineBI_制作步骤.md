# FineBI 大屏制作步骤

> 给 Rex 在本机 FineBI（6.x）里手动拖图用。所有视图和导出的 CSV 都已准备好，字段名和下面写的一一对应。
> FineBI 只接视图（`vw_finebi_*`），不直接读底层表。

---

## 一、接数据（二选一）

### 方式 A：连本机 MySQL（能展示完整的数据库链路，推荐）

1. 在项目根目录依次执行（顺序不能乱：先建表，再灌数据，再建视图，最后建索引）：
   ```bash
   mysql -u root -p < sql/01_schema_mysql.sql
   mysql -u root -p steam_sales_dw < sql/04_full_data_dump_mysql.sql
   mysql -u root -p steam_sales_dw < sql/02_views_and_models_mysql.sql
   mysql -u root -p steam_sales_dw < sql/03_index_explain_mysql.sql
   ```
2. FineBI → 数据连接 → 新建数据连接 → MySQL：数据库 `steam_sales_dw`，编码 UTF-8，填本机账号密码，测试连接后保存。
3. 公共数据 → 新建文件夹 `Steam销量估算` → 新建数据集 → 数据库表，只勾选下面 8 个视图：

| 视图 | 用途 |
|---|---|
| `vw_finebi_overview_dist` | 样本明细（数据概览、类型内排名与份额） |
| `vw_finebi_review_policy_index` | 老游戏月度评测指数（评测提示政策上线前后） |
| `vw_finebi_k_median_by_year` | 各发售年份 k 中位数（窗口函数求中位数） |
| `vw_finebi_k_median_by_factor` | k 按价格档 / 好评率档 / 主类型 / 是否独立 分组的中位数 |
| `vw_finebi_star_sales_switch` | **星型模型主视图**：样本 × 估算方法，切片器联动整屏 |
| `vw_finebi_method_accuracy` | 各方法准确率（含 Gamalytic 对照） |
| `vw_finebi_price_scd2_timeline` | 价格拉链 + 每小时快照 |
| `vw_sql_m0_m1_estimation` | 纯 SQL 实现的 M0 / M1 |

### 方式 B：直接导 CSV（不想配 MySQL 驱动时）

公共数据 → 新建数据集 → Excel/CSV，上传 `data/cleaned/` 下同名的 CSV（UTF-8 带 BOM，中文不会乱码）。

---

## 二、布局（1920 × 1080，三行三列）

| | 左 | 中 | 右 |
|---|---|---|---|
| 顶部 | 标题 + 数据说明 | 切片器：估算方法 `method_name`、样本组 `cohort` | KPI：样本数、±30% 命中率、中位误差 |
| 第一行 | 组件 1 数据概览 | 组件 2 评测提示政策（核心） | 组件 3 影响因素 |
| 第二行 | 组件 4 方法准确率对标 | 组件 5 估算排行（联动） | 组件 6 估算 vs 真实（联动） |
| 第三行 | 组件 7 k 按发售年份 | 组件 8 价格拉链时间线 | 组件 9 ETL 运维卡片 |

---

## 三、逐个组件

### 全局切片器：估算方法（答辩现场演示）
- 过滤组件 → 下拉框（单选），数据集 `vw_finebi_star_sales_switch`，字段 `method_name`
- 默认选 `融合 A(M4)+B`
- 再加一个下拉框绑 `cohort`，默认 `truth`（真值集 = 公开销量；`train_2018` = 2018 泄露玩家数的交叉验证）
- 效果：切换方法后，KPI、组件 5、组件 6 一起刷新

### KPI 卡片（绑 `vw_finebi_star_sales_switch`，受切片器控制）
- 样本数：`sample_id` 计数
- ±30% 命中率：`within_30pct_flag` 求平均，显示为百分比
- 中位误差：`ape_pct` 取中位数

### 组件 1：数据概览
- 数据集 `vw_finebi_overview_dist`
- 分组柱形图：横轴 `release_year`，纵轴 `sample_id` 计数，颜色 `cohort`
- 再放一个柱形图：横轴 `price_tier`，纵轴计数，过滤 `cohort = train_2018` 且 `model_eligible = 1`

### 组件 2（核心）：评测提示政策上线
- 数据集 `vw_finebi_review_policy_index`
- 折线图：横轴 `month`，纵轴 `median_index`
- 加参考线 y = 1（基线），并把 `is_policy_month = 1` 的那个月（2019-11）标注出来
- 讲法：40 款老游戏的月新增评测，在 2019-11 之后长期是之前的 3 倍多；同样的销量对应的评测变多了，所以 k 变小

### 组件 3：影响因素
- 数据集 `vw_finebi_k_median_by_factor`
- 柱形图：横轴 `factor_value`，纵轴 `median_k`，用过滤组件或分面选 `factor`（价格档 / 好评率档 / 主类型 / 是否独立）

### 组件 4：方法准确率对标
- 数据集 `vw_finebi_method_accuracy`
- 条形图：纵轴 `method_name`，横轴 `hit_rate_30pct`
- 过滤 `eval_cohort` 为 `真值集 全部` 或 `Gamalytic 公布(约120款)`，颜色按 `eval_cohort` 区分
- 讲法：Gamalytic 原文只公布了 ±30% 命中率；另外用他们表里对同一批游戏的估算算了一行 `Gamalytic_same`，这是同口径对比

### 组件 5（联动）：估算排行
- 数据集 `vw_finebi_star_sales_switch`
- 明细表：`game_name`、`release_year`、`true_value`、`estimated_sales`、`estimated_low_bound`、`estimated_high_bound`、`ape_pct`
- 按 `estimated_sales` 降序，条件格式：`within_30pct_flag = 1` 标绿

### 组件 6（联动）：估算 vs 真实
- 数据集 `vw_finebi_star_sales_switch`
- 散点图：横轴 `true_value`，纵轴 `estimated_sales`，两轴都设为对数刻度，颜色 `era_group`
- 加一条 y = x 的参考线；FineBI 画不了 ±30% 带的话，报告里用 `figures/fig06` 代替

### 组件 7：k 按发售年份
- 数据集 `vw_finebi_k_median_by_year`
- 折线图：横轴 `release_year`，纵轴 `median_k`，颜色 `cohort`，纵轴用对数刻度
- 只保留 `game_count >= 3` 的点（过滤条件）

### 组件 8：价格拉链时间线
- 数据集 `vw_finebi_price_scd2_timeline`
- 用过滤组件选一款游戏（`name`）
- 折线图（阶梯样式）：横轴 `effective_start`，纵轴 `final_price_usd`；明细表列出 `version_no`、`effective_start`、`effective_end`、`is_current`、`observation_source`
- 说明：价格来自互联网档案馆存档 + 本次采集时的商店价格，都是真实观测，不是模拟数据

### 组件 9：ETL 运维卡片
- 用 `data/cleaned/etl_clean_error_log.csv` 做一个按 `rule_code` 计数的条形图
- 用 `data/cleaned/etl_job_log.csv` 做一张明细表（`job_name`、`status`、`rows_loaded`、`message`）

---

## 四、和 Python 图的分工

拟合线、误差带、±30% 灰带、置信区间这类统计图 FineBI 不好做，报告里直接用 `figures/` 下的图：

| 图 | 内容 |
|---|---|
| fig01 / fig02 | 原型阶段：k 的分布、对数散点（能看到抽样门槛把下沿截平） |
| fig03 | 数据概览 |
| fig04 | 评测提示政策：月度评测指数 + 各发售年份 k |
| fig05 | 影响 k 的因素 |
| fig06 | 真值集上估算 vs 真实（含 ±30% 带） |
| fig07 | M2 回归系数（含 95% 置信区间）与 M4 特征重要性 |
| fig08 | ±30% 命中率对标 Gamalytic |
| fig09 | 价格拉链（档案馆真实价格） |
| fig10 | 系统架构 |

截图建议：切片器分别选 `M0 固定倍数 k=30` 和 `融合 A(M4)+B` 各截一张整屏，放进报告 6.4 节对比。
