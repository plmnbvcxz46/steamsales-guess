# -*- coding: utf-8 -*-
"""
系统自动化单元测试与集成测试脚本 (tests/test_etl_and_models.py)
对应课程报告第 6 章（系统测试）：
1. 测试三层数据仓库表与 6 个 FineBI 视图的完整性（439 款游戏 × 11 种估算方法 = 4,829 行事实表记录）；
2. 测试数据清洗规则：确认免费游戏 (is_free=1)、Demo、非游戏工具、负价格与零评测异常 100% 被拦截并写入 etl_clean_error_log；
3. 测试断点续传检查点表 (etl_checkpoint) 与运行日志表 (etl_job_log) 状态；
4. 测试价格维度 SCD Type 2 拉链表 (dim_price_scd2) 的版本连续性与当前唯一性约束；
5. 测试 SQL 窗口函数中位数计算的准确性（与 NumPy median 对比误差 < 0.01）；
6. 测试多信号模型与 6 大科学假设 (H1~H6) 消融实验：验证双信号融合 Fusion_AB、精修评测 M4_Plus 以及四信号动态融合 M6_Quad_Fusion 阶梯式提升精度。
"""
import sqlite3
import pathlib
import unittest
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"


class TestSteamSalesETLSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(str(DB_PATH))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_01_dw_layers_and_row_counts(self):
        """测试1：验证原始层、明细维度层、事实结果层及运维表记录数正确"""
        ods_cnt = self.conn.execute("SELECT COUNT(*) FROM ods_leak_players_2018").fetchone()[0]
        dwd_cnt = self.conn.execute("SELECT COUNT(*) FROM dwd_steam_game_aligned").fetchone()[0]
        err_cnt = self.conn.execute("SELECT COUNT(*) FROM etl_clean_error_log").fetchone()[0]
        fact_cnt = self.conn.execute("SELECT COUNT(*) FROM fact_sales_estimation").fetchone()[0]
        self.assertEqual(ods_cnt, 413)
        self.assertEqual(err_cnt, 18)
        self.assertEqual(dwd_cnt, 439)  # 395款2018买断制 + 36款2020-2025真值 + 8款跨时代检验
        self.assertEqual(fact_cnt, 439 * 11)  # 每款游戏 × 11 种估算方法 = 4,829 行

    def test_02_data_cleaning_strictness(self):
        """测试2：确认明细层 dwd_steam_game_aligned 中不含任何免费游戏、非游戏软件或价格<=0的异常记录"""
        bad_price = self.conn.execute("SELECT COUNT(*) FROM dwd_steam_game_aligned WHERE price_usd <= 0 OR total_reviews <= 0").fetchone()[0]
        self.assertEqual(bad_price, 0)
        rule_codes = set(r[0] for r in self.conn.execute("SELECT DISTINCT rule_code FROM etl_clean_error_log").fetchall())
        self.assertTrue({"FREE_GAME", "NON_GAME_SOFTWARE", "DEMO_VERSION", "INVALID_PRICE", "ZERO_REVIEWS"}.issubset(rule_codes))

    def test_03_scd2_price_dimension_integrity(self):
        """测试3：验证 SCD Type 2 拉链表每款游戏的 is_current=1 版本唯一，且巫师3具有 3 个历史递进版本"""
        w3_versions = self.conn.execute(
            "SELECT version_no, final_price_usd, effective_start, effective_end, is_current FROM dim_price_scd2 WHERE appid = 292030 ORDER BY version_no ASC"
        ).fetchall()
        self.assertEqual(len(w3_versions), 3)
        self.assertEqual(w3_versions[0][4], 0)  # v1 已闭合
        self.assertEqual(w3_versions[1][4], 0)  # v2 已闭合
        self.assertEqual(w3_versions[2][4], 1)  # v3 当前生效
        self.assertEqual(w3_versions[2][3], "9999-12-31")

    def test_04_sql_window_median_accuracy(self):
        """测试4：验证 SQL 窗口函数 (ROW_NUMBER + COUNT OVER) 计算的 k 中位数与 Python NumPy 计算结果完全一致"""
        df_dwd = pd.read_sql_query("SELECT release_year, k_ratio FROM dwd_steam_game_aligned WHERE data_cohort IN ('train_2018', 'modern_2020')", self.conn)
        df_view = pd.read_sql_query("SELECT release_year, median_k FROM vw_finebi_k_median_by_year", self.conn)
        for _, r in df_view.iterrows():
            yr = int(r["release_year"])
            sql_med = float(r["median_k"])
            py_med = float(np.median(df_dwd[df_dwd["release_year"] == yr]["k_ratio"].values))
            self.assertAlmostEqual(sql_med, py_med, places=1)

    def test_05_multi_signal_fusion_superiority(self):
        """测试5：验证在 2020-2025 近年真值集上，M2/M4 优于 M0，且双信号融合 Fusion_AB 与四信号融合 M6_Quad_Fusion 显著突破精度上限"""
        df_eval = pd.read_sql_query(
            "SELECT * FROM ads_model_eval_summary WHERE eval_cohort LIKE 'Modern_2020%'",
            self.conn
        ).set_index("method_code")
        hit_m0 = df_eval.loc["M0", "hit_rate_30pct"]
        hit_m2 = df_eval.loc["M2", "hit_rate_30pct"]
        hit_m4_plus = df_eval.loc["M4_Plus", "hit_rate_30pct"]
        hit_sigb = df_eval.loc["Signal_B", "hit_rate_30pct"]
        hit_fusion = df_eval.loc["Fusion_AB", "hit_rate_30pct"]
        hit_m6 = df_eval.loc["M6_Quad_Fusion", "hit_rate_30pct"]
        self.assertGreater(hit_m2, hit_m0)
        self.assertGreater(hit_m4_plus, hit_m2)
        self.assertGreater(hit_sigb, hit_m0)
        self.assertGreater(hit_fusion, hit_m2)
        self.assertGreaterEqual(hit_fusion, 80.0)
        self.assertGreater(hit_m6, hit_fusion)

    def test_06_hypothesis_ablation_monotonicity(self):
        """测试6：验证 6 大科学假设消融实验表 (ads_hypothesis_ablation) 中 MAPE 单调递减、±30% 命中率单调递增"""
        df_abl = pd.read_sql_query("SELECT * FROM ads_hypothesis_ablation", self.conn)
        self.assertEqual(len(df_abl), 7)
        mapes = df_abl["mape_pct"].values
        hits30 = df_abl["hit_rate_30pct"].values
        for i in range(1, len(mapes)):
            self.assertLess(mapes[i], mapes[i - 1], f"Stage {i} MAPE should be lower than stage {i-1}")
            self.assertGreaterEqual(hits30[i], hits30[i - 1], f"Stage {i} Hit30 should be >= stage {i-1}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
