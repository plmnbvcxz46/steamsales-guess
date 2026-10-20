# -*- coding: utf-8 -*-
"""
自动化测试 (tests/test_etl_and_models.py)，对应报告第 6 章。
先跑完 etl_pipeline.py 和 models.py 再运行：python tests/test_etl_and_models.py

1. 分层完整：每个可建模样本在事实表里都有该有的方法行
2. 清洗规则：明细层里没有免费/非游戏/无价格/零评测/上架晚于截止日的记录
3. 不泄露：改动第 i 款游戏的真值，它自己的估算不变（留一法、嵌套留一法都要满足）
4. 不泄露：模型输入特征与真值无关（把真值打乱，特征矩阵一模一样）
5. SCD2：每款游戏恰好一个当前版本，版本首尾相接
6. 窗口函数中位数 = NumPy 中位数
7. 断点续传：中途中断再续跑，结果与一次跑完完全相同
8. 增量抽取：水位线等于已入库的最大时间，再跑一次不会重复装载
9. 索引：建索引后执行计划改走索引
10. 不泄露（按游戏留一法）：同一款游戏有多个里程碑时，改动这款游戏所有行的真值，它自己的估算不变
11. 按可信度加权的融合：同样不泄露；只有 M4 可用时结果就是 M4；结果夹在 A 与（修正偏差后的）B 之间
"""
import sys
import shutil
import sqlite3
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
import etl_pipeline  # noqa: E402
import models  # noqa: E402

DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"


class TestSteamSalesSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(str(DB_PATH))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def q(self, sql):
        return pd.read_sql_query(sql, self.conn)

    def test_01_layers_and_fact_completeness(self):
        n_ars = self.conn.execute("SELECT COUNT(*) FROM ods_ars_players_2018").fetchone()[0]
        self.assertEqual(n_ars, 13281)  # 原始 CSV 的全部行（含 6 组重复）原样入原始层
        n_tr = self.conn.execute("SELECT COUNT(*) FROM dwd_game_sample WHERE cohort='train_2018' AND model_eligible=1").fetchone()[0]
        n_te = self.conn.execute("SELECT COUNT(*) FROM dwd_game_sample WHERE cohort='truth' AND model_eligible=1").fetchone()[0]
        n_v2 = self.conn.execute("SELECT COUNT(*) FROM dwd_game_sample WHERE cohort='truth_v2' AND model_eligible=1").fetchone()[0]
        self.assertGreater(n_tr, 300)
        self.assertGreater(n_te, 50)
        f = self.q("SELECT cohort, method_code, COUNT(*) n FROM fact_sales_estimation GROUP BY 1, 2")
        cnt = {(r.cohort, r.method_code): r.n for r in f.itertuples()}
        for m in ["M0", "M1", "M2", "M3", "M4"]:
            self.assertEqual(cnt[("train_2018", m)], n_tr)
        for m in ["M0", "M1", "M2", "M3", "M4", "Fusion_AB"]:
            self.assertEqual(cnt[("truth", m)], n_te)
            self.assertEqual(cnt.get(("truth_v2", m), 0), n_v2)

    def test_02_cleaning_rules(self):
        bad = self.conn.execute(
            """SELECT COUNT(*) FROM dwd_game_sample WHERE price_usd <= 0 OR total_reviews <= 0
               OR release_date > cutoff_date""").fetchone()[0]
        self.assertEqual(bad, 0)
        free_in_dwd = self.conn.execute(
            """SELECT COUNT(*) FROM dwd_game_sample d JOIN ods_steam_appdetails a ON a.appid = d.appid
               WHERE a.is_free = 1 OR a.app_type <> 'game' OR a.success = 0""").fetchone()[0]
        self.assertEqual(free_in_dwd, 0)
        codes = {r[0] for r in self.conn.execute("SELECT DISTINCT rule_code FROM etl_clean_error_log")}
        self.assertTrue({"FREE_GAME", "NO_PRICE", "TRUNCATION_RISK"}.issubset(codes))
        # 被标记 TRUNCATION_RISK 的样本都不参与建模
        n = self.conn.execute("SELECT COUNT(*) FROM dwd_game_sample WHERE cohort='train_2018' AND total_reviews < ? AND model_eligible = 1",
                              (etl_pipeline.MIN_REVIEWS_2018,)).fetchone()[0]
        self.assertEqual(n, 0)

    def _truth_inputs(self):
        df = self.q("SELECT * FROM dwd_game_sample ORDER BY sample_id")
        tr = df[(df["cohort"] == "train_2018") & (df["model_eligible"] == 1)].reset_index(drop=True)
        te = df[df["cohort"] == "truth"].reset_index(drop=True)
        pt = float(tr["playtime_review_median_hrs"].median())
        X_tr, _ = models.build_features(tr, pt)
        X_te, _ = models.build_features(te, pt)
        beta = models.fit_ols(X_tr, tr["log_k"].values)["beta"]
        off = X_te @ beta
        Z = models.recal_design(te, X_te)
        y_lk = np.log(te["truth_value"].values / te["total_reviews"].values)
        return te, Z, off, y_lk

    def test_03_no_leakage_loo(self):
        te, Z, off, y_lk = self._truth_inputs()
        base = models.loo_recal(Z, off, y_lk)
        for i in [0, len(y_lk) // 2, len(y_lk) - 1]:
            y2 = y_lk.copy()
            y2[i] += 5.0  # 把第 i 款的真值放大 e^5 倍
            again = models.loo_recal(Z, off, y2)
            self.assertAlmostEqual(base[i], again[i], places=9)  # 它自己的估算不受影响
        # 嵌套留一法融合同理
        has_b = np.array([True] * len(y_lk))
        log_n = np.log(te["total_reviews"].values)
        log_b = log_n + 3.0 + 0.1 * np.arange(len(y_lk)) / len(y_lk)
        f0, _ = models.nested_loo_fusion(Z, off, y_lk, log_n, log_b, has_b)
        y2 = y_lk.copy()
        y2[3] -= 4.0
        f1, _ = models.nested_loo_fusion(Z, off, y2, log_n, log_b, has_b)
        self.assertAlmostEqual(f0[3], f1[3], places=9)

    def test_04_features_do_not_depend_on_truth(self):
        df = self.q("SELECT * FROM dwd_game_sample WHERE cohort='truth' ORDER BY sample_id")
        X1, _ = models.build_features(df, 10.0)
        df2 = df.copy()
        df2["truth_value"] = np.random.default_rng(0).permutation(df2["truth_value"].values)
        df2["k_ratio"] = df2["truth_value"] / df2["total_reviews"]
        df2["log_k"] = np.log(df2["k_ratio"])
        X2, _ = models.build_features(df2, 10.0)
        self.assertTrue(np.array_equal(X1, X2))
        # ext 特征集（语言占比、价格历史等）同样与真值无关
        E1, _ = models.build_features(df, 10.0, "ext", models.fit_fills(df))
        E2, _ = models.build_features(df2, 10.0, "ext", models.fit_fills(df2))
        self.assertTrue(np.allclose(E1, E2, equal_nan=True))

    def test_05_scd2_integrity(self):
        p = self.q("SELECT * FROM dim_price_scd2 ORDER BY appid, version_no")
        if p.empty:
            self.skipTest("没有价格拉链数据")
        for appid, g in p.groupby("appid"):
            self.assertEqual(int(g["is_current"].sum()), 1)
            self.assertEqual(g[g["is_current"] == 1]["effective_end"].iloc[0], "9999-12-31")
            self.assertEqual(list(g["version_no"]), list(range(1, len(g) + 1)))
            ends, starts = g["effective_end"].tolist()[:-1], g["effective_start"].tolist()[1:]
            self.assertEqual(ends, starts)  # 上一版本的结束日 = 下一版本的开始日

    def test_06_sql_window_median(self):
        d = self.q("SELECT cohort, release_year, k_ratio FROM dwd_game_sample WHERE model_eligible = 1")
        v = self.q("SELECT cohort, release_year, median_k FROM vw_finebi_k_median_by_year")
        for r in v.itertuples():
            py = float(np.median(d[(d["cohort"] == r.cohort) & (d["release_year"] == r.release_year)]["k_ratio"]))
            self.assertAlmostEqual(r.median_k, round(py, 2), places=1)

    def test_07_checkpoint_resume_equals_full_run(self):
        tmp = pathlib.Path(tempfile.mkdtemp()) / "dw.db"
        shutil.copy(DB_PATH, tmp)
        conn = sqlite3.connect(str(tmp))
        policy = etl_pipeline.load_policy()
        with conn:
            conn.execute("UPDATE etl_checkpoint SET status='COMPLETED' WHERE job_name='CLEAN_ODS_TO_DWD'")
        with self.assertRaises(etl_pipeline.SimulatedCrash):
            etl_pipeline.clean_ods_to_dwd(conn, policy, crash_after_batches=2)
        status = conn.execute("SELECT status FROM etl_checkpoint WHERE job_name='CLEAN_ODS_TO_DWD'").fetchone()[0]
        self.assertEqual(status, "INTERRUPTED")
        etl_pipeline.clean_ods_to_dwd(conn, policy)
        cols = "appid, cohort, cutoff_date, total_reviews, truth_value, model_eligible"
        a = pd.read_sql_query(f"SELECT {cols} FROM dwd_game_sample ORDER BY appid, cohort, cutoff_date", conn)
        b = self.q(f"SELECT {cols} FROM dwd_game_sample ORDER BY appid, cohort, cutoff_date")
        conn.close()
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_08_incremental_watermark(self):
        wm = self.conn.execute("SELECT last_update_time FROM etl_checkpoint WHERE job_name='INCR_SNAPSHOT'").fetchone()
        if wm is None:
            self.skipTest("没有快照数据")
        mx = self.conn.execute("SELECT MAX(fetched_at) FROM fact_snapshot").fetchone()[0]
        self.assertEqual(wm[0], mx)
        tmp = pathlib.Path(tempfile.mkdtemp()) / "dw.db"
        shutil.copy(DB_PATH, tmp)
        conn = sqlite3.connect(str(tmp))
        before = conn.execute("SELECT COUNT(*) FROM fact_snapshot").fetchone()[0]
        etl_pipeline.incremental_snapshot_load(conn)  # 原始文件没有新行时，再跑一次不该多出任何行
        after = conn.execute("SELECT COUNT(*) FROM fact_snapshot").fetchone()[0]
        conn.close()
        raw_max = pd.read_csv(BASE_DIR / "data" / "raw" / "snapshots.csv")["fetched_at"].max()
        if raw_max <= mx:
            self.assertEqual(before, after)

    def test_09_index_used_after_creation(self):
        plan = self.conn.execute("EXPLAIN QUERY PLAN " + models.EXPLAIN_SQL).fetchall()
        self.assertTrue(any("idx_fact_cohort_method" in str(r[-1]) for r in plan))



    def test_10_no_leakage_leave_one_game_out(self):
        df = self.q("SELECT * FROM dwd_game_sample ORDER BY sample_id")
        tr = df[(df["cohort"] == "train_2018") & (df["model_eligible"] == 1)].reset_index(drop=True)
        te = df[df["cohort"].isin(["truth", "truth_v2"]) & (df["model_eligible"] == 1)].reset_index(drop=True)
        pt = float(tr["playtime_review_median_hrs"].median())
        X_tr, _ = models.build_features(tr, pt)
        X_te, _ = models.build_features(te, pt)
        off = X_te @ models.fit_ols(X_tr, tr["log_k"].values)["beta"]
        Z = models.recal_design(te, X_te)
        y_lk = np.log(te["truth_value"].values / te["total_reviews"].values)
        groups = te["appid"].values
        multi = te["appid"].value_counts()
        appid = multi.index[0]  # 里程碑最多的那款游戏
        rows = np.where(groups == appid)[0]
        base = models.loo_recal(Z, off, y_lk, groups)
        y2 = y_lk.copy()
        y2[rows] += 5.0
        again = models.loo_recal(Z, off, y2, groups)
        np.testing.assert_allclose(base[rows], again[rows], atol=1e-9)


    def test_11_precision_fusion(self):
        df = self.q("SELECT * FROM dwd_game_sample ORDER BY sample_id")
        tr = df[(df["cohort"] == "train_2018") & (df["model_eligible"] == 1)].reset_index(drop=True)
        te = df[df["cohort"].isin(["truth", "truth_v2"]) & (df["model_eligible"] == 1)].reset_index(drop=True)
        pt = float(tr["playtime_review_median_hrs"].median())
        X_tr, _ = models.build_features(tr, pt)
        X_te, _ = models.build_features(te, pt)
        off = X_te @ models.fit_ols(X_tr, tr["log_k"].values)["beta"]
        Z = models.recal_design(te, X_te)
        y_lk = np.log(te["truth_value"].values / te["total_reviews"].values)
        log_n = np.log(te["total_reviews"].values)
        has_b = (te["charts_coverage"] == 1).values & te["playtime_review_median_hrs"].notna().values
        ln_b = np.where(has_b, log_n + 4.0 + 0.3 * np.sin(np.arange(len(te))), 0.0)
        qa, qb = models.quality_design(te)
        groups = te["appid"].values
        args = (Z, off, y_lk, log_n, ln_b, has_b, qa, qb, groups)
        base, _ = models.logo_fusion(*args, "quality")
        g = te["appid"].value_counts().index[0]
        rows = np.where(groups == g)[0]
        y2 = y_lk.copy()
        y2[rows] += 5.0
        again, _ = models.logo_fusion(Z, off, y2, log_n, ln_b, has_b, qa, qb, groups, "quality")
        np.testing.assert_allclose(base[rows], again[rows], atol=1e-9)
        # 没有信号 B 的行：结果等于（同一折校准的）M4；有 B 的行：夹在 A、B 之间
        m4, _ = models.logo_fusion(*args, "m4")
        np.testing.assert_allclose(base[~has_b], m4[~has_b], atol=1e-9)
        # 夹在 A 与"修正平均偏差后的 B"之间（B 没在真值上校准过，融合时先加上它的平均偏差）
        params = models.fusion_fit(*args, "quality")
        fused, det = models.fusion_apply(params, Z, off, log_n, ln_b, has_b, qa, qb)
        b_adj = ln_b + params["B"][0]
        lo, hi = np.minimum(det["ln_a"], b_adj), np.maximum(det["ln_a"], b_adj)
        self.assertTrue(np.all((fused[has_b] >= lo[has_b] - 1e-9) & (fused[has_b] <= hi[has_b] + 1e-9)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
