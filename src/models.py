# -*- coding: utf-8 -*-
"""
多方法销量估算、交叉验证、政策校准、双信号融合与结果回写 (src/models.py)

三类样本：
    train_2018：2018-07-01 泄露玩家数 + 截至当天的评测（只用 model_eligible=1 的，避开抽样截断）
    truth     ：Gamalytic 测试表里 118 条有出处的公开销量 + 截至公布日的评测
    truth_v2  ：2026-10-20 新收集的开发商/发行商公布的 Steam 销量（只用 Steam 口径的分层，见 src/truth_v2_merge.py）
近年校准与融合在 truth + truth_v2 上做；同一款游戏可能有好几个里程碑，所以"留一法"按游戏留出（leave-one-game-out）。
另外单独做一次外部检验：只用旧的 truth 校准，去估 truth_v2，两批游戏没有重复。

几条硬规矩（原版本就是在这里出的问题）：
1. 任何模型输入都只来自接口采集的数据，绝不由真值反推；
2. 2018 样本上的指标一律是 5 折交叉验证的"袋外"预测；
3. 凡是要在真值集上拟合的参数（政策校准系数、融合权重），一律用留一法：
   估计第 i 款游戏时，参数只用另外 n-1 款拟合，所以第 i 款的误差是真正的"没见过"的误差。
"""
import json
import time
import math
import sqlite3
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_sales_dw.db"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
RNG_SEED = 20261015

GENRES = ["Action", "Adventure", "RPG", "Strategy", "Simulation", "Casual", "Sports & Racing",
          "Massively Multiplayer", "Indie Only", "Other"]
POS_CENTER = 0.80  # 好评率二次项的中心，只为数值稳定，不影响拟合出的曲线形状

# Gamalytic 博客公布的、约 120 款游戏上各方法的 ±30% 命中率（只有这一列是原文给出的）
GAMALYTIC_PUBLISHED = [
    ("G_Fixed", "Gamalytic: 评测数×固定倍数", 42.74),
    ("G_Adjusted", "Gamalytic: 评测数×修正倍数", 50.43),
    ("G_CCU_Playtime", "Gamalytic: 在线人数+游玩时长", 64.44),
    ("G_TopSeller", "Gamalytic: 热销榜排名", 61.40),
    ("G_Profiles", "Gamalytic: 抽样公开玩家资料", 77.91),
    ("G_Aggregated", "Gamalytic: 多方法加权", 76.92),
]


# ==============================================================================
# 一、特征
# ==============================================================================
# ext 特征集新增的列：(dwd 列名或派生规则, 特征代码, 中文名)
EXT_FEATURES = [
    ("ln_launch_over_cutoff", "ln(首发价 / 截止日原价)"),
    ("log1p_discounts", "ln(1 + 截止日前打折次数)"),
    ("max_cut", "截止日前最深折扣"),
    ("frac_time_discounted", "打折时间占比"),
    ("frac_time_cut75", "75% 以上折扣时间占比"),
    ("log1p_bundles", "ln(1 + 进慈善包次数)"),
] + [(f"share_{l}", f"{l} 评测占比") for l in ["english", "russian", "german", "french", "polish", "brazilian",
                                               "spanish_all", "japanese", "koreana", "tchinese", "turkish"]] + [
    ("ln_regional_cn", "中国区相对定价"), ("ln_regional_tr", "土耳其区相对定价"), ("ln_regional_br", "巴西区相对定价"),
]


def _ext_raw(df):
    """从明细层的列算出 ext 特征的原始值（缺失的保留 NaN，交给 fills 填）。df 里没有的列一律当缺失。"""
    def col(c):
        return df[c].astype(float) if c in df else pd.Series(np.nan, index=df.index)
    price_now = np.exp(df["log_price"].astype(float))
    pc = col("price_at_cutoff_usd")
    raw = {
        "log_price_at_cutoff": np.log(pc.where(pc > 0, price_now)),
        "ln_launch_over_cutoff": np.log(col("launch_price_usd") / pc),
        "log1p_discounts": np.log1p(col("n_discounts")),
        "max_cut": col("max_cut") / 100.0,
        "frac_time_discounted": col("frac_time_discounted"),
        "frac_time_cut75": col("frac_time_cut75"),
        "log1p_bundles": np.log1p(col("n_bundles")),
    }
    for c, _ in EXT_FEATURES:
        if c.startswith("share_") or c.startswith("ln_regional_"):
            raw[c] = col(c)
    return pd.DataFrame(raw, index=df.index).replace([np.inf, -np.inf], np.nan)


def fit_fills(tr):
    """ext 特征的缺失值填充值：一律取 2018 训练样本的中位数（只看训练集，不看真值集）。"""
    raw = _ext_raw(tr)
    return {c: float(raw[c].median()) if raw[c].notna().any() else 0.0 for c in raw.columns}


def build_features(df, pt_fill, feature_set="base", fills=None):
    """
    ln k 的解释变量。每一项为什么放进来：
    - ln 价格：越贵买家越认真，评测意愿越高，k 越低（文献）
    - 好评率及其平方：文献说好评率与 k 是倒 U 形
    - ln 评测数：文献说评测很少和很多的游戏 k 都偏低
    - ln 游戏年龄：老游戏多年来攒了慈善包、送 Key、免费周末的玩家，k 越老越高（我们自己的数据里很明显）
    - 中文评测占比：文献矛盾，作为待检验特征
    - Steam 内购买评测占比：越低说明激活码/赠送来的玩家越多，这些人写评测少（评测接口 purchase_type 拆出来的真实值）
    - ln 评测者时长：玩得越久越愿意写评测
    - 类型虚拟变量：以 Action 为基准
    """
    n = len(df)
    pos = df["positive_rate"].values
    pt = df["playtime_review_median_hrs"].fillna(pt_fill).clip(lower=0.1).values
    if feature_set == "ext":
        fills = fills if fills is not None else fit_fills(df)
        raw = _ext_raw(df).fillna(fills)
        # ext 用截止日当时的原价（ITAD 价格历史），没有记录时退回 2026 年抓取的商店价
        log_price = raw["log_price_at_cutoff"].values
    else:
        log_price = df["log_price"].values
    cols = [
        np.ones(n),
        log_price,
        pos - POS_CENTER,
        (pos - POS_CENTER) ** 2,
        df["log_reviews"].values,
        np.log(df["game_age_years"].values),
        df["schinese_ratio"].values,
        df["steam_purchase_ratio"].values,
        np.log(pt),
        df["is_indie"].values.astype(float),
    ]
    names = [
        ("intercept", "截距"),
        ("log_price", "ln 价格"),
        ("positive_rate_c", "好评率 - 0.8"),
        ("positive_rate_sq", "(好评率 - 0.8)²"),
        ("log_reviews", "ln 评测数"),
        ("log_age", "ln 游戏年龄(年)"),
        ("schinese_ratio", "简中评测占比"),
        ("steam_purchase_ratio", "Steam 内购买评测占比"),
        ("log_playtime", "ln 评测者时长中位数"),
        ("is_indie", "是否独立游戏"),
    ]
    for g in GENRES[1:]:
        cols.append((df["genre"].values == g).astype(float))
        names.append((f"genre_{g}", f"类型: {g}"))
    if feature_set == "ext":
        names[1] = ("log_price", "ln 价格（截止日当时的原价）")
        for c, cn in EXT_FEATURES:
            v = raw[c].values
            if c.startswith("ln_regional_"):
                v = v - fills[c]  # 各区推荐价对美元价大体是固定比例，看的是相对同类游戏偏贵还是偏便宜
            cols.append(v)
            names.append((c, cn))
    return np.column_stack(cols), names


# ==============================================================================
# 二、M2 最小二乘 / M3 分位数回归
# ==============================================================================
def fit_ols(X, y):
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / (n - p)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    t = beta / se
    pv = 2 * (1 - stats.t.cdf(np.abs(t), df=n - p))
    return {"beta": beta, "se": se, "t": t, "p": pv, "r2": 1 - rss / tss, "resid": resid}


def fit_quantile(X, y, tau):
    """最小化分位数损失（pinball loss）。用 OLS 结果当初值，Powell 法不需要梯度，对折线损失更稳。"""
    b0, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    def loss(b):
        u = y - X @ b
        return np.mean(np.maximum(tau * u, (tau - 1) * u))

    res = minimize(loss, b0, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-9})
    return res.x


# ==============================================================================
# 三、M4 手写 CART 回归树、随机森林、梯度提升树
# 为什么手写：不依赖 sklearn 等黑盒库，答辩时能讲清楚每个节点怎么分裂、特征重要性怎么来的
# ==============================================================================
class SimpleCARTTree:
    def __init__(self, max_depth=4, min_samples_leaf=8, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.tree = None
        self.feat_gains = None

    def fit(self, X, y):
        self.feat_gains = np.zeros(X.shape[1])
        self.tree = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        n, p = X.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.var(y) < 1e-9:
            return {"leaf": True, "val": float(np.mean(y))}
        feats = np.arange(p)
        if self.max_features and self.max_features < p:
            feats = self.rng.choice(p, self.max_features, replace=False)
        parent_sse = float(((y - y.mean()) ** 2).sum())
        best = (0.0, None, None)
        for f in feats:
            col = X[:, f]
            # 候选切分点只取若干分位数，速度快很多，效果差别不大
            for thr in np.unique(np.percentile(col, [10, 20, 30, 40, 50, 60, 70, 80, 90])):
                left = col <= thr
                nl = int(left.sum())
                if nl < self.min_samples_leaf or n - nl < self.min_samples_leaf:
                    continue
                yl, yr = y[left], y[~left]
                gain = parent_sse - float(((yl - yl.mean()) ** 2).sum() + ((yr - yr.mean()) ** 2).sum())
                if gain > best[0]:
                    best = (gain, f, float(thr))
        if best[1] is None:
            return {"leaf": True, "val": float(np.mean(y))}
        gain, f, thr = best
        self.feat_gains[f] += gain
        left = X[:, f] <= thr
        return {"leaf": False, "f": f, "thr": thr,
                "l": self._build(X[left], y[left], depth + 1), "r": self._build(X[~left], y[~left], depth + 1)}

    def _one(self, node, x):
        while not node["leaf"]:
            node = node["l"] if x[node["f"]] <= node["thr"] else node["r"]
        return node["val"]

    def predict(self, X):
        return np.array([self._one(self.tree, x) for x in X])


class RandomForest:
    def __init__(self, n_estimators=60, max_depth=6, min_samples_leaf=10, seed=0):
        self.n_estimators, self.max_depth, self.min_samples_leaf = n_estimators, max_depth, min_samples_leaf
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, p = X.shape
        self.trees, gains = [], np.zeros(p)
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, n)  # 有放回抽样
            t = SimpleCARTTree(self.max_depth, self.min_samples_leaf, max(2, int(p * 0.6)), self.rng).fit(X[idx], y[idx])
            self.trees.append(t)
            gains += t.feat_gains
        self.feature_importances_ = gains / max(gains.sum(), 1e-12)
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


class GBDT:
    def __init__(self, n_estimators=80, learning_rate=0.06, max_depth=3, min_samples_leaf=12, seed=0):
        self.n_estimators, self.lr, self.max_depth, self.min_samples_leaf = n_estimators, learning_rate, max_depth, min_samples_leaf
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.base = float(np.mean(y))
        pred = np.full(len(y), self.base)
        self.trees, gains = [], np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            t = SimpleCARTTree(self.max_depth, self.min_samples_leaf, None, self.rng).fit(X, y - pred)  # 拟合残差
            pred += self.lr * t.predict(X)
            self.trees.append(t)
            gains += t.feat_gains
        self.feature_importances_ = gains / max(gains.sum(), 1e-12)
        return self

    def predict(self, X):
        return self.base + self.lr * np.sum([t.predict(X) for t in self.trees], axis=0)


class TreeEnsemble:
    """M4 = 随机森林和梯度提升树的预测取平均。"""

    def __init__(self, seed=0):
        self.rf = RandomForest(seed=seed)
        self.gb = GBDT(seed=seed + 1)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.rf.predict(X) + 0.5 * self.gb.predict(X)


# ==============================================================================
# 四、指标
# ==============================================================================
def metrics(y_true, y_est):
    y_true = np.asarray(y_true, float)
    y_est = np.asarray(y_est, float)
    ape = np.abs(y_est - y_true) / y_true
    lt, le = np.log(y_true), np.log(np.maximum(y_est, 1))
    r2 = 1 - ((lt - le) ** 2).sum() / max(((lt - lt.mean()) ** 2).sum(), 1e-12)
    return {"n": len(y_true), "mape": 100 * ape.mean(), "mdape": 100 * np.median(ape), "r2_log": r2,
            "hit10": 100 * (ape <= 0.10).mean(), "hit30": 100 * (ape <= 0.30).mean(), "hit50": 100 * (ape <= 0.50).mean()}


# ==============================================================================
# 五、政策校准：政策上线后每条评测对应的销量是之前的 r 倍
#     估算销量 = k_前 × (政策前评测数 + r × 政策后评测数)
# ==============================================================================
def fit_policy_r(k_pre, n_before, n_after, y_true):
    """一维搜索 r，使对数误差平方和最小。只有政策后评测数 > 0 的样本才对 r 有信息。"""
    mask = n_after > 0
    if mask.sum() < 3:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.05), np.log(2.0), 400))
    best_r, best_loss = 1.0, np.inf
    for r in grid:
        est = k_pre[mask] * (n_before[mask] + r * n_after[mask])
        loss = float(((np.log(y_true[mask]) - np.log(est)) ** 2).sum())
        if loss < best_loss:
            best_r, best_loss = r, loss
    return float(best_r)


def loo_policy_estimates(k_pre, n_before, n_after, y_true):
    """留一法：第 i 款的 r 只用其余样本估计。返回每款的估算销量和各自用到的 r。"""
    n = len(y_true)
    est, rs = np.zeros(n), np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        r = fit_policy_r(k_pre[keep], n_before[keep], n_after[keep], y_true[keep])
        rs[i] = r
        est[i] = k_pre[i] * (n_before[i] + r * n_after[i])
    return est, rs


def recal_design(df, X):
    """
    近年校准的设计矩阵：截距 + 政策后评测占比 + 9 个核心特征（不含类型虚拟变量）。
    含义：ln k_真值 = ln k_2018模型(x) + γ0 + δ·政策后评测占比 + γ·x
      - γ0：玩家数 -> 销量的整体换算，以及年代整体差异
      - δ ：政策后评测占比越高 k 越低，全是政策后评测时相当于 k 乘 exp(δ)
      - γ ：哪些特征的作用在"2018 玩家数"和"近年销量"之间变了（设计文档 3.6 节第二种情况）
    """
    s_after = df["reviews_after_policy"].values / df["total_reviews"].values
    return np.column_stack([np.ones(len(df)), s_after, X[:, 1:10]])


def fit_recal(Z, offset, y_lk):
    w, _, _, _ = np.linalg.lstsq(Z, y_lk - offset, rcond=None)
    return w


def loo_recal(Z, offset, y_lk, groups=None):
    """
    留一法：第 i 款的校准系数只用其余 n-1 款拟合。返回 ln k 的预测。
    groups 给定时按组留出（同一款游戏的几个里程碑一起留出），否则一行一组。
    """
    n = len(y_lk)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    out = np.zeros(n)
    for i in range(n):
        keep = groups != groups[i]
        w = fit_recal(Z[keep], offset[keep], y_lk[keep])
        out[i] = offset[i] + Z[i] @ w
    return out


def nested_loo_fusion(Z, off_a, y_lk, log_n, log_b, has_b, groups=None):
    """
    嵌套留一法融合：估第 i 款时，
      1) 用其余 n-1 款拟合信号 A 的近年校准；
      2) 用这套校准算出其余 n-1 款的 ln Y_A，再在其中"有信号 B"的样本上拟合 ln Y = v0 + vA ln Y_A + vB ln Y_B；
      3) 用 1) 2) 的参数估第 i 款。
    这样第 i 款的真值在任何一步都没被用到。没有信号 B 的样本直接用校准后的信号 A。
    """
    n = len(y_lk)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    y_log = y_lk + log_n
    out, weights = np.zeros(n), []
    for i in range(n):
        keep = groups != groups[i]
        w = fit_recal(Z[keep], off_a[keep], y_lk[keep])
        ln_a_all = log_n + off_a + Z @ w
        if not has_b[i]:
            out[i] = ln_a_all[i]
            continue
        kb = keep & has_b
        F = np.column_stack([np.ones(kb.sum()), ln_a_all[kb], log_b[kb]])
        v, _, _, _ = np.linalg.lstsq(F, y_log[kb], rcond=None)
        out[i] = v[0] + v[1] * ln_a_all[i] + v[2] * log_b[i]
        weights.append(v)
    return out, np.array(weights)


# ==============================================================================
# 五点五、按可信度加权的融合（2026-10-20）
#   每路信号对每款游戏给出：ln 估算值 + 这次大概会错多少（σ）。σ 由这路信号在其他真值游戏上的
#   "诚实"误差（本身也是留一法算的）拟合，随少数几个事先看得到的质量指标变化。
#   融合 = 按 1/σ² 对"可用的信号"加权平均：
#     - 哪路信号算不出来就不参与，只剩 M4 时结果就是 M4；
#     - 结果永远夹在可用信号之间；
#     - 以后加新信号，只要给出"估算值 + 质量指标"，融合公式不用改。
#   旧做法是 ln Y = v0 + vA ln Y_A + vB ln Y_B 的固定权重，一路离谱时会把另一路一起拖偏。
# ==============================================================================
SIGNALS = [
    # (代码, 名称, 类型, 质量指标说明)
    ("A", "M4 评测数推算（近年校准后）", "estimate", "ln 评测数：评测越多越准"),
    ("B", "信号 B 在线人数 × 时长", "estimate", "ln 上架月数：SteamCharts 记录的月份越多越准"),
]
FUSION_VARIANTS = {
    "m4": "只用 M4（不融合）",
    "old": "旧：固定权重回归 ln Y = v0 + vA ln A + vB ln B",
    "geo": "A、B 各占一半（几何平均）",
    "const": "按可信度加权，σ 为常数（每路信号一个平均误差）",
    "quality": "按可信度加权，σ 随质量指标变化",
}


def quality_design(df):
    """各信号的质量指标设计矩阵（含常数列）。只用截止日当时就能看到的量。"""
    n = len(df)
    qa = np.column_stack([np.ones(n), np.log(df["total_reviews"].values.astype(float))])
    months = np.clip(df["game_age_years"].values.astype(float) * 12.0, 1.0, None)
    qb = np.column_stack([np.ones(n), np.log(months)])
    return qa, qb


def fit_error_model(resid, Q):
    """
    误差模型：resid = ln(真值/估算)。返回 (mu, gamma)：mu 是这路信号的平均偏差，
    ln σ² = Q @ gamma，用 ln((resid - mu)² + 0.01) 对 Q 回归（加 0.01 防止残差恰好为 0 时取对数爆掉）。
    对数残差平方的期望比 ln σ² 低一个常数，但各路信号低的是同一个常数，算权重时会约掉。
    """
    mu = float(np.mean(resid))
    gamma, _, _, _ = np.linalg.lstsq(Q, np.log((resid - mu) ** 2 + 0.01), rcond=None)
    return mu, gamma


def fusion_fit(Z, off_a, y_lk, log_n, ln_b, has_b, qa, qb, groups, variant):
    """用给定的真值行拟合融合需要的全部参数：M4 的近年校准 + 各信号的误差模型（或旧做法的回归权重）。"""
    params = {"variant": variant, "w_recal": fit_recal(Z, off_a, y_lk)}
    if variant in ("m4", "geo"):
        return params
    y_log = y_lk + log_n
    if variant == "old":
        ln_a_in = log_n + off_a + Z @ params["w_recal"]
        F = np.column_stack([np.ones(has_b.sum()), ln_a_in[has_b], ln_b[has_b]])
        params["v"], _, _, _ = np.linalg.lstsq(F, y_log[has_b], rcond=None)
        return params
    # M4 在这批行上的"诚实"估算：按游戏留一法，避免用自己的真值校准自己，让 M4 看起来比实际更准
    ln_a_honest = log_n + loo_recal(Z, off_a, y_lk, groups)
    cols = slice(None) if variant == "quality" else slice(0, 1)
    params["cols"] = cols
    params["A"] = fit_error_model(y_log - ln_a_honest, qa[:, cols])
    params["B"] = fit_error_model((y_log - ln_b)[has_b], qb[has_b][:, cols])
    return params


def fusion_apply(params, Z, off_a, log_n, ln_b, has_b, qa, qb):
    """返回 (ln 估算, 明细)。明细里有每路信号的 σ 和信号 B 的权重占比，写进 fact_signal_estimate。"""
    ln_a = log_n + off_a + Z @ params["w_recal"]
    n = len(ln_a)
    det = {"ln_a": ln_a, "sigma_a": np.full(n, np.nan), "sigma_b": np.full(n, np.nan),
           "share_b": np.where(has_b, 0.5, 0.0)}
    v = params["variant"]
    if v == "m4":
        det["share_b"] = np.zeros(n)
        return ln_a, det
    if v == "geo":
        return np.where(has_b, 0.5 * ln_a + 0.5 * ln_b, ln_a), det
    if v == "old":
        c = params["v"]
        return np.where(has_b, c[0] + c[1] * ln_a + c[2] * ln_b, ln_a), det
    cols = params["cols"]
    (mu_a, g_a), (mu_b, g_b) = params["A"], params["B"]
    var_a, var_b = np.exp(qa[:, cols] @ g_a), np.exp(qb[:, cols] @ g_b)
    w_a, w_b = 1.0 / var_a, np.where(has_b, 1.0 / var_b, 0.0)
    # A 已经过近年校准（截距就是它的偏差修正），不再加 mu_a，这样只剩 A 时结果严格等于 M4；
    # B 没在真值上校准过，加上它的平均偏差 mu_b
    fused = (w_a * ln_a + w_b * (ln_b + mu_b)) / (w_a + w_b)
    det.update(sigma_a=np.sqrt(var_a), sigma_b=np.where(has_b, np.sqrt(var_b), np.nan), share_b=w_b / (w_a + w_b))
    return fused, det


def fusion_predict(train, test, Z, off_a, y_lk, log_n, ln_b, has_b, qa, qb, groups, variant):
    """用 train 行拟合、估 test 行（两者是布尔掩码）。"""
    p = fusion_fit(Z[train], off_a[train], y_lk[train], log_n[train], ln_b[train], has_b[train],
                   qa[train], qb[train], groups[train], variant)
    return fusion_apply(p, Z[test], off_a[test], log_n[test], ln_b[test], has_b[test], qa[test], qb[test])


def logo_fusion(Z, off_a, y_lk, log_n, ln_b, has_b, qa, qb, groups, variant):
    """按游戏留一法：估第 g 款游戏时，融合需要的所有参数都只用其余游戏拟合（嵌套：误差模型用的残差本身也是留一法的）。"""
    n = len(y_lk)
    out = np.zeros(n)
    det = {k: np.full(n, np.nan) for k in ("sigma_a", "sigma_b", "share_b")}
    for g in np.unique(groups):
        test = groups == g
        ln, d = fusion_predict(~test, test, Z, off_a, y_lk, log_n, ln_b, has_b, qa, qb, groups, variant)
        out[test] = ln
        for k in det:
            det[k][test] = d[k]
    return out, det


# ==============================================================================
# 六、主流程
# ==============================================================================
def m1_k(year):
    if year <= 2016:
        return 74.0
    if year <= 2018:
        return 60.0
    if year == 2019:
        return 51.0
    return 38.0


def kfold_indices(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def run_all_models():
    t_start = time.time()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    df = pd.read_sql_query("SELECT * FROM dwd_game_sample ORDER BY sample_id", conn)
    tr = df[(df["cohort"] == "train_2018") & (df["model_eligible"] == 1)].reset_index(drop=True)
    te = df[df["cohort"].isin(["truth", "truth_v2"]) & (df["model_eligible"] == 1)].reset_index(drop=True)
    groups_te = te["appid"].values  # 同一款游戏的几个里程碑一起留出
    print(f"[建模] 2018 训练样本 {len(tr)}；真值样本 {len(te)} 条 / {te['appid'].nunique()} 款"
          f"（旧 {int((te.cohort == 'truth').sum())} 条，新 {int((te.cohort == 'truth_v2').sum())} 条）", flush=True)

    pt_fill = float(tr["playtime_review_median_hrs"].median())
    fills = fit_fills(tr)
    y_tr = tr["log_k"].values

    # ---------------- 特征集：base（原有 19 列）还是 ext（加语言占比、价格历史、慈善包、区域价） ----------------
    # 只按 2018 五折交叉验证的 R² 选，不看任何真值集，避免"在测试集上挑模型"
    folds = kfold_indices(len(tr), 5, RNG_SEED)

    def r2(pred):
        return float(1 - ((y_tr - pred) ** 2).sum() / ((y_tr - y_tr.mean()) ** 2).sum())

    cv, oof_sets = {}, {}
    for fs in ("base", "ext"):
        X_fs, _ = build_features(tr, pt_fill, fs, fills)
        o2, o4 = np.zeros(len(tr)), np.zeros(len(tr))
        for fi, val in enumerate(folds):
            trn = np.setdiff1d(np.arange(len(tr)), val)
            o2[val] = X_fs[val] @ fit_ols(X_fs[trn], y_tr[trn])["beta"]
            o4[val] = TreeEnsemble(seed=fi).fit(X_fs[trn][:, 1:], y_tr[trn]).predict(X_fs[val][:, 1:])
        cv[fs], oof_sets[fs] = (r2(o2), r2(o4)), (o2, o4)
        print(f"[特征集 {fs}] {X_fs.shape[1]} 列，2018 五折 ln k 的 R²：M2 {cv[fs][0]:.3f}，M4 {cv[fs][1]:.3f}", flush=True)
    feature_set = "ext" if cv["ext"][1] > cv["base"][1] else "base"
    print(f"[特征集] 选用 {feature_set}", flush=True)

    X_tr, feat_names = build_features(tr, pt_fill, feature_set, fills)
    X_te, _ = build_features(te, pt_fill, feature_set, fills)

    # ---------------- 2018 上 5 折交叉验证 ----------------
    oof = {m: np.zeros(len(tr)) for m in ["M3", "M3_lo", "M3_hi"]}
    oof["M2"], oof["M4"] = oof_sets[feature_set]
    for fi, val in enumerate(folds):
        trn = np.setdiff1d(np.arange(len(tr)), val)
        for tau, key in [(0.5, "M3"), (0.1, "M3_lo"), (0.9, "M3_hi")]:
            oof[key][val] = X_tr[val] @ fit_quantile(X_tr[trn], y_tr[trn], tau)
        print(f"[建模] M3 第 {fi + 1}/5 折完成", flush=True)

    # ---------------- 全量 2018 拟合（给真值集用） ----------------
    ols = fit_ols(X_tr, y_tr)
    q = {tau: fit_quantile(X_tr, y_tr, tau) for tau in (0.1, 0.5, 0.9)}
    tree = TreeEnsemble(seed=99).fit(X_tr[:, 1:], y_tr)
    k_pre = {
        "M2": np.exp(X_te @ ols["beta"]),
        "M3": np.exp(X_te @ q[0.5]), "M3_lo": np.exp(X_te @ q[0.1]), "M3_hi": np.exp(X_te @ q[0.9]),
        "M4": np.exp(tree.predict(X_te[:, 1:])),
    }

    # ---------------- 信号 B：总游玩小时 ÷ 人均小时 ----------------
    # 人均小时 = exp(a + b·ln 评测者时长)，a、b 用 2018 年"总小时 ÷ 真实玩家数"拟合；2018 上也做 5 折
    trb = tr[(tr["charts_coverage"] == 1) & tr["playtime_review_median_hrs"].notna()].reset_index(drop=True)
    lh = np.log(trb["charts_total_hours"].values / trb["truth_value"].values)
    lp = np.log(trb["playtime_review_median_hrs"].clip(lower=0.1).values)
    Zb = np.column_stack([np.ones(len(trb)), lp])
    ab, _, _, _ = np.linalg.lstsq(Zb, lh, rcond=None)
    oof_b = np.zeros(len(trb))
    for val in kfold_indices(len(trb), 5, RNG_SEED + 1):
        trn = np.setdiff1d(np.arange(len(trb)), val)
        w, _, _, _ = np.linalg.lstsq(Zb[trn], lh[trn], rcond=None)
        oof_b[val] = trb["charts_total_hours"].values[val] / np.exp(Zb[val] @ w)
    oof_b_map = dict(zip(trb["sample_id"], oof_b))
    print(f"[信号B] ln(人均小时) = {ab[0]:.3f} + {ab[1]:.3f}·ln(评测者时长)，2018 可用样本 {len(trb)}", flush=True)

    has_b_te = ((te["charts_coverage"] == 1) & te["playtime_review_median_hrs"].notna()).values
    y_b_te = np.full(len(te), np.nan)
    pt_te = te["playtime_review_median_hrs"].clip(lower=0.1).values
    y_b_te[has_b_te] = te["charts_total_hours"].values[has_b_te] / np.exp(ab[0] + ab[1] * np.log(pt_te[has_b_te]))

    # ---------------- 真值集：先看"直接迁移"，再做近年校准（都用留一法） ----------------
    y_true_te = te["truth_value"].values.astype(float)
    nb = te["reviews_before_policy"].values.astype(float)
    na = te["reviews_after_policy"].values.astype(float)
    pol = json.loads((BASE_DIR / "data" / "raw" / "policy_change_detected.json").read_text(encoding="utf-8"))
    r_scan = 1.0 / pol["jump_ratio"]

    # (1) 直接迁移：2018 模型 + 一个政策系数 r（设计文档 3.4 节的拆分公式），r 有四种取法
    transfer = {
        "M2_no_calib": k_pre["M2"] * (nb + na),
        "M2_r_lit": k_pre["M2"] * (nb + 0.5 * na),
        "M2_r_scan": k_pre["M2"] * (nb + r_scan * na),
    }
    transfer["M2_r_loo"], r_used = loo_policy_estimates(k_pre["M2"], nb, na, y_true_te)
    r_full = fit_policy_r(k_pre["M2"], nb, na, y_true_te)

    # (2) 近年校准：残差检验发现比值随特征变化（见 ads_policy_calibration），按 3.6 节用近年真值重估系数
    y_lk_te = np.log(y_true_te / te["total_reviews"].values)
    log_n_te = np.log(te["total_reviews"].values)
    Z = recal_design(te, X_te)
    est_te = {}
    for m in ["M2", "M4"]:
        est_te[m] = np.exp(log_n_te + loo_recal(Z, np.log(k_pre[m]), y_lk_te, groups_te))
    # M3：中位数沿用 M2 的校准；区间宽度取 2018 分位数回归的 10%/90% 与 50% 之差
    est_te["M3"] = est_te["M2"]
    est_te["M3_lo"] = est_te["M2"] * k_pre["M3_lo"] / k_pre["M3"]
    est_te["M3_hi"] = est_te["M2"] * k_pre["M3_hi"] / k_pre["M3"]
    w_recal_full = fit_recal(Z, np.log(k_pre["M2"]), y_lk_te)

    # ---------------- 融合（嵌套留一法） ----------------
    log_b = np.log(np.where(has_b_te, y_b_te, 1.0))
    w_m4_full = fit_recal(Z, np.log(k_pre["M4"]), y_lk_te)
    ln_a_full = log_n_te + np.log(k_pre["M4"]) + Z @ w_m4_full
    Ff = np.column_stack([np.ones(has_b_te.sum()), ln_a_full[has_b_te], log_b[has_b_te]])
    w_full, _, _, _ = np.linalg.lstsq(Ff, np.log(y_true_te[has_b_te]), rcond=None)
    print(f"[直接迁移] 政策系数 r：留一法全样本 {r_full:.3f}，评测量跳升倒数 {r_scan:.3f}，文献 0.5", flush=True)
    print(f"[近年校准] M2 校准中政策后评测占比的系数 δ={w_recal_full[1]:.3f}（相当于 k × {math.exp(w_recal_full[1]):.3f}）", flush=True)
    print(f"[融合·旧做法] 全样本权重 ln Y = {w_full[0]:.3f} + {w_full[1]:.3f} ln Y_A + {w_full[2]:.3f} ln Y_B", flush=True)

    # ---------------- 外部检验：只用旧真值校准，估新真值集（两批游戏没有重复，不需要留一法） ----------------
    ext_val = {}
    old, new = (te["cohort"] == "truth").values, (te["cohort"] == "truth_v2").values
    for fs in ("base", "ext"):
        if fs == feature_set:
            Xa, Xe, ols_b, tree_fs = X_tr, X_te, ols["beta"], tree
        else:
            Xa, _ = build_features(tr, pt_fill, fs, fills)
            Xe, _ = build_features(te, pt_fill, fs, fills)
            ols_b, tree_fs = fit_ols(Xa, y_tr)["beta"], TreeEnsemble(seed=99).fit(Xa[:, 1:], y_tr)
        Ze = recal_design(te, Xe)
        off2, off4 = Xe @ ols_b, tree_fs.predict(Xe[:, 1:])
        w2 = fit_recal(Ze[old], off2[old], y_lk_te[old])
        w4 = fit_recal(Ze[old], off4[old], y_lk_te[old])
        ln_a = log_n_te + off4 + Ze @ w4
        hb = has_b_te & old
        v, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(hb.sum()), ln_a[hb], log_b[hb]]),
                                     np.log(y_true_te[hb]), rcond=None)
        fused = np.where(has_b_te, v[0] + v[1] * ln_a + v[2] * log_b, ln_a)
        ext_val[fs] = {"M0": te["total_reviews"].values * 30.0,
                       "M2": np.exp(log_n_te + off2 + Ze @ w2), "M4": np.exp(ln_a), "Fusion_AB": np.exp(fused)}
    for fs, e in ext_val.items():
        le = np.log(e["Fusion_AB"][new] / y_true_te[new])
        print(f"[外部检验 {fs}] 新真值集 {new.sum()} 条：融合 ±30% 命中 {100 * np.mean(np.abs(np.exp(le) - 1) <= 0.3):.1f}%，"
              f"中位 |ln 误差| {np.median(np.abs(le)):.3f}", flush=True)

    # ---------------- 融合方式：五种做法对比，按事先定好的规则选一种 ----------------
    # 规则（2026-10-20 定，先定规则再看结果）：在外部检验（旧真值拟合 -> 估新真值集）上比平均 |ln 误差|，
    # 取最小的；如果最好的那种还不如只用 M4，就只用 M4。再用新旧合并的按游戏留一法报告选中方案的成绩。
    off_a = np.log(k_pre["M4"])
    qa_te, qb_te = quality_design(te)
    fus_args = (Z, off_a, y_lk_te, log_n_te, log_b, has_b_te, qa_te, qb_te, groups_te)
    fus_ext, fus_logo, fus_det = {}, {}, {}
    for v in FUSION_VARIANTS:
        ln_new, _ = fusion_predict(old, new, *fus_args, v)
        fus_ext[v] = float(np.mean(np.abs(ln_new - np.log(y_true_te[new]))))
        fus_logo[v], fus_det[v] = logo_fusion(*fus_args, v)
    fusion_variant = min(["old", "geo", "const", "quality"], key=lambda v: fus_ext[v])
    if fus_ext[fusion_variant] >= fus_ext["m4"]:
        fusion_variant = "m4"
    for v, name in FUSION_VARIANTS.items():
        le = fus_logo[v] - np.log(y_true_te)
        print(f"[融合方式] {name}：外部检验平均 |ln 误差| {fus_ext[v]:.3f}；合并真值留一法 {np.mean(np.abs(le)):.3f}，"
              f"±30% 命中 {100 * np.mean(np.abs(np.exp(le) - 1) <= 0.3):.1f}%" + ("  <- 选用" if v == fusion_variant else ""),
              flush=True)
    est_te["Fusion_AB"] = np.exp(fus_logo[fusion_variant])
    fus_full = fusion_fit(*fus_args, fusion_variant)

    # ---------------- 写事实表 ----------------
    game_key = dict(conn.execute("SELECT appid, game_key FROM dim_game").fetchall())
    genre_id = dict(conn.execute("SELECT genre_name, genre_id FROM dim_genre").fetchall())
    # 区间：用 2018 交叉验证残差的 10%/90% 分位（M3 自带分位数区间）
    resid_q = {}
    for m in ["M2", "M4"]:
        r_ = y_tr - oof[m]
        resid_q[m] = (float(np.exp(np.percentile(r_, 10))), float(np.exp(np.percentile(r_, 90))))
    resid_q["M0"] = resid_q["M1"] = resid_q["M2"]
    rb = np.log(trb["truth_value"].values / oof_b)
    resid_q["Signal_B"] = (float(np.exp(np.percentile(rb, 10))), float(np.exp(np.percentile(rb, 90))))
    resid_q["Fusion_AB"] = resid_q["M4"]

    fact_rows = []

    def add_fact(row, method_id, code, est, lo, hi):
        est = max(1, int(round(est)))
        lo = max(1, int(round(min(lo, est))))
        hi = max(est, int(round(hi)))
        truth = int(row["truth_value"])
        err = (est - truth) / truth
        fact_rows.append((
            int(row["sample_id"]), int(row["appid"]), game_key[int(row["appid"])], genre_id.get(row["genre"], 10),
            int(row["release_year"]), method_id, code, row["cohort"], row["era_group"], int(row["total_reviews"]),
            float(row["price_usd"]), truth, est, lo, hi, round(est * float(row["price_usd"]), 2),
            round(est / row["total_reviews"], 3), round(float(row["k_ratio"]), 3), round(err, 5), round(abs(err), 5),
            int(abs(err) <= 0.30), int(abs(err) <= 0.50)))

    for i, row in tr.iterrows():
        n = row["total_reviews"]
        e0, e1 = n * 30.0, n * m1_k(row["release_year"])
        add_fact(row, 1, "M0", e0, e0 * resid_q["M0"][0], e0 * resid_q["M0"][1])
        add_fact(row, 2, "M1", e1, e1 * resid_q["M1"][0], e1 * resid_q["M1"][1])
        e2 = n * math.exp(oof["M2"][i])
        add_fact(row, 3, "M2", e2, e2 * resid_q["M2"][0], e2 * resid_q["M2"][1])
        add_fact(row, 4, "M3", n * math.exp(oof["M3"][i]), n * math.exp(oof["M3_lo"][i]), n * math.exp(oof["M3_hi"][i]))
        e4 = n * math.exp(oof["M4"][i])
        add_fact(row, 5, "M4", e4, e4 * resid_q["M4"][0], e4 * resid_q["M4"][1])
        if row["sample_id"] in oof_b_map:
            eb = oof_b_map[row["sample_id"]]
            add_fact(row, 6, "Signal_B", eb, eb * resid_q["Signal_B"][0], eb * resid_q["Signal_B"][1])
    for i, row in te.iterrows():
        n = row["total_reviews"]
        e0, e1 = n * 30.0, n * m1_k(row["release_year"])
        add_fact(row, 1, "M0", e0, e0 * resid_q["M0"][0], e0 * resid_q["M0"][1])
        add_fact(row, 2, "M1", e1, e1 * resid_q["M1"][0], e1 * resid_q["M1"][1])
        add_fact(row, 3, "M2", est_te["M2"][i], est_te["M2"][i] * resid_q["M2"][0], est_te["M2"][i] * resid_q["M2"][1])
        add_fact(row, 4, "M3", est_te["M3"][i], est_te["M3_lo"][i], est_te["M3_hi"][i])
        add_fact(row, 5, "M4", est_te["M4"][i], est_te["M4"][i] * resid_q["M4"][0], est_te["M4"][i] * resid_q["M4"][1])
        if has_b_te[i]:
            eb = y_b_te[i]
            add_fact(row, 6, "Signal_B", eb, eb * resid_q["Signal_B"][0], eb * resid_q["Signal_B"][1])
        ef = est_te["Fusion_AB"][i]
        add_fact(row, 7, "Fusion_AB", ef, ef * resid_q["Fusion_AB"][0], ef * resid_q["Fusion_AB"][1])

    with conn:
        conn.execute("DELETE FROM fact_sales_estimation;")
        conn.executemany(
            """INSERT INTO fact_sales_estimation (sample_id, appid, game_key, genre_id, date_key, method_id, method_code,
               cohort, era_group, total_reviews, price_usd, true_value, estimated_sales, estimated_low_bound,
               estimated_high_bound, estimated_revenue_usd, implied_k, observed_k, pct_error, abs_pct_error,
               within_30pct_flag, within_50pct_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            fact_rows)

    # ---------------- 各路信号的估算与权重（融合是怎么做决定的） ----------------
    d = fus_det[fusion_variant]
    sig_rows = []
    for i, row in te.iterrows():
        sid = int(row["sample_id"])
        share_b = float(d["share_b"][i]) if not np.isnan(d["share_b"][i]) else 0.0
        sig_rows.append((sid, "A", 1, round(float(np.log(est_te["M4"][i])), 6),
                         None if np.isnan(d["sigma_a"][i]) else round(float(d["sigma_a"][i]), 4), round(1 - share_b, 4)))
        sig_rows.append((sid, "B", int(has_b_te[i]), round(float(log_b[i]), 6) if has_b_te[i] else None,
                         None if np.isnan(d["sigma_b"][i]) else round(float(d["sigma_b"][i]), 4),
                         round(share_b, 4) if has_b_te[i] else 0.0))
    with conn:
        conn.execute("DELETE FROM fact_signal_estimate;")
        conn.execute("DELETE FROM dim_signal;")
        conn.executemany("INSERT INTO dim_signal (signal_code, signal_name, signal_kind, quality_feature) VALUES (?, ?, ?, ?)",
                         SIGNALS)
        conn.executemany("""INSERT INTO fact_signal_estimate (sample_id, signal_code, available, ln_estimate, sigma,
                            weight_share) VALUES (?, ?, ?, ?, ?, ?)""", sig_rows)

    # ---------------- 评估汇总 ----------------
    method_name = dict(conn.execute("SELECT method_code, method_name FROM dim_estimation_method").fetchall())
    fact = pd.read_sql_query("SELECT * FROM fact_sales_estimation", conn)
    eval_rows = []

    def add_eval(cohort_label, code, name, y, e, note):
        if len(y) == 0:
            return
        m = metrics(y, e)
        eval_rows.append((cohort_label, code, name, m["n"], round(m["mape"], 2), round(m["mdape"], 2),
                          round(m["r2_log"], 4), round(m["hit10"], 2), round(m["hit30"], 2), round(m["hit50"], 2), note))

    is_truth = fact["cohort"].isin(["truth", "truth_v2"])
    groups = [
        ("2018 五折交叉验证", fact["cohort"] == "train_2018", "袋外预测；真值是泄露玩家数"),
        ("合并真值集 全部", is_truth, "旧 Gamalytic 表 + 新收集的 Steam 口径销量；校准与融合按游戏留一法"),
        ("新真值集 全部", fact["cohort"] == "truth_v2", "2026-10-20 新收集，Steam 口径（S1+S2+Q 分层）"),
        ("合并真值集 政策前公布", is_truth & (fact["era_group"] == "pre_policy"), ""),
        ("合并真值集 跨时代", is_truth & (fact["era_group"] == "cross_era"), "政策前发售、政策后公布"),
        ("合并真值集 政策后发售", is_truth & (fact["era_group"] == "post_policy"), ""),
        ("真值集 全部", fact["cohort"] == "truth", "真值是公开销量；政策系数与融合权重为留一法"),
        ("真值集 政策前公布", (fact["cohort"] == "truth") & (fact["era_group"] == "pre_policy"), ""),
        ("真值集 跨时代", (fact["cohort"] == "truth") & (fact["era_group"] == "cross_era"), "政策前发售、政策后公布"),
        ("真值集 政策后发售", (fact["cohort"] == "truth") & (fact["era_group"] == "post_policy"), ""),
    ]
    for label, mask, note in groups:
        sub = fact[mask]
        for code in ["M0", "M1", "M2", "M3", "M4", "Signal_B", "Fusion_AB"]:
            s = sub[sub["method_code"] == code]
            extra = "（只含有 SteamCharts 完整覆盖的样本）" if code == "Signal_B" else ""
            add_eval(label, code, method_name[code], s["true_value"].values, s["estimated_sales"].values, note + extra)

    # 公平比较：只看"信号 B 可用"的那部分真值样本，A、B、融合三者同一批游戏
    b_ids = set(fact[is_truth & (fact["method_code"] == "Signal_B")]["sample_id"])
    sub = fact[is_truth & fact["sample_id"].isin(b_ids)]
    for code in ["M0", "M2", "M4", "Signal_B", "Fusion_AB"]:
        s = sub[sub["method_code"] == code]
        add_eval("合并真值集 信号B可用子集", code, method_name[code], s["true_value"].values, s["estimated_sales"].values,
                 "同一批游戏上比较 A / B / 融合")

    # 融合方式对比（外部检验 + 合并真值按游戏留一法）；平均 |ln 误差| 写在备注里，命中率等照常
    for v, name in FUSION_VARIANTS.items():
        le = fus_logo[v] - np.log(y_true_te)
        add_eval("融合方式对比（合并真值，按游戏留一法）", "F_" + v, name, y_true_te, np.exp(fus_logo[v]),
                 f"平均 |ln 误差| {np.mean(np.abs(le)):.3f}；外部检验 {fus_ext[v]:.3f}" + ("；选用" if v == fusion_variant else ""))

    # 外部检验：只用旧真值校准，估新真值集（base 与 ext 两套特征都报）
    for fs, e in ext_val.items():
        for code in ["M0", "M2", "M4", "Fusion_AB"]:
            add_eval(f"外部检验 {fs}（旧真值校准→新真值集）", code, method_name[code], y_true_te[new], e[code][new],
                     f"特征集 {fs}；新真值集与旧真值集没有重复的游戏" + ("（本次选用）" if fs == feature_set else ""))

    # 直接迁移（只调政策系数 r）与近年校准的对比
    for code, note in [("M2_no_calib", "直接迁移，不做政策校准 r=1"),
                       ("M2_r_lit", "直接迁移，文献值 r=0.5，不用真值"),
                       ("M2_r_scan", f"直接迁移，评测量跳升倒数 r={r_scan:.3f}，不用真值"),
                       ("M2_r_loo", "直接迁移，r 用留一法拟合")]:
        add_eval("合并真值集 迁移方式对比", code, "M2 " + note, y_true_te, transfer[code], note)
    add_eval("合并真值集 迁移方式对比", "M2", "M2 近年校准（全部系数按游戏留一法重估）", y_true_te, est_te["M2"], "即主表中的 M2")

    # Gamalytic 在同一批游戏上的估算（来自同一张表的 Estimate 列）
    truth_ods = pd.read_sql_query("SELECT truth_id, appid, report_date, gamalytic_estimate FROM ods_recent_sales_truth", conn)
    g = te.merge(truth_ods, left_on=["appid", "cutoff_date"], right_on=["appid", "report_date"], how="left")
    g = g[g["gamalytic_estimate"].notna()]
    add_eval("真值集 全部", "Gamalytic_same", "Gamalytic 自己的估算(同一批游戏)", g["truth_value"].values,
             g["gamalytic_estimate"].values, "来自 Gamalytic 测试表的 Estimate 列，作同口径对照")
    for code, name, hit in GAMALYTIC_PUBLISHED:
        eval_rows.append(("Gamalytic 公布(约120款)", code, name, 120, None, None, None, None, hit, None,
                          "原文只给了 ±30% 命中率"))

    with conn:
        conn.execute("DELETE FROM ads_model_eval_summary;")
        conn.executemany(
            """INSERT INTO ads_model_eval_summary (eval_cohort, method_code, method_name, sample_size, mape_pct, mdape_pct,
               r2_log, hit_rate_10pct, hit_rate_30pct, hit_rate_50pct, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            eval_rows)

    # ---------------- 特征重要性与回归系数 ----------------
    feat_rows = []
    for j, (code, cn) in enumerate(feat_names):
        rf_imp = float(tree.rf.feature_importances_[j - 1] * 100) if j > 0 else None
        gb_imp = float(tree.gb.feature_importances_[j - 1] * 100) if j > 0 else None
        feat_rows.append((code, cn, round(float(ols["beta"][j]), 5), round(float(ols["se"][j]), 5),
                          round(float(ols["t"][j]), 3), float(ols["p"][j]),
                          None if rf_imp is None else round(rf_imp, 2), None if gb_imp is None else round(gb_imp, 2)))
    with conn:
        conn.execute("DELETE FROM ads_feature_importance;")
        conn.executemany(
            """INSERT INTO ads_feature_importance (feature_code, feature_cn_name, m2_ols_coef, m2_std_err, m2_t_stat,
               m2_p_value, m4_rf_importance_pct, m4_gbdt_importance_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", feat_rows)

    # ---------------- 政策校准与系数稳定性检验（设计文档 3.6 节） ----------------
    # 用 2018 模型 + 留一法 r 估算真值样本后，看 ln(真实/估算) 是否还和某个特征相关：
    # 不相关 -> 政策只改变了整体水平，2018 的系数可以沿用；相关 -> 该系数也变了
    resid_te = np.log(y_true_te / transfer["M2_r_loo"])  # 检验的是"直接迁移"的残差
    calib_rows = [
        ("feature_set_ext", 1.0 if feature_set == "ext" else 0.0, "1 = 选用 ext 特征集（按 2018 五折 M4 的 R² 选）"),
        ("fusion_variant", float(list(FUSION_VARIANTS).index(fusion_variant)),
         "融合方式编号：" + "，".join(f"{i}={k}" for i, k in enumerate(FUSION_VARIANTS)) + f"；选用 {fusion_variant}"),
    ] + [(f"fusion_ext_mean_abs_ln_{v}", fus_ext[v], f"外部检验平均 |ln 误差|：{FUSION_VARIANTS[v]}") for v in FUSION_VARIANTS] + [
        ("cv_r2_m2_base", cv["base"][0], "2018 五折 ln k 的 R²，base 特征集"),
        ("cv_r2_m4_base", cv["base"][1], ""),
        ("cv_r2_m2_ext", cv["ext"][0], "2018 五折 ln k 的 R²，ext 特征集"),
        ("cv_r2_m4_ext", cv["ext"][1], ""),
        ("truth_rows_used", float(len(te)), "参与校准的真值样本条数（旧 + 新）"),
        ("truth_games_used", float(te["appid"].nunique()), "参与校准的真值游戏数"),
        ("transfer_r_loo_full", r_full, "直接迁移时政策系数 r 的全样本拟合值"),
        ("transfer_r_loo_min", float(r_used.min()), "留一法各次拟合 r 的最小值"),
        ("transfer_r_loo_max", float(r_used.max()), "留一法各次拟合 r 的最大值"),
        ("r_review_jump_inverse", r_scan, "40 款老游戏评测量跳升倍数的倒数"),
        ("r_literature", 0.5, "Gamalytic: 2019 年后 k 减半"),
        ("recal_gamma0", float(w_recal_full[0]), "M2 近年校准：截距（全样本拟合，仅供报告解读）"),
        ("recal_delta_after_share", float(w_recal_full[1]), "M2 近年校准：政策后评测占比的系数，exp 后即 k 的倍数"),
    ] + [(f"recal_gamma_{feat_names[j][0]}", float(w_recal_full[j + 1]), f"M2 近年校准：{feat_names[j][1]} 的系数修正")
         for j in range(1, 10)] + [
        ("signalB_a", float(ab[0]), "ln(人均小时) = a + b·ln(评测者时长) 中的 a"),
        ("signalB_b", float(ab[1]), "同上的 b"),
        ("fusion_w0", float(w_full[0]), "全样本融合权重（报告用）"),
        ("fusion_wA", float(w_full[1]), ""),
        ("fusion_wB", float(w_full[2]), ""),
        ("m2_cv_r2_2018", float(1 - ((y_tr - oof["M2"]) ** 2).sum() / ((y_tr - y_tr.mean()) ** 2).sum()), "2018 五折 ln k 的 R²"),
        ("m4_cv_r2_2018", float(1 - ((y_tr - oof["M4"]) ** 2).sum() / ((y_tr - y_tr.mean()) ** 2).sum()), ""),
        ("m3_interval_coverage_2018", float(np.mean((y_tr >= oof["M3_lo"]) & (y_tr <= oof["M3_hi"]))), "名义 80%"),
        ("m3_interval_coverage_truth", float(np.mean((y_true_te >= est_te["M3_lo"]) & (y_true_te <= est_te["M3_hi"]))), "名义 80%"),
    ]
    if fusion_variant in ("const", "quality"):
        for sig in ("A", "B"):
            mu, g = fus_full[sig]
            calib_rows.append((f"fusion_{sig}_bias", mu, f"信号 {sig} 的平均偏差 ln(真值/估算)（全样本拟合，报告用）"))
            for j, gj in enumerate(g):
                calib_rows.append((f"fusion_{sig}_lnvar_coef{j}", float(gj),
                                   f"信号 {sig} 的 ln σ² 系数（0 = 常数项，1 = 质量指标；全样本拟合，报告用）"))
    for j, (code, cn) in enumerate(feat_names[1:10], start=1):
        x = X_te[:, j]
        if np.std(x) > 0:
            rho, pval = stats.spearmanr(x, resid_te)
            calib_rows.append((f"resid_corr_{code}", float(rho), f"直接迁移残差与 {cn} 的 Spearman 相关，p={pval:.3f}"))
    with conn:
        conn.execute("DELETE FROM ads_policy_calibration;")
        conn.executemany("INSERT INTO ads_policy_calibration (item, value, note) VALUES (?, ?, ?)", calib_rows)

    log_msg = f"写入事实表 {len(fact_rows)} 行，评估 {len(eval_rows)} 行"
    with conn:
        conn.execute(
            """INSERT INTO etl_job_log (job_name, start_time, end_time, rows_extracted, rows_loaded, rows_rejected,
               status, duration_ms, message) VALUES ('MODEL_TRAIN_AND_LOAD', ?, ?, ?, ?, 0, 'SUCCESS', ?, ?)""",
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)), time.strftime("%Y-%m-%d %H:%M:%S"),
             len(df), len(fact_rows), int((time.time() - t_start) * 1000), log_msg))

    explain_index_comparison(conn)
    export_views(conn)
    conn.close()

    ev = pd.DataFrame(eval_rows, columns=["cohort", "method", "name", "n", "mape", "mdape", "r2", "hit10", "hit30", "hit50", "note"])
    print(ev[["cohort", "method", "n", "mape", "mdape", "r2", "hit30", "hit50"]].to_string(index=False))


# ==============================================================================
# 七、索引与 EXPLAIN 对比；视图导出
# ==============================================================================
EXPLAIN_SQL = """
SELECT dm.method_name, dge.genre_cn_name, COUNT(*) AS game_cnt,
       ROUND(AVG(f.abs_pct_error) * 100, 2) AS mean_ape_pct,
       ROUND(SUM(f.within_30pct_flag) * 100.0 / COUNT(*), 2) AS hit_rate_30pct
FROM fact_sales_estimation f
JOIN dim_game dg ON f.game_key = dg.game_key
JOIN dim_genre dge ON f.genre_id = dge.genre_id
JOIN dim_estimation_method dm ON f.method_id = dm.method_id
WHERE f.cohort = 'truth' AND f.method_code IN ('M0', 'M2', 'M4', 'Fusion_AB')
GROUP BY dm.method_name, dge.genre_cn_name
"""
INDEXES = [
    ("idx_fact_cohort_method", "fact_sales_estimation(cohort, method_code, genre_id)"),
    ("idx_fact_game_key", "fact_sales_estimation(game_key)"),
    ("idx_dwd_cohort_year", "dwd_game_sample(cohort, release_year)"),
    ("idx_scd2_appid_dates", "dim_price_scd2(appid, effective_start, effective_end)"),
    ("idx_snapshot_appid_time", "fact_snapshot(appid, fetched_at)"),
]


def explain_index_comparison(conn, repeat=300):
    with conn:
        for name, _ in INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {name};")
    plan_before = conn.execute("EXPLAIN QUERY PLAN " + EXPLAIN_SQL).fetchall()
    t0 = time.perf_counter()
    for _ in range(repeat):
        conn.execute(EXPLAIN_SQL).fetchall()
    ms_before = (time.perf_counter() - t0) * 1000
    with conn:
        for name, target in INDEXES:
            conn.execute(f"CREATE INDEX {name} ON {target};")
    plan_after = conn.execute("EXPLAIN QUERY PLAN " + EXPLAIN_SQL).fetchall()
    t0 = time.perf_counter()
    for _ in range(repeat):
        conn.execute(EXPLAIN_SQL).fetchall()
    ms_after = (time.perf_counter() - t0) * 1000
    n_fact = conn.execute("SELECT COUNT(*) FROM fact_sales_estimation").fetchone()[0]
    lines = [
        "=== 索引前后 EXPLAIN QUERY PLAN 与耗时对比（SQLite，本机实测，每次运行数字会略有不同）===",
        f"事实表行数: {n_fact}；同一条四表关联聚合查询重复执行 {repeat} 次",
        "建索引前执行计划:",
    ] + [f"    {r[-1]}" for r in plan_before] + [
        f"    耗时 {ms_before:.1f} ms",
        "建索引后执行计划:",
    ] + [f"    {r[-1]}" for r in plan_after] + [
        f"    耗时 {ms_after:.1f} ms",
        f"提速倍数: {ms_before / max(ms_after, 0.01):.2f}x",
    ]
    text = "\n".join(lines)
    (CLEANED_DIR / "explain_comparison_report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def export_views(conn):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["vw_sql_m0_m1_estimation", "vw_finebi_k_median_by_year", "vw_finebi_k_median_by_factor",
                 "vw_finebi_overview_dist", "vw_finebi_star_sales_switch", "vw_finebi_price_scd2_timeline",
                 "vw_finebi_review_policy_index", "vw_finebi_method_accuracy", "ads_feature_importance",
                 "ads_policy_calibration", "etl_job_log", "etl_checkpoint", "etl_clean_error_log", "dim_price_scd2"]:
        pd.read_sql_query(f"SELECT * FROM {name}", conn).to_csv(CLEANED_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run_all_models()
