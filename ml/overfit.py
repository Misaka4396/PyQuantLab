"""C4 过拟合检测主模块：DSR、PBO/CSCV、OOS/IS 退化对比与上线判定。

公式来源（全部可审计）：
- Probabilistic Sharpe Ratio (PSR) 与 Deflated Sharpe Ratio (DSR)：
  Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
  Journal of Portfolio Management.
- Sharpe 估计量的标准误（非正态修正）：
  Mertens (2002) / Lo (2002)：
  SE[SR] = sqrt( (1 - γ3·SR + (γ4 - 1)/4 · SR²) / (T - 1) )，
  其中 γ3 为偏度，γ4 为峰度（full kurtosis，正态分布 = 3）。
- 期望最大 Sharpe（多重检验校正基准）：
  E[max_N] ≈ SR0 + SE[SR] · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]，
  γ = 欧拉-马歇罗尼常数 ≈ 0.5772。
- PBO / CSCV（组合对称交叉验证）：
  Bailey, Borwein, López de Prado & Zhu (2015), "The Probability of Backtest
  Overfitting", Journal of Computational Finance.

口径约定：
- DSR 是概率，对 Sharpe 是否年化**不变**；但 SE 公式的分母只对**非年化（每期）
  Sharpe** 成立，故本模块一律在每期口径上计算 DSR；展示年化 Sharpe 另行换算。
- CSCV 的 "performance" 默认用各策略在 IS/OOS 行子集上的平均收益（可替换）。
- 判定阈值（上线建议用）见下方常量与 ``conclude_overfitting``。

与 A4 联动：``oosis_degradation`` / ``assess_overfitting`` 消费
``backtest.metrics_enhanced.compare_is_oos`` 的 IS/OOS 权益对比结果。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from backtest.metrics_enhanced import (
    DEFAULT_PERIODS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    compare_is_oos,
)

EULER_MASCHERONI = 0.5772156649015329

# ---------------------------------------------------------------------------
# 判定阈值（上线建议用）
# ---------------------------------------------------------------------------
DSR_SAFE = 0.95  # DSR >= 0.95：样本外显著（低风险）
DSR_HIGH_RISK = 0.80  # DSR < 0.80：高风险
PBO_LOW_RISK = 0.10  # PBO <= 0.10：低风险
PBO_HIGH_RISK = 0.30  # PBO > 0.30：高风险
DEGRADATION_SAFE = 0.80  # OOS/IS Sharpe 衰减 >= 0.8：低风险
DEGRADATION_HIGH_RISK = 0.50  # 衰减 < 0.5：高风险

RISK_LOW = "低"
RISK_MEDIUM = "中"
RISK_HIGH = "高"


# ---------------------------------------------------------------------------
# DSR / PSR 基础计算
# ---------------------------------------------------------------------------
def sharpe_standard_error(
    observed_sr: float,
    num_observations: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Sharpe 估计量的标准误（非正态修正，Mertens 2002 / Lo 2002）。

    参数 ``kurtosis`` 为 full kurtosis（正态分布 = 3；scipy 默认返回超额峰度，
    需 +3 换算）。``observed_sr`` 为非年化（每期）Sharpe。
    """
    if num_observations < 2:
        raise ValueError("num_observations 至少为 2")
    var = (1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2) / (
        num_observations - 1.0
    )
    return float(np.sqrt(max(var, 0.0)))


def skewness_kurtosis(returns) -> tuple[float, float, float]:
    """返回 (偏度 γ3, 超额峰度, full 峰度 γ4)。"""
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 3:
        raise ValueError("至少需要 3 个观测计算偏度/峰度")
    skew = float(stats.skew(r))
    excess = float(stats.kurtosis(r))  # fisher=True，超额峰度（正态=0）
    return skew, excess, excess + 3.0


def probabilistic_sharpe_ratio(
    observed_sr: float,
    num_observations: int,
    skew: float,
    kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """PSR = Φ( (SR - SR*) / SE[SR] )，SR* 为基准 Sharpe（多重检验 N=1 时即 DSR）。"""
    se = sharpe_standard_error(observed_sr, num_observations, skew, kurtosis)
    if se == 0.0:
        return 1.0 if observed_sr > benchmark_sr else (0.0 if observed_sr < benchmark_sr else 0.5)
    return float(stats.norm.cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe_ratio(
    trials: int,
    benchmark_sr: float,
    sharpe_se: float,
) -> float:
    """N 次独立试验下期望最大 Sharpe（多重检验校正基准）。

    E[max_N] ≈ SR0 + SE[SR] · [ (1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e)) ]。
    N < 2 时退化为基准 SR0（无多重检验校正）。
    """
    if trials < 2:
        return float(benchmark_sr)
    inv1 = stats.norm.ppf(1.0 - 1.0 / trials)
    inv2 = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    z = (1.0 - EULER_MASCHERONI) * inv1 + EULER_MASCHERONI * inv2
    return float(benchmark_sr + sharpe_se * z)


def deflated_sharpe_ratio(
    observed_sr: float,
    num_observations: int,
    skew: float,
    kurtosis: float,
    trials: int = 1,
    benchmark_sr: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio（多重检验校正后的 PSR）。

    参数：
    - observed_sr：已观测的非年化（每期）Sharpe。
    - num_observations：观测数 T。
    - skew / kurtosis：偏度 γ3 / 峰度 γ4（full，正态=3）。
    - trials：试验次数 N（>=1）。N=1 退化为普通 PSR。
    - benchmark_sr：基准 Sharpe SR0（每期口径，默认 0）。
    """
    if trials <= 1:
        return probabilistic_sharpe_ratio(
            observed_sr, num_observations, skew, kurtosis, benchmark_sr
        )
    se = sharpe_standard_error(observed_sr, num_observations, skew, kurtosis)
    if se == 0.0:
        return 1.0 if observed_sr > benchmark_sr else 0.0
    sr0 = expected_max_sharpe_ratio(trials, benchmark_sr, se)
    return float(stats.norm.cdf((observed_sr - sr0) / se))


def dsr_from_returns(
    returns,
    trials: int = 1,
    benchmark_sr: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """由收益率序列直接计算 DSR 及其构成项（推荐入口）。

    内部在每期口径上计算 Sharpe/偏度/峰度，返回 dict（含每期与年化 Sharpe、DSR）。
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 3:
        raise ValueError("收益率至少 3 个观测（计算偏度/峰度）")
    rf_per = risk_free_rate / periods_per_year
    excess = (r - rf_per).to_numpy(dtype=float)
    n = len(excess)
    std = excess.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        raise ValueError("收益率标准差为 0（或非有限），Sharpe 无定义")
    sr_period = float(excess.mean() / std)
    sr_annual = float(excess.mean() / std * np.sqrt(periods_per_year))
    skew, excess_kurt, full_kurt = skewness_kurtosis(excess)
    dsr = deflated_sharpe_ratio(
        sr_period, n, skew, full_kurt, trials=trials, benchmark_sr=benchmark_sr
    )
    return {
        "n_observations": n,
        "sharpe_period": sr_period,
        "sharpe_annual": sr_annual,
        "skew": skew,
        "kurtosis_excess": excess_kurt,
        "kurtosis": full_kurt,
        "sharpe_se": sharpe_standard_error(sr_period, n, skew, full_kurt),
        "dsr": dsr,
        "trials": int(trials),
        "benchmark_sr": float(benchmark_sr),
    }


# ---------------------------------------------------------------------------
# CSCV / PBO
# ---------------------------------------------------------------------------
@dataclass
class CSCVResult:
    """CSCV 输出。"""

    logits: np.ndarray  # 每个组合的 rank logit λ_c
    omega_ranks: np.ndarray  # 每个组合中 IS 最优策略的 OOS rank（1=最好）
    n_combinations: int
    n_submatrices: int
    pbo: float  # 参数版 PBO = Φ(mean(λ)/std(λ))
    pbo_freq: float  # 非参数版 PBO = P(λ_c > 0)（IS 最优落入 OOS 下半区）


def cscv_splits(
    n_submatrices: int,
    max_combinations: int | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成 CSCV 的组合对称 IS/OOS 划分。

    S 个子矩阵，任取 S/2 个做 IS（其余 OOS），共 C(S, S/2) 个组合。
    组合数超过 ``max_combinations`` 时随机抽样（固定种子可复现）。
    返回 List[(is_sub_idx, oos_sub_idx)]，索引指向子矩阵编号。
    """
    s = int(n_submatrices)
    if s < 2 or s % 2 != 0:
        raise ValueError("n_submatrices 必须为偶数且 >= 2")
    half = s // 2
    all_idx = list(range(s))
    combos = list(combinations(all_idx, half))
    if max_combinations is not None and len(combos) > int(max_combinations):
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(combos), size=int(max_combinations), replace=False))
        combos = [combos[i] for i in chosen]
    out = []
    for c in combos:
        is_set = set(c)
        oos = np.array([i for i in all_idx if i not in is_set], dtype=int)
        out.append((np.array(c, dtype=int), oos))
    return out


def cscv(
    performance_matrix,
    n_submatrices: int = 4,
    performance_func: Callable | None = None,
    max_combinations: int | None = None,
    seed: int = 42,
) -> CSCVResult:
    """组合对称交叉验证（CSCV）。

    ``performance_matrix``：shape (T, N)，T 行=时间观测，N 列=策略配置。
    步骤：
    1. 行切成 S 个等长子矩阵（``np.array_split``）；
    2. 每个组合取 S/2 个子矩阵为 IS，其余为 OOS；
    3. 各列在 IS/OOS 上的 performance（默认均值）→ 找到 IS 最优列 n*；
    4. 计算 n* 的 OOS rank ω（1=最好），logit λ = ln(ω/(N-ω+1))；
    5. PBO = Φ(mean(λ)/std(λ))（参数版）与 P(λ>0)（非参数版）。
    """
    M = np.asarray(performance_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("performance_matrix 必须为 (T, N) 二维数组")
    T, N = M.shape
    if N < 2:
        raise ValueError("至少需要 2 个策略配置（列）")
    s = int(n_submatrices)
    if s < 2 or s % 2 != 0:
        raise ValueError("n_submatrices 必须为偶数且 >= 2")
    if s > T:
        raise ValueError("n_submatrices 不能超过观测行数 T")
    if performance_func is None:

        def performance_func(x):
            return np.nanmean(x, axis=0)

    bounds = np.array_split(np.arange(T), s)
    logits: list[float] = []
    omegas: list[int] = []
    for is_idx, oos_idx in cscv_splits(s, max_combinations=max_combinations, seed=seed):
        is_rows = np.concatenate([bounds[i] for i in is_idx])
        oos_rows = np.concatenate([bounds[i] for i in oos_idx])
        r_is = np.asarray(performance_func(M[is_rows]), dtype=float)
        r_oos = np.asarray(performance_func(M[oos_rows]), dtype=float)
        n_star = int(np.argmax(r_is))
        omega = int(np.sum(r_oos > r_oos[n_star])) + 1  # 1=最好
        logits.append(float(np.log(omega / (N - omega + 1))))
        omegas.append(omega)

    logits_arr = np.asarray(logits, dtype=float)
    pbo_freq = float(np.mean(logits_arr > 0.0))
    mean = float(logits_arr.mean())
    std = float(logits_arr.std(ddof=1)) if len(logits_arr) > 1 else 0.0
    if std > 1e-12:
        pbo = float(stats.norm.cdf(mean / std))
    else:
        pbo = 1.0 if mean > 0.0 else (0.0 if mean < 0.0 else 0.5)
    return CSCVResult(
        logits=logits_arr,
        omega_ranks=np.asarray(omegas, dtype=float),
        n_combinations=len(logits_arr),
        n_submatrices=s,
        pbo=pbo,
        pbo_freq=pbo_freq,
    )


def probability_of_backtest_overfitting(
    performance_matrix,
    n_submatrices: int = 4,
    performance_func: Callable | None = None,
    max_combinations: int | None = None,
    seed: int = 42,
) -> float:
    """PBO（参数版）：IS 最优策略在 OOS 落入下半区的概率。"""
    return cscv(
        performance_matrix,
        n_submatrices=n_submatrices,
        performance_func=performance_func,
        max_combinations=max_combinations,
        seed=seed,
    ).pbo


# ---------------------------------------------------------------------------
# OOS/IS 退化对比（与 A4 联动）
# ---------------------------------------------------------------------------
def oosis_degradation(
    is_equity,
    oos_equity,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """OOS 相对 IS 的退化对比（消费 A4 ``compare_is_oos``）。

    返回 dict：is/oos 年化收益、Sharpe、最大回撤，以及
    sharpe_degradation = OOS/IS Sharpe、return_degradation = OOS/IS 年化收益。
    """
    return compare_is_oos(
        is_equity,
        oos_equity,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )


# ---------------------------------------------------------------------------
# 风险结论与上线建议
# ---------------------------------------------------------------------------
def conclude_overfitting(
    dsr: float | None = None,
    pbo: float | None = None,
    sharpe_degradation: float | None = None,
) -> dict:
    """按阈值给出整体风险等级与上线建议（取各可用信号的最差等级）。"""
    levels: list[str] = []
    reasons: list[str] = []

    if dsr is not None:
        if dsr >= DSR_SAFE:
            lv = RISK_LOW
        elif dsr >= DSR_HIGH_RISK:
            lv = RISK_MEDIUM
        else:
            lv = RISK_HIGH
        levels.append(lv)
        reasons.append(
            f"DSR={dsr:.3f}（≥{DSR_SAFE} 低 / ≥{DSR_HIGH_RISK} 中 / <{DSR_HIGH_RISK} 高）"
        )
    if pbo is not None:
        if pbo <= PBO_LOW_RISK:
            lv = RISK_LOW
        elif pbo <= PBO_HIGH_RISK:
            lv = RISK_MEDIUM
        else:
            lv = RISK_HIGH
        levels.append(lv)
        reasons.append(
            f"PBO={pbo:.3f}（≤{PBO_LOW_RISK} 低 / ≤{PBO_HIGH_RISK} 中 / >{PBO_HIGH_RISK} 高）"
        )
    if sharpe_degradation is not None:
        if sharpe_degradation >= DEGRADATION_SAFE:
            lv = RISK_LOW
        elif sharpe_degradation >= DEGRADATION_HIGH_RISK:
            lv = RISK_MEDIUM
        else:
            lv = RISK_HIGH
        levels.append(lv)
        reasons.append(
            f"OOS/IS Sharpe 衰减={sharpe_degradation:.3f}"
            f"（≥{DEGRADATION_SAFE} 低 / ≥{DEGRADATION_HIGH_RISK} 中 / <{DEGRADATION_HIGH_RISK} 高）"
        )

    if RISK_HIGH in levels:
        overall = RISK_HIGH
    elif RISK_MEDIUM in levels:
        overall = RISK_MEDIUM
    elif levels:
        overall = RISK_LOW
    else:
        overall = RISK_MEDIUM

    recommendation = {
        RISK_HIGH: "不建议上线：存在明显过拟合特征，回测收益难以在实盘复现",
        RISK_MEDIUM: "暂缓上线：需更多样本外验证、简化模型或降低试验次数后重新评估",
        RISK_LOW: "可上线：样本外表现稳健，建议小仓位实盘验证并持续监控衰减",
    }[overall]
    return {"risk_level": overall, "recommendation": recommendation, "reasons": reasons}


# ---------------------------------------------------------------------------
# 汇总评估
# ---------------------------------------------------------------------------
@dataclass
class OverfitAssessment:
    """过拟合评估结论（供报告生成）。"""

    risk_level: str
    recommendation: str
    reasons: list[str] = field(default_factory=list)
    dsr: float | None = None
    pbo: float | None = None
    pbo_freq: float | None = None
    sharpe_degradation: float | None = None
    return_degradation: float | None = None
    is_sharpe: float | None = None
    oos_sharpe: float | None = None
    sharpe_period: float | None = None
    sharpe_annual: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    trials: int = 1
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "reasons": self.reasons,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "pbo_freq": self.pbo_freq,
            "sharpe_degradation": self.sharpe_degradation,
            "return_degradation": self.return_degradation,
            "is_sharpe": self.is_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "sharpe_period": self.sharpe_period,
            "sharpe_annual": self.sharpe_annual,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "trials": self.trials,
            "thresholds": self.thresholds,
        }


def assess_overfitting(
    is_equity=None,
    oos_equity=None,
    returns=None,
    observed_sharpe: float | None = None,
    num_observations: int | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
    trials: int = 1,
    benchmark_sr: float = 0.0,
    performance_matrix=None,
    n_submatrices: int = 4,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OverfitAssessment:
    """过拟合评估入口：组合 DSR / PBO / OOS-IS 退化，输出风险结论。

    至少提供以下之一：``returns``（推荐）、``observed_sharpe`` 三件套
    （observed_sharpe + num_observations + skew + kurtosis）。可选：
    ``is_equity``/``oos_equity``（退化对比，A4 联动）、``performance_matrix``（PBO）。

    注意：``observed_sharpe`` 与 ``benchmark_sr`` 均为**非年化（每期）**口径。
    """
    dsr: float | None = None
    sharpe_annual: float | None = None
    sharpe_period: float | None = None
    skew_v = skew
    kurt_v = kurtosis

    # 1) OOS/IS 退化（A4 联动）
    degradation = None
    if is_equity is not None and oos_equity is not None:
        degradation = compare_is_oos(
            is_equity,
            oos_equity,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

    # 2) DSR
    if returns is not None:
        info = dsr_from_returns(
            returns,
            trials=trials,
            benchmark_sr=benchmark_sr,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        dsr = info["dsr"]
        sharpe_annual = info["sharpe_annual"]
        sharpe_period = info["sharpe_period"]
        skew_v = info["skew"]
        kurt_v = info["kurtosis"]
    elif observed_sharpe is not None:
        if num_observations is None or skew is None or kurtosis is None:
            raise ValueError("提供 observed_sharpe 时需同时提供 num_observations / skew / kurtosis")
        dsr = deflated_sharpe_ratio(
            observed_sharpe,
            num_observations,
            skew,
            kurtosis,
            trials=trials,
            benchmark_sr=benchmark_sr,
        )

    # 3) PBO
    pbo: float | None = None
    pbo_freq: float | None = None
    if performance_matrix is not None:
        res = cscv(performance_matrix, n_submatrices=n_submatrices)
        pbo = res.pbo
        pbo_freq = res.pbo_freq

    # 4) 结论
    sharpe_deg = degradation["sharpe_degradation"] if degradation else None
    conclusion = conclude_overfitting(dsr=dsr, pbo=pbo, sharpe_degradation=sharpe_deg)

    return OverfitAssessment(
        risk_level=conclusion["risk_level"],
        recommendation=conclusion["recommendation"],
        reasons=conclusion["reasons"],
        dsr=dsr,
        pbo=pbo,
        pbo_freq=pbo_freq,
        sharpe_degradation=sharpe_deg,
        return_degradation=degradation["return_degradation"] if degradation else None,
        is_sharpe=degradation["is_sharpe"] if degradation else None,
        oos_sharpe=degradation["oos_sharpe"] if degradation else None,
        sharpe_period=sharpe_period,
        sharpe_annual=sharpe_annual,
        skew=skew_v,
        kurtosis=kurt_v,
        trials=int(trials),
        thresholds={
            "dsr_safe": DSR_SAFE,
            "dsr_high_risk": DSR_HIGH_RISK,
            "pbo_low_risk": PBO_LOW_RISK,
            "pbo_high_risk": PBO_HIGH_RISK,
            "degradation_safe": DEGRADATION_SAFE,
            "degradation_high_risk": DEGRADATION_HIGH_RISK,
        },
    )


__all__ = [
    "DEGRADATION_HIGH_RISK",
    "DEGRADATION_SAFE",
    "DSR_HIGH_RISK",
    "DSR_SAFE",
    "EULER_MASCHERONI",
    "PBO_HIGH_RISK",
    "PBO_LOW_RISK",
    "RISK_HIGH",
    "RISK_LOW",
    "RISK_MEDIUM",
    "CSCVResult",
    "OverfitAssessment",
    "assess_overfitting",
    "conclude_overfitting",
    "cscv",
    "cscv_splits",
    "deflated_sharpe_ratio",
    "dsr_from_returns",
    "expected_max_sharpe_ratio",
    "oosis_degradation",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_standard_error",
    "skewness_kurtosis",
]
