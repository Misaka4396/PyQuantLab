"""C4 过拟合检测与模型评估单元测试。

覆盖验收标准：
- DSR 与手算参考值一致（Bailey & López de Prado 2014 公式）
- PBO/CSCV 与手算一致（IS 好 OOS 差 → PBO=1；IS 好 OOS 也好 → PBO=0）
- OOS/IS 退化对比（构造 IS 好 OOS 差的序列，输出退化幅度并判定风险）
- 特征重要性跨折稳定性（一致 → 1；反转 → 负相关）
- 过拟合风险结论（含阈值）与报告生成
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.overfit import (
    DSR_HIGH_RISK,
    DSR_SAFE,
    cscv,
    cscv_splits,
    conclude_overfitting,
    deflated_sharpe_ratio,
    dsr_from_returns,
    assess_overfitting,
    sharpe_standard_error,
    skewness_kurtosis,
)
from ml.feature_importance import feature_importance_stability, to_rank_series
from ml.overfit_report import generate_overfit_report, render_overfit_report


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def eq_from_returns(returns, start=100.0) -> pd.Series:
    """由收益率序列构造权益序列。"""
    vals = [start]
    for r in returns:
        vals.append(vals[-1] * (1.0 + r))
    return pd.Series(vals)


# ---------------------------------------------------------------------------
# DSR / PSR
# ---------------------------------------------------------------------------
def test_sharpe_standard_error_manual():
    """SE = sqrt((1 - γ3·SR + (γ4-1)/4·SR²)/(T-1))，正态（γ3=0, γ4=3）。"""
    se = sharpe_standard_error(1.0, 100, skew=0.0, kurtosis=3.0)
    assert se == pytest.approx(math.sqrt(1.5 / 99), rel=1e-12)


def test_deflated_sharpe_ratio_matches_reference():
    """与手算参考值一致（returns=[0.02,0.03,-0.01,0.04,0.02,0.01]，rf=0）。"""
    sr = 1.0644053746097524
    skew = -0.495459683238928
    kurt = 2.422042671379877
    n = 6
    assert deflated_sharpe_ratio(sr, n, skew, kurt, trials=1) == pytest.approx(
        0.9566579850561787, abs=1e-9
    )
    assert deflated_sharpe_ratio(sr, n, skew, kurt, trials=2) == pytest.approx(
        0.88364355293574, abs=1e-9
    )
    assert deflated_sharpe_ratio(sr, n, skew, kurt, trials=10) == pytest.approx(
        0.5550997642778439, abs=1e-9
    )


def test_dsr_from_returns_matches_reference():
    returns = [0.02, 0.03, -0.01, 0.04, 0.02, 0.01]
    info1 = dsr_from_returns(returns, trials=1, risk_free_rate=0.0)
    info10 = dsr_from_returns(returns, trials=10, risk_free_rate=0.0)
    assert info1["dsr"] == pytest.approx(0.9566579850561787, abs=1e-9)
    assert info10["dsr"] == pytest.approx(0.5550997642778439, abs=1e-9)
    assert set(info1) >= {"sharpe_period", "sharpe_annual", "skew", "kurtosis", "dsr"}


def test_dsr_decreases_with_trials():
    """试验次数越多，多重检验校正越强，DSR 越低。"""
    returns = [0.02, 0.03, -0.01, 0.04, 0.02, 0.01]
    dsrs = [dsr_from_returns(returns, trials=t, risk_free_rate=0.0)["dsr"] for t in (1, 2, 5, 10)]
    assert all(0.0 <= d <= 1.0 for d in dsrs)
    assert dsrs == sorted(dsrs, reverse=True)


def test_dsr_in_unit_interval():
    """DSR 是概率，必然落在 [0, 1]。"""
    for returns in ([0.01, 0.02, -0.01, 0.015, 0.005], [-0.02, 0.01, 0.03, -0.01, 0.02], [0.1, -0.1, 0.1, -0.1]):
        d = dsr_from_returns(returns, trials=3, risk_free_rate=0.0)["dsr"]
        assert 0.0 <= d <= 1.0


def test_skewness_kurtosis_symmetric():
    """对称数据偏度≈0，正态近似下 full 峰度≈3。"""
    skew, excess, full = skewness_kurtosis([1, -1, 2, -2, 1, -1])
    assert skew == pytest.approx(0.0, abs=1e-12)
    assert full == pytest.approx(excess + 3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# CSCV / PBO
# ---------------------------------------------------------------------------
def test_cscv_splits_count():
    """S=4 → C(4,2)=6 个组合，每个组合 IS/OOS 各 2 个子矩阵。"""
    splits = cscv_splits(4)
    assert len(splits) == 6
    for is_idx, oos_idx in splits:
        assert len(is_idx) == 2
        assert len(oos_idx) == 2
        assert set(is_idx).isdisjoint(set(oos_idx))


def test_cscv_overfit_case_pbo_one():
    """IS 最优者在 OOS 始终最差 → 典型过拟合，PBO=1。"""
    M = np.array([[10, 0], [10, 0], [0, 5], [0, 5]], dtype=float)
    res = cscv(M, n_submatrices=2)
    assert res.pbo == pytest.approx(1.0)
    assert res.pbo_freq == pytest.approx(1.0)
    np.testing.assert_allclose(res.omega_ranks, [2.0, 2.0])
    np.testing.assert_allclose(res.logits, [math.log(2.0), math.log(2.0)])


def test_cscv_clean_case_pbo_zero():
    """IS 最优者在 OOS 也最优 → 无过拟合，PBO=0。"""
    M = np.array([[10, 0], [10, 0], [10, 0], [0, 5]], dtype=float)
    res = cscv(M, n_submatrices=2)
    assert res.pbo == pytest.approx(0.0)
    assert res.pbo_freq == pytest.approx(0.0)
    np.testing.assert_allclose(res.omega_ranks, [1.0, 1.0])


def test_cscv_matches_manual_reference():
    """随机矩阵下与独立手写 CSCV 参考实现一致。"""
    rng = np.random.default_rng(7)
    M = rng.normal(size=(8, 4))

    def ref_logits(M, S):
        T, N = M.shape
        bounds = np.array_split(np.arange(T), S)
        half = S // 2
        import itertools

        out = []
        for combo in itertools.combinations(range(S), half):
            is_set = set(combo)
            is_rows = np.concatenate([bounds[i] for i in combo])
            oos_rows = np.concatenate([bounds[i] for i in range(S) if i not in is_set])
            r_is = M[is_rows].mean(axis=0)
            r_oos = M[oos_rows].mean(axis=0)
            n_star = int(np.argmax(r_is))
            omega = int(np.sum(r_oos > r_oos[n_star])) + 1
            out.append(math.log(omega / (N - omega + 1)))
        return np.array(out)

    res = cscv(M, n_submatrices=4)
    np.testing.assert_allclose(res.logits, ref_logits(M, 4))
    assert res.pbo_freq == pytest.approx(float(np.mean(res.logits > 0.0)))


# ---------------------------------------------------------------------------
# OOS/IS 退化 + 风险结论
# ---------------------------------------------------------------------------
def test_oosis_degradation_is_good_oos_bad():
    """构造 IS 涨 OOS 跌的序列：OOS Sharpe<0、退化幅度<0，判高风险。"""
    is_eq = eq_from_returns([0.1, -0.018, 0.111, -0.0167, 0.1017])
    oos_eq = eq_from_returns([-0.0385, 0.024, -0.0469, 0.0164, -0.0484], start=is_eq.iloc[-1])
    a = assess_overfitting(is_equity=is_eq, oos_equity=oos_eq)
    assert a.is_sharpe > 0
    assert a.oos_sharpe < 0
    assert a.sharpe_degradation < 0
    assert a.risk_level == "高"
    assert "不建议上线" in a.recommendation


def test_assess_overfitting_high_risk_from_dsr_and_pbo():
    """回测漂亮实盘亏损典型特征：低 DSR + 高 PBO + 负退化 → 高风险。"""
    is_eq = eq_from_returns([0.1, -0.018, 0.111, -0.0167, 0.1017])
    oos_eq = eq_from_returns([-0.0385, 0.024, -0.0469, 0.0164, -0.0484], start=is_eq.iloc[-1])
    # 回测样本内收益漂亮，但 OOS 亏损 → 低 DSR
    returns = [-0.0385, 0.024, -0.0469, 0.0164, -0.0484]
    M = np.array([[10, 0], [10, 0], [0, 5], [0, 5]], dtype=float)  # PBO=1
    a = assess_overfitting(
        is_equity=is_eq, oos_equity=oos_eq, returns=returns, trials=10,
        performance_matrix=M, n_submatrices=2,
    )
    assert a.risk_level == "高"
    assert "不建议上线" in a.recommendation
    assert a.dsr < DSR_HIGH_RISK
    assert a.pbo == pytest.approx(1.0)


def test_assess_overfitting_low_risk():
    """稳健样本外 + 高 DSR + 低 PBO → 低风险、可上线。"""
    returns = [0.05, 0.06, 0.04, 0.05, 0.06, 0.04]
    eq = eq_from_returns([0.05, 0.057, 0.045, 0.0517, 0.0492])
    M = np.array([[10, 0], [10, 0], [10, 0], [0, 5]], dtype=float)  # PBO=0
    a = assess_overfitting(is_equity=eq, oos_equity=eq, returns=returns, trials=1, performance_matrix=M)
    assert a.risk_level == "低"
    assert "可上线" in a.recommendation
    assert a.dsr >= DSR_SAFE
    assert a.pbo == pytest.approx(0.0)


def test_conclude_overfitting_thresholds():
    """阈值映射：DSR<0.95 至少中风险，<0.80 高风险，≥0.95 低风险。"""
    assert conclude_overfitting(dsr=0.99)["risk_level"] == "低"
    assert conclude_overfitting(dsr=0.90)["risk_level"] == "中"
    assert conclude_overfitting(dsr=0.70)["risk_level"] == "高"
    assert conclude_overfitting(pbo=0.05)["risk_level"] == "低"
    assert conclude_overfitting(pbo=0.5)["risk_level"] == "高"
    assert conclude_overfitting(sharpe_degradation=-0.1)["risk_level"] == "高"


# ---------------------------------------------------------------------------
# 特征重要性稳定性
# ---------------------------------------------------------------------------
def test_feature_importance_stability_identical():
    """各折重要性完全一致 → 稳定性 1。"""
    imp = {"a": 0.5, "b": 0.3, "c": 0.2}
    res = feature_importance_stability([imp, imp, imp])
    assert res["mean_correlation"] == pytest.approx(1.0)
    assert res["stability_score"] == pytest.approx(1.0)


def test_feature_importance_stability_reversed():
    """重要性反转 → 负相关、稳定性评分截断为 0。"""
    res = feature_importance_stability([
        {"a": 0.5, "b": 0.3, "c": 0.2},
        {"a": 0.1, "b": 0.3, "c": 0.5},
    ])
    assert res["mean_correlation"] == pytest.approx(-1.0)
    assert res["stability_score"] == pytest.approx(0.0)


def test_to_rank_series_accepts_dataframe():
    """DataFrame（feature/importance 两列）输入也能转 rank。"""
    df = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.2]})
    ranks = to_rank_series(df)
    assert ranks["a"] == pytest.approx(1.0)
    assert ranks["c"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def test_overfit_report_contains_thresholds_and_conclusion(tmp_path):
    is_eq = eq_from_returns([0.1, -0.018, 0.111, -0.0167, 0.1017])
    oos_eq = eq_from_returns([-0.0385, 0.024, -0.0469, 0.0164, -0.0484], start=is_eq.iloc[-1])
    a = assess_overfitting(is_equity=is_eq, oos_equity=oos_eq, returns=[-0.0385, 0.024, -0.0469, 0.0164, -0.0484], trials=10)
    text = render_overfit_report(a)
    assert "判定阈值" in text
    assert str(DSR_SAFE) in text
    assert "不建议上线" in text
    assert a.risk_level in text

    path = generate_overfit_report(a, tmp_path, name="overfit")
    assert (tmp_path / "overfit.md").exists()
    assert "Deflated Sharpe" in (tmp_path / "overfit.md").read_text(encoding="utf-8")
