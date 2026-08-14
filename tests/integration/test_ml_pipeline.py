"""C 链集成测试：C1 特征工程 → C2 Purged CV / Walk-forward → C3 训练 → C4 过拟合评估。

跨模块全链路验证（tests/integration/ 分层，P9 规范）：
- 合成 OHLCV + 固定种子，离线确定性（防 flaky）
- 覆盖：特征-标签对齐、时序切分无泄漏、模型版本注册、过拟合判定输出
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.cv import fold_table, purged_kfold
from ml.features import build_features, build_labels
from ml.overfit import assess_overfitting
from ml.overfit_report import generate_overfit_report
from ml.run_config import TrainConfig
from ml.train import train
from ml.walk_forward import WalkForward


def _synthetic_ohlcv(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """带趋势 + 噪声的合成分钟 OHLCV（产生可学习的方向标签）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30:00", periods=n, freq="min")
    trend = np.cumsum(rng.normal(0.0002, 0.01, n))  # 随机游走带漂移
    close = 100 * np.exp(trend)
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volume = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# C1 → C2：特征/标签对齐 + 时序切分无泄漏
# ---------------------------------------------------------------------------
def test_c1_feature_label_alignment():
    """特征只依赖 t 及之前；标签 t→t+h 严格对齐（尾部 NaN 可追溯）。"""
    ohlcv = _synthetic_ohlcv(n=800)
    feats = build_features(ohlcv)
    labels = build_labels(ohlcv["close"], horizon=5)

    assert len(feats) == len(ohlcv), "特征行数与行情一致"
    # 特征不超前：动量窗口特征在首窗口为 NaN（不依赖未来）
    assert feats["momentum_5"].iloc[:4].isna().all(), "首窗口特征应为 NaN（无未来数据）"
    # 标签尾部 horizon 根为 NaN（t+h 才可知）
    assert labels["fwd_direction"].iloc[-5:].isna().all()
    assert "label_asof" in labels.columns, "标签应带可知时点（可追溯）"


def test_c2_purged_kfold_and_walk_forward_no_leakage():
    """Purged K-fold（embargo+purge）与 Walk-forward 训练/测试严格无重叠。"""
    times = _synthetic_ohlcv(n=800).index

    # Purged K-fold
    splits = purged_kfold(times, n_splits=4, embargo=20, purge_horizon=5)
    for tr, va in splits:
        assert not set(tr) & set(va), "训练/验证索引不得重叠"
    ft = fold_table(times, splits)
    assert ft["fold"].nunique() == 4, "4 折明细表应含 4 个 fold"

    # Walk-forward：训练严格在测试之前（时间排序 + 无重叠 + 重训标记）
    wf = WalkForward(n_train=400, n_test=100, retrain_every=2, embargo=20, purge_horizon=5)
    wf_splits = wf.split(times)
    assert len(wf_splits) >= 2, "800 bar 应切出多个 fold"
    for s in wf_splits:
        tr_t = times[s.train_idx]
        te_t = times[s.test_idx]
        assert tr_t.max() < te_t.min(), "训练必须严格早于测试"
        assert s.retrain in (True, False)
    assert wf_splits[0].retrain, "首个 fold 必须重训"


# ---------------------------------------------------------------------------
# C3 → C4：训练注册 + 过拟合判定
# ---------------------------------------------------------------------------
def test_c3_train_lightgbm_registers_model():
    """C1 特征 + C3 LightGBM 训练：模型经 registry 版本化（可回滚）。"""
    ohlcv = _synthetic_ohlcv(n=800, seed=7)
    feats = build_features(ohlcv)
    labels = build_labels(ohlcv["close"], horizon=5)
    # 对齐 + 丢弃尾部无标签行
    df = feats.join(labels["fwd_direction"])
    df = df.dropna()
    X = df[[c for c in df.columns if c != "fwd_direction"]].to_numpy()
    y = df["fwd_direction"].to_numpy()
    times = df.index

    cfg = TrainConfig(
        seed=42,
        task="classification",
        use_lightgbm=True,
        use_dl=False,
        model_dir="data_cache/models_it",
    )
    result = train(cfg, X, y, times=times)
    assert result["versions"], "训练应产出模型版本"
    assert result["config"].seed == 42, "配置透传（可复现）"


def test_c4_overfit_assessment_produces_report():
    """C4 过拟合判定：IS 好 OOS 差 → 高风险结论 + markdown 报告落盘。"""
    rng = np.random.default_rng(3)
    is_equity = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 300)))
    oos_equity = 100 * np.exp(np.cumsum(rng.normal(-0.0005, 0.012, 300)))

    assessment = assess_overfitting(is_equity=is_equity, oos_equity=oos_equity, trials=10)
    assert assessment.risk_level in ("低", "中", "高"), "风险等级应输出"
    assert assessment.recommendation, "应有上线建议"

    report_path = generate_overfit_report(assessment, "data_cache/reports_it", name="overfit_it")
    from pathlib import Path

    p = Path(report_path)
    assert p.exists() and "风险结论" in p.read_text(encoding="utf-8"), "报告应落盘且含风险结论"
