"""C1 特征工程与标注流水线单元测试。

覆盖验收标准：
- label 与特征严格对齐 t→t+h（未来收益 / 方向 / 波动率 + 可追溯 label_asof）
- 特征只用 t 及之前数据（动量手算对照）
- 滚动标准化与手工逐窗口计算一致
- 泄漏审计能检测"特征用了未来数据"（故意引入未来数据必须报错）
- 干净特征通过审计（无误报）
- 可复现 + 特征版本化存储
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    LABEL_ASOF,
    LABEL_FWD_DIRECTION,
    LABEL_FWD_RETURN,
    LABEL_FWD_VOLATILITY,
    FeatureStore,
    assemble,
    build_features,
    build_labels,
    rolling_standardize,
)
from ml.leakage_audit import LeakageAuditor, LeakageError


def make_ohlcv(closes, opens=None, highs=None, lows=None, volumes=None):
    """构造 OHLCV DataFrame（index=工作日）。"""
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    opens = opens if opens is not None else closes
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    volumes = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 标注：t → t+h 对齐
# ---------------------------------------------------------------------------
def test_labels_fwd_return_and_asof():
    prices = pd.Series(
        [100.0, 110.0, 99.0, 108.9], index=pd.date_range("2024-01-01", periods=4, freq="B")
    )
    labels = build_labels(prices, horizon=2)
    # fwd_return_t = close_{t+2}/close_t - 1
    assert labels.loc[prices.index[0], LABEL_FWD_RETURN] == pytest.approx(99.0 / 100.0 - 1)
    assert labels.loc[prices.index[1], LABEL_FWD_RETURN] == pytest.approx(108.9 / 110.0 - 1)
    assert pd.isna(labels.loc[prices.index[2], LABEL_FWD_RETURN])
    # label_asof_t = index_{t+h}
    assert labels.loc[prices.index[0], LABEL_ASOF] == prices.index[2]
    assert labels.loc[prices.index[1], LABEL_ASOF] == prices.index[3]


def test_labels_direction():
    prices = pd.Series([100.0, 90.0, 80.0], index=pd.date_range("2024-01-01", periods=3, freq="B"))
    labels = build_labels(prices, horizon=1)
    assert labels.loc[prices.index[0], LABEL_FWD_DIRECTION] == -1
    assert labels.loc[prices.index[0], LABEL_FWD_RETURN] == pytest.approx(-0.1)


def test_labels_volatility():
    """fwd_volatility_t = std(returns_{t+1..t+h})，用已知收益手算对照。"""
    prices = pd.Series(
        [100.0, 110.0, 99.0, 108.9], index=pd.date_range("2024-01-01", periods=4, freq="B")
    )
    labels = build_labels(prices, horizon=2)
    # returns = [NaN, 0.1, -0.1, 0.1]；h=2：fwd_vol[0]=std(0.1,-0.1)=0.14142
    expected = np.std([0.1, -0.1], ddof=1)
    assert labels.loc[prices.index[0], LABEL_FWD_VOLATILITY] == pytest.approx(expected)
    assert labels.loc[prices.index[1], LABEL_FWD_VOLATILITY] == pytest.approx(
        np.std([-0.1, 0.1], ddof=1)
    )


# ---------------------------------------------------------------------------
# 特征：只用 t 及之前
# ---------------------------------------------------------------------------
def test_features_momentum_uses_past_only():
    closes = [10.0, 10.5, 10.2, 10.8, 11.0, 11.5, 11.3, 11.8]
    ohlcv = make_ohlcv(closes)
    feats = build_features(ohlcv, momentum_windows=(5,))
    col = "momentum_5"
    # momentum_5[t] = close_t/close_{t-5} - 1
    assert feats[col].iloc[5] == pytest.approx(closes[5] / closes[0] - 1)
    assert feats[col].iloc[7] == pytest.approx(closes[7] / closes[2] - 1)
    # 前 5 根无历史，为 NaN（不得前视）
    assert pd.isna(feats[col].iloc[4])


def test_features_intraday_return():
    ohlcv = make_ohlcv(closes=[10.0, 10.5], opens=[9.5, 10.0])
    feats = build_features(ohlcv)
    assert feats["intraday_return"].iloc[0] == pytest.approx(10.0 / 9.5 - 1)


# ---------------------------------------------------------------------------
# 滚动标准化：与手工逐窗口计算一致
# ---------------------------------------------------------------------------
def test_rolling_standardize_matches_manual():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = pd.DataFrame({"x": x})
    out = rolling_standardize(df, window=3)
    # 手工：z_t = (x_t - mean(window)) / std(window, ddof=1)
    for t in range(2, 5):
        win = x[t - 2 : t + 1]
        expected = (x[t] - np.mean(win)) / np.std(win, ddof=1)
        assert out["x"].iloc[t] == pytest.approx(expected)
    assert pd.isna(out["x"].iloc[0])
    assert pd.isna(out["x"].iloc[1])


# ---------------------------------------------------------------------------
# 泄漏审计
# ---------------------------------------------------------------------------
def test_leakage_audit_detects_future_label():
    """故意把特征设为未来标签（fwd_return），审计必须报错。"""
    prices = pd.Series(
        [100.0, 102.0, 101.0, 104.0, 106.0, 103.0, 107.0, 110.0],
        index=pd.date_range("2024-01-01", periods=8, freq="B"),
    )
    labels = build_labels(prices, horizon=2)
    features = pd.DataFrame({"cheat": labels[LABEL_FWD_RETURN]}, index=prices.index)

    auditor = LeakageAuditor()
    report = auditor.audit(features, labels, label_cols=[LABEL_FWD_RETURN])
    assert report.has_leaks
    assert any(i.check == "label_leakage" for i in report.issues)

    # audit_raise 应抛 LeakageError
    with pytest.raises(LeakageError):
        auditor.audit_raise(features, labels, label_cols=[LABEL_FWD_RETURN])


def test_leakage_audit_detects_future_price():
    """故意把特征设为未来价格 close_{t+1}，审计必须报错。"""
    prices = pd.Series(
        [100.0, 102.0, 101.0, 104.0, 106.0], index=pd.date_range("2024-01-01", periods=5, freq="B")
    )
    labels = build_labels(prices, horizon=1)
    features = pd.DataFrame({"future_price": prices.shift(-1)}, index=prices.index)

    auditor = LeakageAuditor()
    report = auditor.audit(features, labels, close=prices, horizons=(1,))
    assert report.has_leaks
    assert any(i.check == "future_price_leakage" for i in report.issues)


def test_leakage_audit_clean_passes():
    """干净特征（只用历史数据）不应误报。"""
    closes = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 40)))
    ohlcv = make_ohlcv(list(closes))
    feats = build_features(ohlcv)
    labels = build_labels(ohlcv["close"], horizon=5)

    auditor = LeakageAuditor()
    report = auditor.audit(feats, labels, close=ohlcv["close"], horizons=(1, 5, 10))
    assert not report.has_leaks, report.to_frame()


# ---------------------------------------------------------------------------
# 组装 + 版本化 + 复现
# ---------------------------------------------------------------------------
def test_assemble_aligns_features_and_labels():
    closes = np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 30)))
    ohlcv = make_ohlcv(list(closes))
    df = assemble(ohlcv, horizon=5, standardize_window=20)
    assert LABEL_FWD_RETURN in df.columns
    assert "momentum_5" in df.columns
    # 内连接后无 NaN 行（标签尾部被 dropna 剔除），特征已滚动标准化
    assert df[LABEL_FWD_RETURN].notna().all()


def test_features_reproducible():
    closes = [10.0, 10.5, 10.2, 10.8, 11.0, 11.5, 11.3, 11.8]
    ohlcv = make_ohlcv(closes)
    f1 = build_features(ohlcv)
    f2 = build_features(ohlcv)
    pd.testing.assert_frame_equal(f1, f2)


def test_feature_store_versioning(tmp_path):
    store = FeatureStore(tmp_path)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    store.save(df, "demo", version=1)
    store.save(df * 2, "demo", version=2)
    assert store.latest_version("demo") == 2
    loaded = store.load("demo", 1)
    pd.testing.assert_frame_equal(loaded, df)
