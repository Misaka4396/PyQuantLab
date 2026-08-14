"""C2 训练/验证框架单元测试（Purged K-fold + Walk-forward）。

覆盖验收标准：
- 无训练/验证重叠（purge+embargo 后 train/val 时间戳不相交）
- purge/embargo 正确（与手工实现一致）
- OOS 与 IS 分报告（walk_forward 输出两组指标）
- 重训时间点明确（输出重训 schedule）
- OOS 只报告一次（report_oos 二次调用抛错）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.cv import as_sorted_index, fold_table, purged_kfold
from ml.walk_forward import WalkForward


def manual_purged_kfold(times, n_splits, embargo, purge_horizon):
    """手工实现的 purged K-fold（用于对照）。"""
    n = len(times)
    folds = np.array_split(np.arange(n), n_splits)
    out = []
    for fold in folds:
        v0, v1 = int(fold[0]), int(fold[-1])
        train = np.array([i for i in range(n) if not (v0 - purge_horizon <= i <= v1 + embargo)])
        out.append((train, fold.astype(int)))
    return out


# ---------------------------------------------------------------------------
# Purged K-fold
# ---------------------------------------------------------------------------
def test_purged_kfold_no_train_val_overlap():
    times = pd.date_range("2024-01-02 09:30", periods=30, freq="1min")
    splits = purged_kfold(times, n_splits=5, embargo=2, purge_horizon=3)
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
        # 训练样本不得落在 [v0-purge, v1+embargo] 区间（无标签重叠/串扰）
        v0, v1 = int(val_idx[0]), int(val_idx[-1])
        bad = [i for i in train_idx if (v0 - 3) <= i <= (v1 + 2)]
        assert bad == []


def test_purged_kfold_matches_manual():
    times = pd.date_range("2024-01-02", periods=10, freq="D")
    embargo, purge = 1, 1
    manual = manual_purged_kfold(times, n_splits=2, embargo=embargo, purge_horizon=purge)
    got = purged_kfold(times, n_splits=2, embargo=embargo, purge_horizon=purge)
    for (mt, mv), (gt, gv) in zip(manual, got, strict=False):
        np.testing.assert_array_equal(mt, gt)
        np.testing.assert_array_equal(mv, gv)


def test_purged_kfold_embargo_purge_exact():
    times = pd.date_range("2024-01-02", periods=10, freq="D")
    splits = purged_kfold(times, n_splits=2, embargo=1, purge_horizon=1)
    # fold0 val=[0..4] → 剔除 [-1..5] → train=[6,7,8,9]
    np.testing.assert_array_equal(splits[0][0], np.array([6, 7, 8, 9]))
    np.testing.assert_array_equal(splits[0][1], np.array([0, 1, 2, 3, 4]))
    # fold1 val=[5..9] → 剔除 [4..10] → train=[0,1,2,3]
    np.testing.assert_array_equal(splits[1][0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(splits[1][1], np.array([5, 6, 7, 8, 9]))


def test_purged_kfold_no_purge_is_standard_kfold():
    times = pd.date_range("2024-01-02", periods=10, freq="D")
    splits = purged_kfold(times, n_splits=2, embargo=0, purge_horizon=0)
    np.testing.assert_array_equal(splits[0][0], np.array([5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(splits[0][1], np.array([0, 1, 2, 3, 4]))


def test_fold_table_shape():
    times = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    splits = purged_kfold(times, n_splits=4, embargo=1, purge_horizon=1)
    tbl = fold_table(times, splits)
    assert set(tbl.columns) == {"fold", "role", "position", "timestamp"}
    assert tbl["role"].isin(["train", "val"]).all()


def test_unsorted_times_raises():
    times = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    with pytest.raises(ValueError):
        as_sorted_index(times)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def _linear_data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
    X = rng.normal(size=(n, 2))
    y = X @ np.array([1.0, -2.0]) + rng.normal(0, 0.1, n)
    return idx, pd.DataFrame(X, index=idx), pd.Series(y, index=idx)


def _model_fn():
    return {"w": None}


def _fit(model, X, y):
    model["w"] = np.linalg.lstsq(
        np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None
    )[0]


def _predict(model, X):
    return np.asarray(X, dtype=float) @ model["w"]


def _mse(y, yp):
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(yp, dtype=float)) ** 2))


def test_walk_forward_split_no_overlap_and_retrain_schedule():
    idx, _X, _y = _linear_data(30)
    wf = WalkForward(n_train=10, n_test=5, step=5, retrain_every=2, embargo=1, purge_horizon=1)
    splits = wf.split(idx)
    assert len(splits) > 0
    for sp in splits:
        assert len(set(sp.train_idx) & set(sp.test_idx)) == 0
        assert sp.train_idx.max() < sp.test_idx.min()  # 训练严格在测试之前
    assert splits[0].retrain is True
    assert splits[1].retrain is False


def test_walk_forward_reports_is_and_oos_separately():
    idx, X, y = _linear_data(30)
    wf = WalkForward(n_train=10, n_test=5, step=5, retrain_every=1, embargo=0, purge_horizon=0)
    res = wf.evaluate(idx, X, y, _model_fn, _fit, _predict, _mse)

    assert "is_metric" in res.is_metrics.columns
    assert "oos_metric" in res.oos_metrics.columns
    assert len(res.oos_metrics) == len(res.is_metrics) > 0
    assert res.oos_overall >= 0
    assert res.is_overall >= 0
    # 每个测试样本只预测一次
    expected_n = sum(len(sp.test_idx) for sp in res.folds)
    assert len(res.oos_predictions) == expected_n
    # 重训 schedule 明确
    assert not res.retrain_schedule.empty
    assert "retrain_time" in res.retrain_schedule.columns


def test_walk_forward_oos_report_only_once():
    idx, X, y = _linear_data(30)
    wf = WalkForward(n_train=10, n_test=5, step=5)
    res = wf.evaluate(idx, X, y, _model_fn, _fit, _predict, _mse)
    res.report_oos()  # 第一次成功
    with pytest.raises(RuntimeError):
        res.report_oos()  # 第二次抛错（防多次偷看）
