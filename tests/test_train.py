"""C3 训练层单元测试。

覆盖验收标准：
- LightGBM 基线可复现（同种子同数据两次训练指标一致）
- torch_dataset 时序切分正确（无跨时间/无未来数据）
- LSTM / Transformer 前向可跑
- 模型可回滚（registry 支持加载历史版本）
- 重训自动触发（scheduler 逻辑测试）
- DL/RL 为开关（config 关闭时训练只跑 LightGBM）
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.model_registry import ModelRegistry
from ml.models_lgb import LightGBMModel
from ml.models_torch import (
    LSTMModel,
    TransformerModel,
    set_seed,
    train_torch_from_arrays,
)
from ml.retrain_scheduler import RetrainScheduler
from ml.run_config import TrainConfig
from ml.torch_dataset import TimeSeriesDataset, train_val_split_by_time
from ml.train import train, train_lightgbm


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def _synthetic(n: int = 400, f: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, f))
    score = X[:, 0] + 0.3 * X[:, 1] + rng.normal(0.0, 0.5, n)
    y = (score > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# LightGBM 可复现
# ---------------------------------------------------------------------------
def test_lightgbm_reproducible_same_seed():
    X, y = _synthetic()
    cfg = TrainConfig(seed=42, task="classification", use_lightgbm=True, use_dl=False, use_rl=False)
    r1 = train_lightgbm(cfg, X, y)
    r2 = train_lightgbm(cfg, X, y)
    # 同种子同数据两次训练指标一致
    assert r1["metrics"]["cv_score"] == pytest.approx(r2["metrics"]["cv_score"], abs=1e-12)
    np.testing.assert_array_equal(r1["model"].predict(X), r2["model"].predict(X))


def test_lightgbm_direction_classification_binary():
    X, y = _synthetic()
    model = LightGBMModel(task="classification", seed=42).fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    imp = model.feature_importance()
    assert list(imp.columns[:2]) == ["feature", "importance"]


def test_lightgbm_regression_runs():
    X, y = _synthetic()
    y_reg = y.astype(float) + np.random.default_rng(1).normal(0, 0.1, len(y))
    model = LightGBMModel(task="regression", seed=7).fit(X, y_reg)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


# ---------------------------------------------------------------------------
# torch_dataset 时序切分
# ---------------------------------------------------------------------------
def test_torch_dataset_time_windows_no_future():
    T, F, seq_len = 30, 3, 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, F)).astype(np.float32)
    y = np.arange(T, dtype=float)
    ds = TimeSeriesDataset(X, y, seq_len)

    assert len(ds) == T - seq_len + 1
    for i in range(len(ds)):
        xi, yi = ds[i]
        # 样本 i 只含窗口 X[i:i+seq_len]，标签取窗口末端（无未来数据）
        np.testing.assert_array_equal(xi.numpy(), X[i : i + seq_len])
        assert yi.item() == pytest.approx(y[i + seq_len - 1])


def test_train_val_split_by_time_contiguous():
    X, y = _synthetic(100)
    Xtr, _ytr, Xva, _yva = train_val_split_by_time(X, y, train_frac=0.8)
    assert len(Xtr) + len(Xva) == 100
    # 训练在前、验证在后（时间顺序，无随机、无重叠）
    Xf = X.astype(np.float32)
    np.testing.assert_array_equal(Xtr, Xf[:80])
    np.testing.assert_array_equal(Xva, Xf[80:])


# ---------------------------------------------------------------------------
# PyTorch 模型前向可跑
# ---------------------------------------------------------------------------
def test_lstm_forward_runs():
    set_seed(0)
    model = LSTMModel(input_size=4, hidden_size=8, num_layers=1)
    x = torch.randn(6, 5, 4)  # (B, seq_len, F)
    out = model(x)
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_transformer_forward_runs():
    set_seed(0)
    model = TransformerModel(input_size=4, d_model=8, nhead=2, num_layers=1)
    x = torch.randn(6, 5, 4)
    out = model(x)
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_train_torch_model_runs_on_tiny_data():
    X, y = _synthetic(120, f=4)
    y_reg = y.astype(float)
    res = train_torch_from_arrays(
        X,
        y_reg,
        model_name="lstm",
        seq_len=5,
        hidden_size=8,
        num_layers=1,
        epochs=1,
        batch_size=16,
        train_frac=0.8,
        seed=0,
    )
    assert "history" in res
    assert len(res["history"]["train_loss"]) == 1
    assert np.isfinite(res["history"]["train_loss"][0])


# ---------------------------------------------------------------------------
# 模型注册 / 回滚
# ---------------------------------------------------------------------------
def test_registry_version_binding_and_rollback(tmp_path):
    reg = ModelRegistry(tmp_path)
    v1 = reg.save(
        {"kind": "dummy", "value": 1.0},
        {
            "model_type": "dummy",
            "task": "regression",
            "data_version": 1,
            "feature_version": 1,
            "hyperparams": {"a": 1},
            "seed": 42,
            "metrics": {"m": 0.1},
        },
    )
    v2 = reg.save(
        {"kind": "dummy", "value": 2.0},
        {
            "model_type": "dummy",
            "task": "regression",
            "data_version": 2,
            "feature_version": 1,
            "hyperparams": {"a": 2},
            "seed": 42,
            "metrics": {"m": 0.2},
        },
    )
    assert v1 == 1 and v2 == 2
    assert reg.current_version() == 2

    # 回滚到历史版本
    model, meta = reg.rollback(1)
    assert reg.current_version() == 1
    assert model == {"kind": "dummy", "value": 1.0}
    assert meta["data_version"] == 1
    assert meta["hyperparams_hash"]

    df = reg.list_versions()
    assert list(df["version"]) == [1, 2]
    assert list(df["data_version"]) == [1, 2]


# ---------------------------------------------------------------------------
# 重训调度
# ---------------------------------------------------------------------------
def test_retrain_scheduler_triggers_periodically():
    sched = RetrainScheduler(retrain_every=3, registry=None)
    calls = []

    def train_fn(step):
        calls.append(step)
        return {"m": step}, {"acc": 0.9}

    log = sched.run(train_fn, n_steps=10)
    assert calls == [0, 3, 6, 9]
    assert log["retrained"].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert sched.should_retrain(0) and sched.should_retrain(6)
    assert not sched.should_retrain(1) and not sched.should_retrain(7)


def test_retrain_scheduler_versioning(tmp_path):
    reg = ModelRegistry(tmp_path)
    sched = RetrainScheduler(retrain_every=2, registry=reg)

    def train_fn(step):
        return {"kind": "dummy", "step": step}, {"acc": 0.8}

    log = sched.run(train_fn, n_steps=5, data_version=3, feature_version=7)
    retrained = log[log["retrained"]]
    assert retrained["version"].tolist() == [1, 2, 3]
    assert reg.latest_version() == 3
    _, meta = reg.load(1)
    assert meta["data_version"] == 3
    assert meta["feature_version"] == 7


# ---------------------------------------------------------------------------
# DL/RL 开关
# ---------------------------------------------------------------------------
def test_train_dl_rl_disabled_runs_only_lightgbm(tmp_path):
    X, y = _synthetic()
    cfg = TrainConfig(
        seed=42,
        task="classification",
        use_lightgbm=True,
        use_dl=False,
        use_rl=False,
        model_dir=str(tmp_path),
    )
    res = train(cfg, X, y)
    assert set(res["models"]) == {"lightgbm"}
    assert set(res["versions"]) == {"lightgbm"}


def test_train_dl_enabled_trains_torch(tmp_path):
    X, y = _synthetic(120, f=4)
    cfg = TrainConfig(
        seed=42,
        task="regression",
        use_lightgbm=True,
        use_dl=True,
        use_rl=False,
        model_dir=str(tmp_path),
        dl_model="lstm",
        seq_len=5,
        hidden_size=8,
        dl_epochs=1,
        batch_size=16,
    )
    res = train(cfg, X, y.astype(float))
    assert set(res["models"]) == {"lightgbm", "torch"}
    assert set(res["versions"]) == {"lightgbm", "torch"}
