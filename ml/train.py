"""C3 训练入口：LightGBM 基线 + DL 开关（参数化），可复现、可重训、版本化。

用法：:

    from ml.train import train, train_lightgbm
    from ml.run_config import TrainConfig

    cfg = TrainConfig(seed=42, task="classification", use_lightgbm=True,
                      use_dl=False, use_rl=False, model_dir="./data_cache/models")
    result = train(cfg, X, y, times=index)   # 返回 {models, versions, config}

要点：
- LightGBM 基线在 C2 Purged CV 框架内做超参/指标评估（``ml.cv.purged_kfold``）。
- DL/RL 为开关：config 关闭时只训练 LightGBM。
- 固定种子 + 记录数据/特征/超参版本，模型经 ``ModelRegistry`` 版本化，可回滚。
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from ml.cv import purged_kfold
from ml.model_registry import ModelRegistry
from ml.models_lgb import LightGBMModel
from ml.models_torch import train_torch_from_arrays
from ml.run_config import TrainConfig, TASK_CLASSIFICATION, TASK_REGRESSION


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _encode_labels(task: str, y) -> np.ndarray:
    """标签编码：分类 → 方向二分类（>0 为 1）；回归 → 连续值。"""
    ya = np.asarray(y).ravel()
    if task == TASK_CLASSIFICATION:
        return (ya > 0).astype(int)
    return ya.astype(float)


def _iloc(obj, idx: np.ndarray):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def _default_times(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="min")


# ---------------------------------------------------------------------------
# LightGBM 基线（C2 Purged CV 内评估）
# ---------------------------------------------------------------------------
def _cv_score(
    config: TrainConfig,
    X,
    y,
    times,
    n_splits: int,
) -> float:
    """在 Purged K-fold 上评估（无前视），返回平均指标。"""
    n = len(np.asarray(y))
    if n < 2 or n_splits < 2:
        return float("nan")
    if times is None:
        times = _default_times(n)
    splits = purged_kfold(times, n_splits=n_splits, embargo=0, purge_horizon=0)
    scores = []
    for train_idx, val_idx in splits:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model = LightGBMModel(task=config.task, params=config.lgb_params, seed=config.seed)
        model.fit(_iloc(X, train_idx), _iloc(y, train_idx))
        pred = model.predict(_iloc(X, val_idx))
        yv = np.asarray(_iloc(y, val_idx)).ravel()
        if config.task == TASK_CLASSIFICATION:
            scores.append(accuracy_score(yv, pred))
        else:
            scores.append(r2_score(yv, pred))
    return float(np.mean(scores)) if scores else float("nan")


def train_lightgbm(
    config: TrainConfig,
    X,
    y,
    times: Optional[pd.DatetimeIndex] = None,
    n_splits: int = 3,
) -> dict:
    """训练 LightGBM 基线：Purged CV 评估 + 全样本训练，返回 {model, metrics}。"""
    ya = _encode_labels(config.task, y)
    n = len(ya)
    n_splits = min(n_splits, n) if n >= 2 else 1
    cv_score = _cv_score(config, X, ya, times, n_splits)

    model = LightGBMModel(task=config.task, params=config.lgb_params, seed=config.seed)
    model.fit(X, ya)
    return {
        "model": model,
        "metrics": {"cv_score": cv_score, "n_samples": int(n)},
    }


# ---------------------------------------------------------------------------
# 深度学习（开关，CPU）
# ---------------------------------------------------------------------------
def train_dl(config: TrainConfig, X, y) -> dict:
    """训练 LSTM/Transformer（严格时序切分，CPU），返回 {model, metrics, history}。"""
    ya = np.asarray(y, dtype=float).ravel()
    res = train_torch_from_arrays(
        X, ya,
        model_name=config.dl_model,
        seq_len=config.seq_len,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        epochs=config.dl_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        train_frac=config.train_frac,
        seed=config.seed,
    )
    history = res["history"]
    return {
        "model": res["model"],
        "metrics": {"final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan")},
        "history": history,
    }


# ---------------------------------------------------------------------------
# 超参（版本绑定用）
# ---------------------------------------------------------------------------
def _hyperparams_for(config: TrainConfig, name: str) -> dict:
    if name == "torch":
        return {
            "dl_model": config.dl_model, "seq_len": config.seq_len,
            "hidden_size": config.hidden_size, "num_layers": config.num_layers,
            "dropout": config.dropout, "dl_epochs": config.dl_epochs,
            "batch_size": config.batch_size, "learning_rate": config.learning_rate,
        }
    return dict(config.lgb_params)


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def train(
    config: TrainConfig,
    X,
    y,
    times: Optional[pd.DatetimeIndex] = None,
) -> dict:
    """训练入口：按 config 开关训练 LightGBM / DL / RL，并版本化保存。

    返回 {"models": {...}, "versions": {...}, "config": config}。
    DL/RL 关闭时只训练 LightGBM（验收：config 关闭时训练只跑 LightGBM）。
    """
    config.validate()
    trained: dict = {}
    if config.use_lightgbm:
        trained["lightgbm"] = train_lightgbm(config, X, y, times=times)
    if config.use_dl:
        trained["torch"] = train_dl(config, X, y)
    if config.use_rl:
        # RL 扩展：stable-baselines3 可选开关，本批次只留接口，不强制实现训练
        trained["rl"] = {
            "model": None,
            "metrics": {},
            "note": "RL 扩展接口预留（stable-baselines3 可选开关）",
        }

    registry = ModelRegistry(config.model_dir)
    versions: dict = {}
    for name, r in trained.items():
        model = r.get("model")
        if model is None:
            continue
        versions[name] = registry.save(model, {
            "model_type": "torch" if name == "torch" else "lightgbm",
            "task": config.task,
            "data_version": config.data_version,
            "feature_version": config.feature_version,
            "hyperparams": _hyperparams_for(config, name),
            "seed": config.seed,
            "metrics": r.get("metrics", {}),
        })

    return {"models": trained, "versions": versions, "config": config}


__all__ = [
    "train",
    "train_lightgbm",
    "train_dl",
    "_encode_labels",
]
