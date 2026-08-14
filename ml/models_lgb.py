"""C3 LightGBM 基线：方向分类 / 收益回归（可复现）。

可复现要点：
- 固定 ``random_state``、``n_jobs=1``、``deterministic=True``、``force_row_wise=True``。
- 同种子同数据两次训练，指标与预测完全一致。
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.run_config import TASK_CLASSIFICATION, TASK_REGRESSION


def _base_params(seed: int) -> dict:
    """LightGBM 默认超参（含确定性开关，保证可复现）。"""
    return {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": int(seed),
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


class LightGBMModel:
    """LightGBM 基线封装（方向二分类 / 收益回归）。"""

    def __init__(
        self,
        task: str = TASK_CLASSIFICATION,
        params: dict | None = None,
        seed: int = 42,
    ):
        if task not in (TASK_CLASSIFICATION, TASK_REGRESSION):
            raise ValueError(f"task 必须为 classification/regression，得到 {task!r}")
        self.task = task
        self.seed = int(seed)
        self.params = {**_base_params(seed), **(params or {})}
        self.model = None
        self.feature_names_: list | None = None

    # ------------------------------------------------------------------
    def fit(self, X, y) -> LightGBMModel:
        """训练。y 为标签：分类=0/1 方向，回归=连续收益。"""
        Xa = self._as_frame(X)
        ya = self._as_labels(y)
        if self.task == TASK_CLASSIFICATION:
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(Xa, ya)
        self.feature_names_ = list(Xa.columns)
        return self

    def predict(self, X):
        """预测：分类返回 0/1 类别，回归返回连续值。"""
        self._check_fitted()
        return np.asarray(self.model.predict(self._as_frame(X)))

    def predict_proba(self, X):
        """分类概率（仅分类任务有效），返回 shape (n, 2)。"""
        self._check_fitted()
        return np.asarray(self.model.predict_proba(self._as_frame(X)))

    def feature_importance(self) -> pd.DataFrame:
        """特征重要度（gain 口径）。"""
        self._check_fitted()
        imp = getattr(self.model, "booster_", None)
        names = self.feature_names_ or [f"f{i}" for i in range(self.model.n_features_in_)]
        if imp is None:
            return pd.DataFrame({"feature": names, "importance": [0.0] * len(names)})
        gain = imp.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": names, "importance": gain})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    def _as_labels(self, y):
        ya = np.asarray(y).ravel()
        if self.task == TASK_CLASSIFICATION:
            # 方向二分类：>0 记为 1，否则 0（-1/0/1 或连续收益均适用）
            return (ya > 0).astype(int)
        return ya.astype(float)

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("模型未训练，先调用 fit()")


def train_direction_classifier(X, y, seed: int = 42, params: dict | None = None) -> LightGBMModel:
    """训练方向分类器（薄封装，语义化入口）。"""
    return LightGBMModel(TASK_CLASSIFICATION, params=params, seed=seed).fit(X, y)


def train_return_regressor(X, y, seed: int = 42, params: dict | None = None) -> LightGBMModel:
    """训练收益回归器（薄封装，语义化入口）。"""
    return LightGBMModel(TASK_REGRESSION, params=params, seed=seed).fit(X, y)


__all__ = [
    "LightGBMModel",
    "train_direction_classifier",
    "train_return_regressor",
]
