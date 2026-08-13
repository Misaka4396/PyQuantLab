"""C2 Walk-forward（滚动回测）框架：时间切分、定期重训、OOS 只报告一次。

时序评估正确姿势：
- 训练/测试严格按时间先后切分，未来数据绝不参与训练。
- 定期重训（retrain_every 可配）：只有标记 retrain 的折才重训模型，其余折沿用
  上一模型；重训时点明确记录（retrain_schedule）。
- purge/embargo：purge 掉训练集末尾与测试标签重叠的样本，embargo 在训练/测试间
  留缓冲，进一步隔离信息。
- OOS 严格隔离：OOS 预测只生成一次，指标只报告一次（report_oos 二次调用抛错，
  防多次偷看导致选择偏差）。

与 ml.cv 的 Purged K-fold 分工：本模块做最终滚动回测评估，Purged K-fold 做超参
选择（见 cv.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from ml.cv import as_sorted_index


@dataclass
class WalkForwardSplit:
    """单次滚动切分。"""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    retrain: bool
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    """Walk-forward 评估结果。"""

    folds: List[WalkForwardSplit]
    is_metrics: pd.DataFrame       # 样本内指标（每 fold）
    oos_metrics: pd.DataFrame      # 样本外指标（每 fold）
    retrain_schedule: pd.DataFrame  # 重训 schedule（fold / retrain_time / 区间）
    oos_predictions: pd.Series     # OOS 预测（每个测试样本只预测一次）
    is_overall: float = 0.0
    oos_overall: float = 0.0
    _oos_reported: bool = field(default=False, repr=False)

    def report_oos(self) -> pd.DataFrame:
        """报告 OOS 指标（只允许一次，防止多次偷看）。"""
        if self._oos_reported:
            raise RuntimeError("OOS 指标只能报告一次（防止多次偷看导致选择偏差）")
        self._oos_reported = True
        return self.oos_metrics


class WalkForward:
    """滚动回测切分与评估框架。"""

    def __init__(
        self,
        n_train: int,
        n_test: int,
        step: Optional[int] = None,
        retrain_every: int = 1,
        embargo: int = 0,
        purge_horizon: int = 0,
    ):
        if n_train < 1 or n_test < 1:
            raise ValueError("n_train / n_test 必须为正")
        if retrain_every < 1:
            raise ValueError("retrain_every 必须为正")
        if embargo < 0 or purge_horizon < 0:
            raise ValueError("embargo / purge_horizon 不能为负")
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.step = int(step) if step is not None else int(n_test)
        self.retrain_every = int(retrain_every)
        self.embargo = int(embargo)
        self.purge_horizon = int(purge_horizon)

    # ------------------------------------------------------------------
    def split(self, times) -> List[WalkForwardSplit]:
        """按时间顺序切分训练/测试，返回切分列表（含重训标记）。"""
        idx = as_sorted_index(times)
        n = len(idx)
        splits: List[WalkForwardSplit] = []
        fold = 0
        start = 0
        while start + self.n_train + self.embargo + self.n_test <= n:
            train_end = start + self.n_train
            test_start = train_end + self.embargo
            test_end = test_start + self.n_test

            # purge：剔除训练集末尾与测试标签重叠的样本（标签区间长度 = purge_horizon）
            train_idx = np.arange(start, train_end)
            if self.purge_horizon > 0:
                cutoff = test_start - self.purge_horizon
                train_idx = train_idx[train_idx < cutoff]

            retrain = (fold % self.retrain_every == 0)
            splits.append(WalkForwardSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=np.arange(test_start, test_end),
                retrain=retrain,
                train_start=idx[int(train_idx[0])] if len(train_idx) else pd.NaT,
                train_end=idx[int(train_idx[-1])] if len(train_idx) else pd.NaT,
                test_start=idx[test_start],
                test_end=idx[test_end - 1],
            ))
            start += self.step
            fold += 1
        return splits

    # ------------------------------------------------------------------
    def evaluate(
        self,
        times,
        X,
        y,
        model_fn: Callable,
        fit: Callable,
        predict: Callable,
        metric: Callable,
    ) -> WalkForwardResult:
        """执行滚动回测评估。

        - model_fn() -> model：新建模型。
        - fit(model, X_train, y_train) -> None：训练。
        - predict(model, X) -> array：预测。
        - metric(y_true, y_pred) -> float：指标（越大越好/越小越好由调用方统一）。
        """
        idx = as_sorted_index(times)
        splits = self.split(idx)
        Xa = X if isinstance(X, (pd.DataFrame, pd.Series)) else pd.DataFrame(np.asarray(X), index=idx)
        ya = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y), index=idx)

        model = model_fn()
        is_rows, oos_rows, retrain_rows = [], [], []
        oos_pred_parts = []
        all_ytest_parts = []
        all_ytr_parts = []
        all_is_pred_parts = []

        for sp in splits:
            if len(sp.train_idx) == 0 or len(sp.test_idx) == 0:
                continue
            Xtr, ytr = Xa.iloc[sp.train_idx], ya.iloc[sp.train_idx]
            Xte, yte = Xa.iloc[sp.test_idx], ya.iloc[sp.test_idx]

            if sp.retrain:
                model = model_fn()
                fit(model, Xtr, ytr)
                retrain_rows.append({
                    "fold": sp.fold,
                    "retrain_time": sp.train_end,
                    "train_start": sp.train_start,
                    "train_end": sp.train_end,
                    "test_start": sp.test_start,
                    "test_end": sp.test_end,
                })

            pred_tr = np.asarray(predict(model, Xtr)).ravel()
            pred_te = np.asarray(predict(model, Xte)).ravel()
            is_rows.append({"fold": sp.fold, "is_metric": float(metric(ytr, pred_tr))})
            oos_rows.append({"fold": sp.fold, "oos_metric": float(metric(yte, pred_te))})

            oos_pred_parts.append(pd.Series(pred_te, index=ya.iloc[sp.test_idx].index))
            all_ytest_parts.append(ya.iloc[sp.test_idx])
            all_ytr_parts.append(ytr)
            all_is_pred_parts.append(pd.Series(pred_tr, index=ya.iloc[sp.train_idx].index))

        is_metrics = pd.DataFrame(is_rows, columns=["fold", "is_metric"])
        oos_metrics = pd.DataFrame(oos_rows, columns=["fold", "oos_metric"])
        retrain_schedule = pd.DataFrame(retrain_rows)
        oos_pred = pd.concat(oos_pred_parts) if oos_pred_parts else pd.Series(dtype=float)

        is_overall = float(metric(pd.concat(all_ytr_parts), pd.concat(all_is_pred_parts))) \
            if all_ytr_parts else float("nan")
        oos_overall = float(metric(pd.concat(all_ytest_parts), oos_pred)) \
            if all_ytest_parts else float("nan")

        return WalkForwardResult(
            folds=splits,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            retrain_schedule=retrain_schedule,
            oos_predictions=oos_pred,
            is_overall=is_overall,
            oos_overall=oos_overall,
        )


__all__ = ["WalkForward", "WalkForwardSplit", "WalkForwardResult"]
