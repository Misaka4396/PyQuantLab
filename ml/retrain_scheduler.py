"""C3 重训调度：按周期自动触发重训，模型版本化，输出调度日志。

与 C2 WalkForward 的 ``retrain_every`` 语义一致：每 ``retrain_every`` 步触发一次
重训；其余步沿用上一模型。每次重训经 ``ModelRegistry.save`` 落新版本，模型与
数据/特征/超参版本绑定，可回滚。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ml.model_registry import ModelRegistry


class RetrainScheduler:
    """按周期自动重训的调度器。"""

    def __init__(
        self,
        retrain_every: int = 252,
        registry: ModelRegistry | None = None,
    ):
        if retrain_every < 1:
            raise ValueError("retrain_every 必须为正")
        self.retrain_every = int(retrain_every)
        self.registry = registry
        self.log: list = []

    # ------------------------------------------------------------------
    def should_retrain(self, step: int) -> bool:
        """第 step 步是否应重训（step % retrain_every == 0）。"""
        return int(step) % self.retrain_every == 0

    # ------------------------------------------------------------------
    def run(
        self,
        train_fn: Callable[[int], tuple],
        n_steps: int,
        *,
        task: str = "classification",
        data_version: int = 1,
        feature_version: int = 1,
        hyperparams: dict | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """推进 n_steps，周期触发 ``train_fn(step)`` 重训并版本化，返回调度日志。

        ``train_fn(step)`` 返回 ``(model, metrics_dict)``；若 registry 为空则只记日志。
        """
        hyperparams = hyperparams or {}
        self.log = []
        for step in range(n_steps):
            if not self.should_retrain(step):
                self.log.append({"step": step, "retrained": False, "version": None})
                continue
            model, metrics = train_fn(step)
            version = None
            if self.registry is not None:
                version = self.registry.save(
                    model,
                    {
                        "model_type": "lightgbm",
                        "task": task,
                        "data_version": data_version,
                        "feature_version": feature_version,
                        "hyperparams": hyperparams,
                        "seed": seed,
                        "metrics": metrics,
                    },
                )
            row = {"step": step, "retrained": True, "version": version}
            row.update({f"metric_{k}": v for k, v in (metrics or {}).items()})
            self.log.append(row)
        return self.schedule_log()

    def schedule_log(self) -> pd.DataFrame:
        """返回调度日志 DataFrame。"""
        return pd.DataFrame(self.log)


__all__ = ["RetrainScheduler"]
