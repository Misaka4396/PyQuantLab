"""C3 训练配置：种子 / 周期 / 模型开关 / 数据版本（模型与数据版本绑定）。

只定义参数，不含训练逻辑。DL/RL 为**可选开关**：关闭时训练只跑 LightGBM 基线。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# 任务类型
TASK_CLASSIFICATION = "classification"   # 方向二分类（>0 为 1）
TASK_REGRESSION = "regression"           # 收益回归
TASKS = (TASK_CLASSIFICATION, TASK_REGRESSION)

# 深度学习模型类型
DL_LSTM = "lstm"
DL_TRANSFORMER = "transformer"
DL_MODELS = (DL_LSTM, DL_TRANSFORMER)


@dataclass
class TrainConfig:
    """训练配置（种子固定保证可复现）。"""

    # ------------------------------------------------------------------
    # 复现与任务
    # ------------------------------------------------------------------
    seed: int = 42
    task: str = TASK_CLASSIFICATION

    # ------------------------------------------------------------------
    # 模型开关（DL/RL 为可选开关，关闭时只跑 LightGBM）
    # ------------------------------------------------------------------
    use_lightgbm: bool = True
    use_dl: bool = False                # 深度学习开关（LSTM/Transformer）
    use_rl: bool = False                # RL 开关（只留接口，不强制实现训练）

    # ------------------------------------------------------------------
    # 数据 / 特征版本（与模型版本绑定，见 model_registry）
    # ------------------------------------------------------------------
    data_version: int = 1
    feature_version: int = 1

    # ------------------------------------------------------------------
    # LightGBM 超参（留空用默认；覆盖项合并进默认）
    # ------------------------------------------------------------------
    lgb_params: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 深度学习超参
    # ------------------------------------------------------------------
    dl_model: str = DL_LSTM
    seq_len: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    dl_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_frac: float = 0.8           # 时序切分训练比例（严格按时间先后）

    # ------------------------------------------------------------------
    # 重训周期（步数 / bar 数）
    # ------------------------------------------------------------------
    retrain_every: int = 252

    # ------------------------------------------------------------------
    # 模型存储目录
    # ------------------------------------------------------------------
    model_dir: str = "./data_cache/models"

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task 必须为 {TASKS} 之一，得到 {self.task!r}")
        if not any((self.use_lightgbm, self.use_dl, self.use_rl)):
            raise ValueError("至少开启一个模型开关（use_lightgbm/use_dl/use_rl）")
        if self.dl_model not in DL_MODELS:
            raise ValueError(f"dl_model 必须为 {DL_MODELS} 之一，得到 {self.dl_model!r}")
        if self.seq_len < 1:
            raise ValueError("seq_len 必须为正")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size / num_layers 必须为正")
        if self.dl_epochs < 1 or self.batch_size < 1:
            raise ValueError("dl_epochs / batch_size 必须为正")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac 必须在 (0, 1)")
        if self.retrain_every < 1:
            raise ValueError("retrain_every 必须为正")


__all__ = [
    "TrainConfig",
    "TASK_CLASSIFICATION",
    "TASK_REGRESSION",
    "DL_LSTM",
    "DL_TRANSFORMER",
    "DL_MODELS",
]
