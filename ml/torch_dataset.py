"""C3 PyTorch Dataset：按时间窗口构建样本，禁止跨时间 shuffle。

关键约束（防前视 / 防泄漏）：
- 第 i 个样本 = 特征窗口 ``X[i : i+seq_len]`` + 标签 ``y[i+seq_len-1]``，
  即只用窗口末端时点 t 及之前的 seq_len 根 bar，绝不用 t 之后的数据。
- 配合 ``DataLoader(shuffle=False)`` 保持时间顺序；训练循环禁止 shuffle。
- 时序切分（train/val/test）必须按时间先后切（见 ``train_val_split_by_time``），
  不能用随机切分，否则未来样本会进入训练集。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """时间窗口数据集。

    参数：
    - X：shape (T, F) 的特征矩阵（按时间升序）。
    - y：shape (T,) 的标签（point-in-time 对齐，y[t] 只依赖 t 及之前信息）。
    - seq_len：时间窗口长度（bar 数）。
    """

    def __init__(self, X, y, seq_len: int):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 2:
            raise ValueError("X 必须为 (T, F) 二维数组")
        if len(X) != len(y):
            raise ValueError(f"X 与 y 长度不一致：{len(X)} vs {len(y)}")
        if seq_len < 1 or seq_len > len(X):
            raise ValueError(f"seq_len 需满足 1 <= seq_len <= {len(X)}")
        self.X = X
        self.y = y
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        # 窗口 [idx, idx+seq_len)，标签取窗口末端 idx+seq_len-1（无未来数据）
        end = idx + self.seq_len
        x = self.X[idx:end]
        y = self.y[end - 1]
        return torch.from_numpy(x.copy()), torch.tensor(float(y), dtype=torch.float32)

    # ------------------------------------------------------------------
    def window(self, idx: int) -> Tuple[np.ndarray, float]:
        """返回第 idx 个样本的 (特征窗口 ndarray, 标签)（供测试/审计对照）。"""
        end = idx + self.seq_len
        return self.X[idx:end].copy(), float(self.y[end - 1])


def train_val_split_by_time(
    X, y, train_frac: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按时间先后严格切分训练/验证（无随机，无未来串扰）。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    n = len(X)
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError(f"train_frac={train_frac} 导致切分无效（n={n}）")
    return X[:cut], y[:cut], X[cut:], y[cut:]


__all__ = ["TimeSeriesDataset", "train_val_split_by_time"]
