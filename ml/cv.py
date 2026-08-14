"""C2 Purged K-fold + embargo（López de Prado）用于时序超参选择。

时序数据不能随机 K-fold：相邻样本的标签在时间上重叠，随机切分会让训练集"偷看"
验证集标签。Purged K-fold 对每个验证折：
1. **purge**：剔除训练集中标签与验证集重叠的样本（标签区间 [t, t+purge_horizon]）。
2. **embargo**：在验证集之后追加 embargo 缓冲，剔除该区间内的训练样本，防止
   验证集之后的信息串扰。

单位约定：``purge_horizon`` 与 ``embargo`` 均以 **bar 数**（观测数）计，便于与
手工实现对照；若需按时间长度 purge，可先把时间戳重采样为等间隔再调用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_sorted_index(times) -> pd.DatetimeIndex:
    """把任意时间序列输入归一化为升序 DatetimeIndex，乱序直接报错。"""
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    if not idx.is_monotonic_increasing:
        raise ValueError("times 必须按时间升序排序")
    return idx


def _excluded_mask(
    pos: np.ndarray, v0: int, v1: int, purge_horizon: int, embargo: int
) -> np.ndarray:
    """返回位置数组 pos 中应被剔除的布尔掩码（True=剔除）。

    剔除区间 = [v0 - purge_horizon, v1 + embargo]：
    - [v0 - purge_horizon, v1]：标签与验证折重叠的训练样本（purge）。
    - (v1, v1 + embargo]：验证折之后的 embargo 缓冲。
    """
    return (pos >= v0 - purge_horizon) & (pos <= v1 + embargo)


def purged_kfold(
    times,
    n_splits: int = 5,
    embargo: int = 0,
    purge_horizon: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold 切分。

    参数：
    - times：升序时间戳（DatetimeIndex / list / Series）。
    - n_splits：折数 K（>=2 且 <= 样本数）。
    - purge_horizon：标签区间长度（bar 数），用于 purge 与验证集标签重叠的训练样本。
    - embargo：验证折之后禁用的样本数（bar 数）。

    返回 List[(train_idx, val_idx)]，均为位置索引（相对 times 顺序，可 .iloc 使用）。
    """
    idx = as_sorted_index(times)
    n = len(idx)
    if n_splits < 2:
        raise ValueError("n_splits 至少为 2")
    if n_splits > n:
        raise ValueError("n_splits 不能超过样本数")
    if embargo < 0 or purge_horizon < 0:
        raise ValueError("embargo / purge_horizon 不能为负")

    fold_bounds = np.array_split(np.arange(n), n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_bounds:
        v0, v1 = int(fold[0]), int(fold[-1])
        pos = np.arange(n)
        train_idx = pos[~_excluded_mask(pos, v0, v1, purge_horizon, embargo)]
        splits.append((train_idx, fold.astype(int)))
    return splits


def fold_table(times, splits) -> pd.DataFrame:
    """把切分结果展开为可视化长表：fold / role / position / timestamp。"""
    idx = as_sorted_index(times)
    rows = []
    for k, (train_idx, val_idx) in enumerate(splits):
        for i in train_idx:
            rows.append({"fold": k, "role": "train", "position": int(i), "timestamp": idx[int(i)]})
        for i in val_idx:
            rows.append({"fold": k, "role": "val", "position": int(i), "timestamp": idx[int(i)]})
    return pd.DataFrame(rows, columns=["fold", "role", "position", "timestamp"])


__all__ = ["as_sorted_index", "fold_table", "purged_kfold"]
