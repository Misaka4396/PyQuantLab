"""C4 特征重要性稳定性分析（跨 fold 的 rank 稳定性 / 相关性）。

多折训练会得到多组特征重要性；若某特征在各折的排名来回跳动，说明模型对特征
的依赖不稳定、易过拟合。本模块把每组重要性转成 rank（1=最重要，并列取平均秩），
再计算两两折间的 Spearman / Kendall 相关性，汇总为稳定性评分。

来源：特征重要性跨折一致性（rank stability）是 ML 过拟合诊断的常用手段，参见
López de Prado (2018) *Advances in Financial Machine Learning* 第 8 章特征重要性
相关讨论。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_rank_series(importance) -> pd.Series:
    """把单折重要性转为 rank Series（1=最重要，并列取平均秩）。

    接受：
    - DataFrame（含 feature / importance 两列，与 LightGBMModel.feature_importance 一致）
    - dict {feature: importance}
    - Series（index=特征名）
    """
    if isinstance(importance, pd.DataFrame):
        if "feature" not in importance.columns or "importance" not in importance.columns:
            raise ValueError("DataFrame 需含 feature / importance 两列")
        s = importance.set_index("feature")["importance"].astype(float)
    elif isinstance(importance, dict):
        s = pd.Series(importance, dtype=float)
    elif isinstance(importance, pd.Series):
        s = importance.astype(float)
    else:
        raise TypeError("importance 需为 DataFrame / dict / Series")
    return s.rank(ascending=False, method="average")


def feature_importance_stability(
    importances: list,
    method: str = "spearman",
) -> dict:
    """跨 fold 特征重要性稳定性分析。

    参数：
    - importances：多折重要性列表（每个元素见 ``to_rank_series``）。
    - method：'spearman' 或 'kendall'（相关性口径）。

    返回 dict：
    - mean_correlation：两两折间 rank 相关性的均值。
    - stability_score：mean_correlation 截断到 [0, 1] 的稳定性评分（越大越稳定）。
    - rank_std_mean：各特征跨折 rank 标准差的均值（越小越稳定）。
    - pairwise_correlations：两两折间相关性列表。
    - feature_stats：各特征的 mean_rank / rank_std。
    """
    if method not in ("spearman", "kendall"):
        raise ValueError("method 需为 spearman 或 kendall")
    if len(importances) < 2:
        raise ValueError("至少需要 2 折的重要性")

    ranks = [to_rank_series(x) for x in importances]
    all_features = sorted(set().union(*[set(r.index) for r in ranks]))
    mat = pd.DataFrame({i: r.reindex(all_features) for i, r in enumerate(ranks)})

    corr_mat = mat.corr(method=method)  # pairwise complete 处理缺失
    n = len(ranks)
    pairwise = [float(corr_mat.iloc[i, j]) for i in range(n) for j in range(i + 1, n)]
    mean_corr = float(np.nanmean(pairwise))
    stability_score = float(max(0.0, min(1.0, mean_corr)))

    rank_std = mat.std(axis=1)
    feature_stats = (
        pd.DataFrame(
            {
                "feature": all_features,
                "mean_rank": mat.mean(axis=1).to_numpy(),
                "rank_std": rank_std.to_numpy(),
            }
        )
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )

    return {
        "mean_correlation": mean_corr,
        "stability_score": stability_score,
        "rank_std_mean": float(rank_std.mean()),
        "pairwise_correlations": pairwise,
        "feature_stats": feature_stats,
    }


__all__ = [
    "feature_importance_stability",
    "to_rank_series",
]
