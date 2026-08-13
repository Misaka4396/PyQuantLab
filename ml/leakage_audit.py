"""C1 泄漏审计：检测"特征用了未来数据"。

三类检测：
1. **标签泄漏**：某特征列的值与未来标签列（同一时点）近乎相等（即特征 = 标签）。
2. **未来价格泄漏**：某特征列等于未来第 h 期的价格 close_{t+h}（shift(-h)）。
3. **相关性泄漏**：某特征与未来标签的相关系数接近 1（线性泄漏）。

审计返回 ``LeakageReport``（含 issues 列表），``audit_raise`` 在有泄漏时抛
``LeakageError``。配合 ``ml.features.build_labels`` 的 ``label_asof`` 列可追溯。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class LeakageError(Exception):
    """检测到数据泄漏（特征使用了未来数据）。"""


@dataclass
class LeakageIssue:
    """单条泄漏问题。"""

    check: str          # 检测类型：label_leakage / future_price_leakage / correlation_leakage
    feature: str        # 问题特征列
    detail: str         # 说明（含被比对的对象与数值）


@dataclass
class LeakageReport:
    """泄漏审计报告。"""

    issues: List[LeakageIssue] = field(default_factory=list)

    @property
    def has_leaks(self) -> bool:
        return len(self.issues) > 0

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["check", "feature", "detail"])
        return pd.DataFrame([{"check": i.check, "feature": i.feature, "detail": i.detail} for i in self.issues])


def _near_equal(a: pd.Series, b: pd.Series, tolerance: float) -> bool:
    """两序列在共同非 NaN 处是否近乎相等。"""
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) == 0:
        return False
    diff = (aligned.iloc[:, 0].astype(float) - aligned.iloc[:, 1].astype(float)).abs().max()
    return diff <= tolerance


def _numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """过滤出数值列（排除 label_asof 等时间戳/非数值列）。"""
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


class LeakageAuditor:
    """特征泄漏审计器。"""

    def __init__(self, value_tolerance: float = 1e-8, corr_threshold: float = 0.999):
        self.value_tolerance = value_tolerance
        self.corr_threshold = corr_threshold

    # ------------------------------------------------------------------
    def audit_label_leakage(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        label_cols: Optional[Sequence[str]] = None,
    ) -> List[LeakageIssue]:
        """检测特征列是否等于未来标签列（同一时点）。"""
        issues: List[LeakageIssue] = []
        raw_cols = list(label_cols) if label_cols is not None else list(labels.columns)
        label_cols = _numeric_cols(labels, raw_cols)
        for lcol in label_cols:
            if lcol not in labels.columns:
                continue
            for fcol in features.columns:
                if fcol in labels.columns:  # 跳过标签自身
                    continue
                if _near_equal(features[fcol], labels[lcol], self.value_tolerance):
                    issues.append(LeakageIssue(
                        "label_leakage", fcol,
                        f"特征 {fcol} 与未来标签 {lcol} 近乎相等（max|diff|<={self.value_tolerance}）",
                    ))
        return issues

    def audit_future_price_leakage(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        horizons: Sequence[int] = (1, 5, 10),
    ) -> List[LeakageIssue]:
        """检测特征列是否等于未来第 h 期价格 close_{t+h}。"""
        issues: List[LeakageIssue] = []
        for h in horizons:
            future = close.shift(-h)
            for fcol in features.columns:
                if _near_equal(features[fcol], future, self.value_tolerance):
                    issues.append(LeakageIssue(
                        "future_price_leakage", fcol,
                        f"特征 {fcol} 等于未来第 {h} 期价格（close.shift(-{h})）",
                    ))
        return issues

    def audit_correlation_leakage(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        label_cols: Optional[Sequence[str]] = None,
    ) -> List[LeakageIssue]:
        """检测特征与未来标签的相关系数是否接近 ±1。"""
        issues: List[LeakageIssue] = []
        raw_cols = list(label_cols) if label_cols is not None else list(labels.columns)
        label_cols = _numeric_cols(labels, raw_cols)
        for lcol in label_cols:
            if lcol not in labels.columns:
                continue
            for fcol in features.columns:
                aligned = pd.concat([features[fcol], labels[lcol]], axis=1).dropna()
                if len(aligned) < 3:
                    continue
                a = aligned.iloc[:, 0].astype(float)
                b = aligned.iloc[:, 1].astype(float)
                if a.std() == 0 or b.std() == 0:  # 常数序列无相关性，跳过
                    continue
                corr = a.corr(b)
                if corr is not None and abs(corr) > self.corr_threshold:
                    issues.append(LeakageIssue(
                        "correlation_leakage", fcol,
                        f"特征 {fcol} 与未来标签 {lcol} 相关系数 {corr:.6f} 超过阈值 {self.corr_threshold}",
                    ))
        return issues

    # ------------------------------------------------------------------
    def audit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        close: Optional[pd.Series] = None,
        horizons: Sequence[int] = (1, 5, 10),
        label_cols: Optional[Sequence[str]] = None,
    ) -> LeakageReport:
        """执行全部泄漏检测，返回报告。"""
        issues: List[LeakageIssue] = []
        issues += self.audit_label_leakage(features, labels, label_cols)
        if close is not None:
            issues += self.audit_future_price_leakage(features, close, horizons)
        issues += self.audit_correlation_leakage(features, labels, label_cols)
        return LeakageReport(issues=issues)

    def audit_raise(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        close: Optional[pd.Series] = None,
        horizons: Sequence[int] = (1, 5, 10),
        label_cols: Optional[Sequence[str]] = None,
    ) -> LeakageReport:
        """审计并在发现泄漏时抛 LeakageError。"""
        report = self.audit(features, labels, close, horizons, label_cols)
        if report.has_leaks:
            details = "\n".join(f"- [{i.check}] {i.feature}: {i.detail}" for i in report.issues)
            raise LeakageError(f"检测到 {len(report.issues)} 处数据泄漏：\n{details}")
        return report


__all__ = [
    "LeakageAuditor",
    "LeakageReport",
    "LeakageIssue",
    "LeakageError",
]
