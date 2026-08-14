"""C1 特征工程与标注流水线（严格 point-in-time）。

设计目标：
- **标注**：未来收益/方向/波动率，label 与特征严格对齐 t→t+h（label 在 t 时点不可知，
  仅 t+h 才可知，用 ``label_asof`` 列追溯）。
- **特征**：动量/波动/成交量/均线/微观结构，全部只用 t 及之前数据。
- **防泄漏**：滚动标准化在滚动窗口内计算（不用全样本）；异常值裁剪提供
  point-in-time 的滚动裁剪与全样本分位裁剪（后者仅供 EDA，明确标注会前视）。
- **存储**：parquet + 版本化（FeatureStore）。

每个特征的"可用时点"见 ``ml/features_schema.md``。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import schemas as sc

# 特征列名
FEAT_MOMENTUM = "momentum_{n}"
FEAT_VOLATILITY = "volatility_{n}"
FEAT_VOLUME_RATIO = "volume_ratio_{n}"
FEAT_MA_RATIO = "ma_ratio_{n}"
FEAT_INTRADAY_RET = "intraday_return"
FEAT_RANGE = "range_pct"

# 标注列名
LABEL_FWD_RETURN = "fwd_return"
LABEL_FWD_DIRECTION = "fwd_direction"
LABEL_FWD_VOLATILITY = "fwd_volatility"
LABEL_ASOF = "label_asof"

DEFAULT_FEATURE_ROOT = Path("./data_cache/features")


# ---------------------------------------------------------------------------
# 标注：未来收益 / 方向 / 波动率（t → t+h）
# ---------------------------------------------------------------------------
def build_labels(prices: pd.Series, horizon: int) -> pd.DataFrame:
    """构造未来标签，严格对齐 t→t+h。

    - fwd_return_t    = close_{t+h} / close_t - 1        （t+h 才可知）
    - fwd_direction_t = sign(fwd_return_t)
    - fwd_volatility_t = std(returns_{t+1..t+h})          （未来 h 期收益的波动）
    - label_asof_t     = index_{t+h}                      （标签可追溯的可知时点）

    最后 horizon 根 bar 无未来数据，标签为 NaN（调用方应 dropna 后再训练）。
    """
    close = pd.Series(prices).astype(float)
    ret = close.pct_change(fill_method=None)

    fwd_return = close.shift(-horizon) / close - 1.0
    fwd_direction = np.sign(fwd_return)

    # 未来 h 期收益波动：对收益率序列反向 rolling(h).std() 再反向，得到
    # std(r_t..r_{t+h-1})，再 shift(-1) 得到 std(r_{t+1}..r_{t+h})。
    rev_std = ret.iloc[::-1].rolling(horizon, min_periods=horizon).std().iloc[::-1]
    fwd_volatility = rev_std.shift(-1)

    # label 可知时点 = 未来 h 根 bar 的时间戳
    idx_series = pd.Series(close.index, index=close.index)
    label_asof = idx_series.shift(-horizon)

    df = pd.DataFrame(
        {
            LABEL_FWD_RETURN: fwd_return,
            LABEL_FWD_DIRECTION: fwd_direction,
            LABEL_FWD_VOLATILITY: fwd_volatility,
            LABEL_ASOF: label_asof,
        },
        index=close.index,
    )
    df.attrs["horizon"] = int(horizon)
    return df


# ---------------------------------------------------------------------------
# 特征：动量 / 波动 / 成交量 / 均线 / 微观结构（只用 t 及之前数据）
# ---------------------------------------------------------------------------
def build_features(
    ohlcv: pd.DataFrame,
    momentum_windows: Sequence[int] = (5, 20),
    volatility_windows: Sequence[int] = (10, 20),
    volume_windows: Sequence[int] = (5, 20),
    ma_windows: Sequence[int] = (5, 20),
) -> pd.DataFrame:
    """由 OHLCV（open/high/low/close/volume，index=datetime）生成特征。

    全部特征只依赖 t 及之前数据：
    - 动量 momentum_n = close_t / close_{t-n} - 1
    - 波动 volatility_n = std(r_{t-n+1..t})（滚动窗口）
    - 成交量 volume_ratio_n = volume_t / mean(volume_{t-n+1..t})
    - 均线 ma_ratio_n = close_t / mean(close_{t-n+1..t})
    - 微观结构 intraday_return = close/open - 1；range_pct = (high-low)/close
    """
    df = ohlcv.copy()
    if sc.COL_DATETIME in df.columns:
        df = df.set_index(sc.COL_DATETIME)
    close = pd.to_numeric(df[sc.COL_CLOSE], errors="coerce")
    open_ = pd.to_numeric(df[sc.COL_OPEN], errors="coerce")
    high = pd.to_numeric(df[sc.COL_HIGH], errors="coerce")
    low = pd.to_numeric(df[sc.COL_LOW], errors="coerce")
    volume = pd.to_numeric(
        df.get(sc.COL_VOLUME, pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    ret = close.pct_change(fill_method=None)

    out = pd.DataFrame(index=df.index)

    for n in momentum_windows:
        out[FEAT_MOMENTUM.format(n=n)] = close / close.shift(n) - 1.0
    for n in volatility_windows:
        out[FEAT_VOLATILITY.format(n=n)] = ret.rolling(n).std()
    for n in volume_windows:
        out[FEAT_VOLUME_RATIO.format(n=n)] = volume / volume.rolling(n).mean()
    for n in ma_windows:
        out[FEAT_MA_RATIO.format(n=n)] = close / close.rolling(n).mean()
    out[FEAT_INTRADAY_RET] = close / open_ - 1.0
    out[FEAT_RANGE] = (high - low) / close

    out.attrs["feature_schema"] = "ml/features_schema.md"
    return out


# ---------------------------------------------------------------------------
# 防泄漏预处理
# ---------------------------------------------------------------------------
def rolling_standardize(
    df: pd.DataFrame,
    window: int,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """滚动标准化：z_t = (x_t - mean(x_{t-w+1..t})) / std(x_{t-w+1..t})。

    只在滚动窗口内计算均值/标准差（sample 内），不使用全样本 → 无前视。
    """
    cols = list(columns) if columns is not None else list(df.columns)
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        mu = x.rolling(window).mean()
        sd = x.rolling(window).std()
        out[c] = (x - mu) / sd.replace(0, np.nan)
    return out


def clip_rolling(
    df: pd.DataFrame,
    window: int,
    n_std: float = 3.0,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """point-in-time 异常值裁剪：以滚动均值 ± n_std*滚动标准差为界 clip（无前视）。"""
    cols = list(columns) if columns is not None else list(df.columns)
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        mu = x.rolling(window).mean()
        sd = x.rolling(window).std()
        lower = mu - n_std * sd
        upper = mu + n_std * sd
        out[c] = x.clip(lower=lower, upper=upper)
    return out


def winsorize(
    df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """全样本分位裁剪（Winsorize）。

    **注意**：分位数来自全样本，属"用了未来信息"，仅供探索性分析（EDA），
    严禁在训练/回测流水线中使用；生产请用 ``clip_rolling``（point-in-time）。
    """
    cols = list(columns) if columns is not None else list(df.columns)
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        lo, hi = x.quantile(lower_quantile), x.quantile(upper_quantile)
        out[c] = x.clip(lower=lo, upper=hi)
    return out


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """缺失处理：前向填充（停牌沿用最近值），仍缺失补 0。"""
    out = df.copy()
    if method == "ffill":
        out = out.ffill()
    elif method == "zero":
        pass
    else:
        raise ValueError(f"未知缺失处理方法: {method}")
    return out.fillna(0.0)


def encode_categorical(
    df: pd.DataFrame,
    column: str,
    drop_first: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """类别编码（one-hot）。"""
    dummies = pd.get_dummies(df[column], prefix=prefix or column, drop_first=drop_first)
    return pd.concat([df.drop(columns=[column]), dummies], axis=1)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def assemble(
    ohlcv: pd.DataFrame,
    horizon: int,
    standardize_window: int | None = None,
    momentum_windows: Sequence[int] = (5, 20),
    volatility_windows: Sequence[int] = (10, 20),
    volume_windows: Sequence[int] = (5, 20),
    ma_windows: Sequence[int] = (5, 20),
) -> pd.DataFrame:
    """端到端组装特征 + 标签（内连接对齐，可选滚动标准化）。"""
    feats = build_features(ohlcv, momentum_windows, volatility_windows, volume_windows, ma_windows)
    labels = build_labels(ohlcv[sc.COL_CLOSE], horizon)
    if standardize_window is not None:
        feats = rolling_standardize(feats, standardize_window)
    df = pd.concat([feats, labels], axis=1, join="inner")
    # 丢弃标签为 NaN 的行（最后 horizon 根 bar 无未来数据，无法训练）
    df = df.dropna(subset=[LABEL_FWD_RETURN, LABEL_FWD_DIRECTION, LABEL_FWD_VOLATILITY])
    return df


# ---------------------------------------------------------------------------
# 特征存储（parquet + 版本化）
# ---------------------------------------------------------------------------
@dataclass
class FeatureStore:
    """特征集版本化存储：root/{name}/v{version}.parquet + _meta.json。"""

    root: str | Path = DEFAULT_FEATURE_ROOT

    def __post_init__(self):
        self.root = Path(self.root)

    def _dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, df: pd.DataFrame, name: str, version: int) -> Path:
        """落盘特征集，写 parquet + 元数据（版本/时间/列/形状）。"""
        d = self._dir(name)
        path = d / f"v{version}.parquet"
        df.to_parquet(path, index=True)
        meta = {
            "name": name,
            "version": int(version),
            "columns": list(df.columns),
            "shape": list(df.shape),
            "index_name": str(df.index.name),
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        (d / "_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load(self, name: str, version: int) -> pd.DataFrame:
        """读取指定版本特征集。"""
        path = self._dir(name) / f"v{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"特征集不存在: {path}")
        return pd.read_parquet(path)

    def latest_version(self, name: str) -> int | None:
        """返回某特征集的最新版本号（无则 None）。"""
        d = self.root / name
        if not d.exists():
            return None
        versions = [int(p.stem[1:]) for p in d.glob("v*.parquet") if p.stem[1:].isdigit()]
        return max(versions) if versions else None


__all__ = [
    "LABEL_ASOF",
    "LABEL_FWD_DIRECTION",
    "LABEL_FWD_RETURN",
    "LABEL_FWD_VOLATILITY",
    "FeatureStore",
    "assemble",
    "build_features",
    "build_labels",
    "clip_rolling",
    "encode_categorical",
    "fill_missing",
    "rolling_standardize",
    "winsorize",
]
