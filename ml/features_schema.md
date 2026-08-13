# 特征 / 标签 Schema（含"可用时点"）

约定：`t` 为当前 bar 时间戳。所有特征**只能使用 t 及之前**的数据；标签使用
**未来**数据，仅在 `t+h` 时点才可知。

## 标签（`ml.features.build_labels`，t → t+h）

| 列名 | 定义 | 可用时点 |
|---|---|---|
| `fwd_return` | `close_{t+h} / close_t - 1` | t+h（未来 h 期收益） |
| `fwd_direction` | `sign(fwd_return)` | t+h |
| `fwd_volatility` | `std(returns_{t+1..t+h})` | t+h（未来 h 期收益的标准差） |
| `label_asof` | `index_{t+h}` | 追溯标签可知时点的时间戳 |

最后 h 根 bar 无未来数据，标签为 NaN，训练前须 `dropna()`。

## 特征（`ml.features.build_features`，只用 t 及之前）

| 列名 | 定义 | 可用时点 |
|---|---|---|
| `momentum_{n}` | `close_t / close_{t-n} - 1` | t（用 t 与 t-n 收盘，均为历史） |
| `volatility_{n}` | `std(r_{t-n+1..t})` | t（滚动窗口历史波动） |
| `volume_ratio_{n}` | `volume_t / mean(volume_{t-n+1..t})` | t |
| `ma_ratio_{n}` | `close_t / mean(close_{t-n+1..t})` | t |
| `intraday_return` | `close_t / open_t - 1` | t（当日盘内） |
| `range_pct` | `(high_t - low_t) / close_t` | t（当日盘内） |

## 预处理（防泄漏）

- `rolling_standardize(df, window)`：`z_t = (x_t - mean_{window}) / std_{window}`，
  均值/标准差只在滚动窗口内（样本内）计算，**不用全样本**。
- `clip_rolling(df, window, n_std)`：point-in-time 异常值裁剪（滚动均值 ± n_std×滚动标准差）。
- `winsorize(df, ...)`：全样本分位裁剪，**会前视，仅供 EDA，禁止进入训练流水线**。
- `fill_missing(df)`：前向填充（停牌沿用最近值），剩余补 0。

## 版本化存储

`FeatureStore.save(df, name, version)` → `data_cache/features/{name}/v{version}.parquet`
（附 `_meta.json` 记录列、形状、时间）。`load(name, version)` / `latest_version(name)` 读取。
