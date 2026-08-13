# Parquet 数据层 Schema 文档

数据层根目录默认 `./data_cache/pit`（可通过 `DataLoader(data_root=...)` 覆盖）。
所有表均为 Apache Parquet 格式，时间戳一律为 **tz-naive** 的 `datetime64[ns]`。
与代码中的列名常量保持一致（见 `data/schemas.py`）。

## 通用约定

- `as_of`（point-in-time 时间戳）：该条数据"可被观察到"的时间。回测在时点 T 只能读取
  `as_of <= T` 的记录，杜绝未来函数。
- 落盘数据为"原始已归一化"数据；清洗（缺失/异常/停牌/涨跌停）在读取时进行。

---

## 1. OHLCV 行情表

- 路径：`ohlcv/{symbol}.parquet`（每只证券一个文件）
- 索引：`datetime`（bar 时间戳，分钟级为主，也支持日线）

| 列名 | 类型 | 说明 |
|------|------|------|
| `open` | float64 | 开盘价 |
| `high` | float64 | 最高价 |
| `low` | float64 | 最低价 |
| `close` | float64 | 收盘价 |
| `volume` | float64 | 成交量 |
| `amount` | float64 | 成交额（可空，补 0） |
| `symbol` | str | 证券代码（冗余存储，便于纵向拼接） |
| `as_of` | datetime64[ns] | 该 bar 可被观察到的时间 |

清洗后新增的派生列（**不落盘**）：`is_suspended`（bool）、`limit_status`
（str：`normal` / `limit_up` / `limit_down` / `suspended`）。

## 2. 复权因子表

- 路径：`adj_factor/{symbol}.parquet`
- 索引：`ex_date`（除权除息生效日）

| 列名 | 类型 | 说明 |
|------|------|------|
| `ratio` | float64 | 本次除权比例（新价/旧价，例如 10 送 10 = 0.5） |
| `adj_factor` | float64 | 累计复权因子（ratio 的 cumprod，除权前 = 1） |
| `as_of` | datetime64[ns] | 该因子可被得知的时间（默认 = ex_date） |
| `symbol` | str | 证券代码 |

前复权公式：`qfq_t = close_t * f_end / f_t`；后复权：`hfq_t = close_t / f_t`。
其中 `f_t` 为截至 t 的累计因子，`f_end` 为最后一个因子的值。

## 3. 成分股表（幸存者偏差）

- 路径：`constituents/{index_code}.parquet`
- 无索引（普通行式表，`index=False` 写出）

| 列名 | 类型 | 说明 |
|------|------|------|
| `index_code` | str | 所属指数/ETF 代码 |
| `symbol` | str | 证券代码 |
| `entry_date` | datetime64[ns] | 调入生效日 |
| `exit_date` | datetime64[ns] | 调出/退市生效日（NaT = 仍有效） |
| `as_of` | datetime64[ns] | 该会员记录可被得知的时间（公告日） |
| `reason` | str | `add` / `remove` / `delist` / `initial` |

"当时可交易全集" = 满足 `entry_date <= T`、`exit_date > T`（或为空）、`as_of <= T` 的 symbol 集合。

## 4. PCF 篮子文件（ETF 套利用，预留 schema）

> 本批次仅定义 schema 与加载接口，实际抓取在 B1 完成。

- 路径：`pcf/{etf_code}_{trade_date}.parquet`
- 无索引（行式表）

| 列名 | 类型 | 说明 |
|------|------|------|
| `etf_code` | str | ETF 代码 |
| `trade_date` | datetime64[ns] | 申赎清单交易日 |
| `symbol` | str | 篮子成分股代码 |
| `quantity` | float64 | 篮子中该成分股数量 |
| `cash_component` | float64 | 现金差额 |
| `creation_unit` | float64 | 最小申赎单位（份） |
| `as_of` | datetime64[ns] | PCF 文件发布时间 |

## 5. NAV / IOPV 表

- 路径：`nav/{etf_code}.parquet`
- 索引：`datetime`

| 列名 | 类型 | 说明 |
|------|------|------|
| `etf_code` | str | ETF 代码 |
| `nav` | float64 | 基金净值 |
| `iopv` | float64 | 盘中参考净值（估算） |
| `as_of` | datetime64[ns] | 该净值可被观察到的时间 |
