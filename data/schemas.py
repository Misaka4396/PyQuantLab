"""parquet 表 schema 定义与列名常量（与 schemas.md 保持一致）。

集中管理各数据表的列名，避免多模块间列名字符串漂移。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# OHLCV 行情表（长表，index=datetime）
# ---------------------------------------------------------------------------
COL_DATETIME = "datetime"          # bar 时间戳（索引）
COL_SYMBOL = "symbol"              # 证券代码
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"              # 成交量
COL_AMOUNT = "amount"              # 成交额
COL_AS_OF = "as_of"                # point-in-time：该条数据可被观察到的时间

# 清洗后新增的派生列（不落盘）
COL_IS_SUSPENDED = "is_suspended"  # 是否停牌
COL_LIMIT_STATUS = "limit_status"  # 涨跌停状态

OHLCV_REQUIRED = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]
OHLCV_OPTIONAL = [COL_VOLUME, COL_AMOUNT]
OHLCV_STORED_COLUMNS = [COL_SYMBOL] + OHLCV_REQUIRED + OHLCV_OPTIONAL + [COL_AS_OF]

# 涨跌停状态枚举
LIMIT_NORMAL = "normal"
LIMIT_UP = "limit_up"
LIMIT_DOWN = "limit_down"
SUSPENDED = "suspended"

# ---------------------------------------------------------------------------
# 复权因子表（index=ex_date 除权除息生效日）
# ---------------------------------------------------------------------------
COL_EX_DATE = "ex_date"            # 除权除息生效日（索引）
COL_RATIO = "ratio"                # 本次除权比例（新价/旧价）
COL_ADJ_FACTOR = "adj_factor"      # 累计复权因子（ratio 的 cumprod）

ADJ_FACTOR_STORED_COLUMNS = [COL_RATIO, COL_ADJ_FACTOR, COL_AS_OF]

# ---------------------------------------------------------------------------
# 成分股表（幸存者偏差，index 无，普通列）
# ---------------------------------------------------------------------------
COL_INDEX_CODE = "index_code"      # 所属指数/ETF 代码
COL_ENTRY_DATE = "entry_date"      # 调入生效日
COL_EXIT_DATE = "exit_date"        # 调出/退市生效日（NaT = 仍有效）
COL_REASON = "reason"              # add / remove / delist / initial

CONSTITUENTS_STORED_COLUMNS = [
    COL_INDEX_CODE, COL_SYMBOL, COL_ENTRY_DATE, COL_EXIT_DATE, COL_AS_OF, COL_REASON,
]

# 调进调出原因枚举
REASON_ADD = "add"
REASON_REMOVE = "remove"
REASON_DELIST = "delist"
REASON_INITIAL = "initial"

# ---------------------------------------------------------------------------
# PCF 篮子文件（ETF 套利用，A 底座仅预留 schema，抓取在 B1 完成）
# ---------------------------------------------------------------------------
COL_ETF_CODE = "etf_code"          # ETF 代码
COL_TRADE_DATE = "trade_date"      # 申赎清单交易日
COL_QUANTITY = "quantity"          # 篮子中该成分股数量
COL_CASH_COMPONENT = "cash_component"  # 现金差额
COL_CREATION_UNIT = "creation_unit"    # 最小申赎单位（份）

PCF_STORED_COLUMNS = [
    COL_ETF_CODE, COL_TRADE_DATE, COL_SYMBOL, COL_QUANTITY,
    COL_CASH_COMPONENT, COL_CREATION_UNIT, COL_AS_OF,
]

# ---------------------------------------------------------------------------
# NAV / IOPV 表（index=datetime）
# ---------------------------------------------------------------------------
COL_NAV = "nav"                    # 基金净值
COL_IOPV = "iopv"                  # 盘中参考净值

NAV_STORED_COLUMNS = [COL_ETF_CODE, COL_NAV, COL_IOPV, COL_AS_OF]
