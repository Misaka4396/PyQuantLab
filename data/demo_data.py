"""生成小样本演示数据（优先 akshare，网络失败自动回退合成数据）。

用法：``python -m data.demo_data``（在仓库根目录运行）。

生成内容（写入 DataLoader 的 data_root，默认 ./data_cache/pit）：
- ohlcv/：2 只 A 股 ETF 日线（akshare 实盘）+ 若干成分股合成日线
- adj_factor/：ETF 复权因子（合成除权事件）
- constituents/：指数成分股会员记录（实盘快照 + 合成调进调出/退市历史）
- nav/：ETF 净值/参考净值（合成）
- pcf/：ETF 申赎篮子文件（合成，预留 schema）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data import schemas as sc
from data.data_loader import DEFAULT_DATA_ROOT, DataLoader

ETF_CODES = ["510300", "510500"]  # 沪深300ETF / 中证500ETF
INDEX_CODE = "000300"  # 沪深300 指数
N_CONSTITUENTS = 6  # 成分股演示数量
DELISTED_SYMBOL = "000888"  # 虚构退市股（演示幸存者偏差）


# ---------------------------------------------------------------------------
# akshare 抓取
# ---------------------------------------------------------------------------
def fetch_etf_daily_akshare(
    code: str, start: str, end: str, retries: int = 2
) -> pd.DataFrame | None:
    """用 akshare 拉取 ETF 日线，转换为本数据层标准长表。失败返回 None（由调用方回退合成）。"""
    import time

    import akshare as ak

    for attempt in range(retries):
        try:
            raw = ak.fund_etf_hist_em(
                symbol=code, period="daily", start_date=start, end_date=end, adjust=""
            )
            if raw is None or raw.empty:
                return None
            df = pd.DataFrame(
                {
                    sc.COL_OPEN: pd.to_numeric(raw["开盘"], errors="coerce"),
                    sc.COL_HIGH: pd.to_numeric(raw["最高"], errors="coerce"),
                    sc.COL_LOW: pd.to_numeric(raw["最低"], errors="coerce"),
                    sc.COL_CLOSE: pd.to_numeric(raw["收盘"], errors="coerce"),
                    sc.COL_VOLUME: pd.to_numeric(raw["成交量"], errors="coerce"),
                    sc.COL_AMOUNT: pd.to_numeric(raw["成交额"], errors="coerce"),
                }
            )
            df.index = pd.to_datetime(raw["日期"])
            df.index.name = sc.COL_DATETIME
            df[sc.COL_AS_OF] = df.index  # 日线：收盘后可知
            return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def fetch_constituents_akshare(index_code: str, retries: int = 2) -> list[str] | None:
    """用 akshare 拉取指数最新成分股代码列表。失败返回 None。"""
    import time

    import akshare as ak

    for attempt in range(retries):
        try:
            raw = ak.index_stock_cons_csindex(symbol=index_code)
            if raw is None or raw.empty:
                return None
            return [str(c).zfill(6) for c in raw["成分券代码"].tolist()]
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def _synthetic_ohlcv(index: pd.DatetimeIndex, start_price: float, seed: int) -> pd.DataFrame:
    """随机游走生成合成 OHLCV（index 为交易日）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, len(index))
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0.0, 0.005, len(index)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.005, len(index))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.005, len(index))))
    volume = rng.integers(1_000_000, 10_000_000, len(index)).astype(float)
    df = pd.DataFrame(
        {
            sc.COL_OPEN: open_,
            sc.COL_HIGH: high,
            sc.COL_LOW: low,
            sc.COL_CLOSE: close,
            sc.COL_VOLUME: volume,
            sc.COL_AMOUNT: volume * close,
            sc.COL_AS_OF: index,
        },
        index=index,
    )
    df.index.name = sc.COL_DATETIME
    return df


def _synthetic_membership(
    codes: list[str],
    start: pd.Timestamp,
    mid: pd.Timestamp,
) -> pd.DataFrame:
    """由当前成分股快照合成会员生命周期（含调进调出与退市）。"""
    rows = []
    for i, code in enumerate(codes):
        if i == 1:
            # 第 2 只：中途调出（演示调出）
            rows.append(
                {
                    sc.COL_SYMBOL: code,
                    sc.COL_ENTRY_DATE: start,
                    sc.COL_EXIT_DATE: mid,
                    sc.COL_AS_OF: start,
                    sc.COL_REASON: sc.REASON_REMOVE,
                }
            )
        else:
            rows.append(
                {
                    sc.COL_SYMBOL: code,
                    sc.COL_ENTRY_DATE: start,
                    sc.COL_EXIT_DATE: pd.NaT,
                    sc.COL_AS_OF: start,
                    sc.COL_REASON: sc.REASON_INITIAL,
                }
            )
    # 一只虚构退市股：历史存在但当前快照没有（幸存者偏差核心演示）
    rows.append(
        {
            sc.COL_SYMBOL: DELISTED_SYMBOL,
            sc.COL_ENTRY_DATE: start,
            sc.COL_EXIT_DATE: mid,
            sc.COL_AS_OF: start,
            sc.COL_REASON: sc.REASON_DELIST,
        }
    )
    # 一只中途调入
    rows.append(
        {
            sc.COL_SYMBOL: "601318",
            sc.COL_ENTRY_DATE: mid,
            sc.COL_EXIT_DATE: pd.NaT,
            sc.COL_AS_OF: mid,
            sc.COL_REASON: sc.REASON_ADD,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def make_demo_data(
    data_root: Path | None = None,
    use_akshare: bool = True,
    periods: int = 60,
) -> dict:
    """生成全套演示数据，返回含来源说明的汇总 dict。"""
    loader = DataLoader(data_root or DEFAULT_DATA_ROOT)
    end = pd.Timestamp(datetime.now().date())
    start = end - pd.Timedelta(days=int(periods * 1.8))
    trading_index = pd.bdate_range(start=start, end=end)
    mid = trading_index[len(trading_index) // 2]

    summary: dict = {"data_root": str(loader.data_root), "sources": {}}

    # 1) ETF 行情：akshare 优先，回退合成
    for i, code in enumerate(ETF_CODES):
        df = (
            fetch_etf_daily_akshare(code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
            if use_akshare
            else None
        )
        if df is None or df.empty:
            df = _synthetic_ohlcv(trading_index, start_price=3.0 + i, seed=100 + i)
            summary["sources"][code] = "synthetic"
        else:
            summary["sources"][code] = "akshare"
        loader.save_ohlcv(code, df)

    # 2) 成分股会员记录：实盘快照 + 合成历史
    codes = fetch_constituents_akshare(INDEX_CODE) if use_akshare else None
    if not codes:
        codes = [f"600{i:03d}" for i in range(N_CONSTITUENTS)]
        summary["sources"]["constituents"] = "synthetic"
    else:
        codes = codes[:N_CONSTITUENTS]
        summary["sources"]["constituents"] = "akshare(快照)+synthetic(历史)"
    membership = _synthetic_membership(codes, start, mid)
    loader.save_constituents(INDEX_CODE, membership)

    # 3) 成分股行情（合成）
    all_codes = membership[sc.COL_SYMBOL].unique().tolist()
    for i, code in enumerate(all_codes):
        loader.save_ohlcv(code, _synthetic_ohlcv(trading_index, start_price=5.0 + i, seed=200 + i))

    # 4) 复权因子（合成一次除权事件）
    loader.save_adj_factor(
        "510300",
        pd.DataFrame(
            {
                sc.COL_EX_DATE: [mid],
                sc.COL_RATIO: [0.8],
            }
        ),
    )

    # 5) NAV / IOPV（合成）
    etf = loader.load_ohlcv("510300", clean=False)
    nav = pd.DataFrame(
        {
            sc.COL_NAV: etf[sc.COL_CLOSE] * 0.99,
            sc.COL_IOPV: etf[sc.COL_CLOSE] * 1.0,
            sc.COL_AS_OF: etf.index,
        }
    )
    loader.save_nav("510300", nav)

    # 6) PCF（合成，预留 schema）
    pcf = pd.DataFrame(
        {
            sc.COL_SYMBOL: all_codes[:3],
            sc.COL_QUANTITY: [1000, 2000, 1500],
            sc.COL_CASH_COMPONENT: [0.0, 0.0, 0.0],
            sc.COL_CREATION_UNIT: [500_000, 500_000, 500_000],
        }
    )
    loader.save_pcf("510300", mid.strftime("%Y%m%d"), pcf)

    summary["universe"] = {
        "as_of_start": loader.get_tradeable_universe(INDEX_CODE, start),
        "as_of_mid": loader.get_tradeable_universe(INDEX_CODE, mid + pd.Timedelta(days=1)),
        "all_historical": sorted(membership[sc.COL_SYMBOL].unique().tolist()),
    }
    return summary


if __name__ == "__main__":
    result = make_demo_data(use_akshare=True)
    print("演示数据已生成：")
    print(f"  data_root = {result['data_root']}")
    for k, v in result["sources"].items():
        print(f"  {k}: {v}")
    print("  时点全集示例：")
    for k, v in result["universe"].items():
        print(f"    {k}: {v}")
