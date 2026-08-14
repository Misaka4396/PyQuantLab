"""B1 PCF 篮子文件解析（申赎清单）。

PCF（Portfolio Composition File）是 ETF 每日盘前公布的申赎清单，含成分股代码、
数量、现金差额、最小申赎单位，以及现金替代标志。基金公司公开的 PCF 文本格式
各家略有差异，本模块定义一套**标准文本接口**，同时兼容最简 CSV：

标准格式（key=value 头部 + [成分股] 表格）：:

    ETF代码=510300
    清单日期=20240102
    生效日期=20240103
    最小申赎单位=900000
    现金差额=1234.56
    [成分股]
    证券代码,证券名称,数量,现金替代标志,前收盘价
    600000,浦发银行,1000,允许,7.20
    ...

关键约定：
- ``trade_date``（清单交易日，T 日）与 ``effective_date``（生效日，T+1）分离，
  防止 T 日 PCF 串期：T 日只能用 T-1 日公布的清单（次日生效）。
- 权重按"数量 × 前收盘价"价值加权归一（无价格时按数量归一），保证权重和 = 1。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.exceptions import DataError
from data import schemas as sc

DateLike = str | pd.Timestamp

# 现金替代标志枚举
CS_ALLOWED = "允许"  # 允许现金替代
CS_REQUIRED = "必须"  # 必须现金替代
CS_FORBIDDEN = "禁止"  # 禁止现金替代（必须用股票）


@dataclass
class PCFConstituent:
    """PCF 中的单只成分股。"""

    symbol: str
    name: str = ""
    quantity: float = 0.0
    cash_substitute: str = CS_ALLOWED
    price: float | None = None  # 前收盘价（估算 IOPV/权重用，可选）


@dataclass
class PCFBasket:
    """一只 ETF 在某交易日的完整申赎清单。"""

    etf_code: str
    trade_date: pd.Timestamp  # 清单交易日（T 日）
    effective_date: pd.Timestamp  # 生效日（T+1）
    creation_unit: float  # 最小申赎单位（份）
    cash_component: float  # 现金差额
    constituents: list[PCFConstituent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 派生
    # ------------------------------------------------------------------
    def weights(self) -> dict[str, float]:
        """成分股权重（价值加权 = 数量×前收盘价，无价格时退化为数量加权）。

        归一化保证权重之和 = 1。
        """
        vals: dict[str, float] = {}
        for c in self.constituents:
            if c.quantity <= 0:
                continue
            if c.price is not None and c.price > 0:
                vals[c.symbol] = c.quantity * c.price
            else:
                vals[c.symbol] = c.quantity
        total = sum(vals.values())
        if total <= 0:
            return {}
        return {s: v / total for s, v in vals.items()}

    def validate_weight_sum(self, tolerance: float = 0.005) -> bool:
        """校验篮子权重之和是否等于 1（±tolerance 容差）。"""
        w = self.weights()
        if not w:
            return False
        return abs(sum(w.values()) - 1.0) <= tolerance

    def to_dataframe(self) -> pd.DataFrame:
        """成分股明细 DataFrame（含 symbol/name/quantity/cash_substitute/price）。"""
        return pd.DataFrame(
            [
                {
                    "symbol": c.symbol,
                    "name": c.name,
                    "quantity": c.quantity,
                    "cash_substitute": c.cash_substitute,
                    "price": c.price,
                }
                for c in self.constituents
            ]
        )

    def to_pcf_dataframe(self) -> pd.DataFrame:
        """转为 A1 DataLoader.save_pcf 可消费的列（symbol/quantity + 现金差额/申赎单位）。"""
        df = self.to_dataframe()
        df[sc.COL_CASH_COMPONENT] = self.cash_component
        df[sc.COL_CREATION_UNIT] = self.creation_unit
        return df


# ---------------------------------------------------------------------------
# 交易日
# ---------------------------------------------------------------------------
def next_trading_day(ts: DateLike, calendar: Iterable[DateLike] | None = None) -> pd.Timestamp:
    """返回 ts 的下一个交易日。

    - calendar 提供时（升序交易日序列）：返回其中严格大于 ts 的第一个日期。
    - 否则用 pandas 工作日偏移（BDay），**不处理法定节假日**，生产环境应传入
      真实交易日历（如 akshare trade_cal）。
    """
    t = pd.Timestamp(ts)
    if calendar is not None:
        days = sorted(pd.Timestamp(d) for d in calendar)
        for d in days:
            if d > t:
                return d
        raise DataError(f"交易日历中找不到 {t} 之后的交易日")
    return t + pd.offsets.BDay(1)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _parse_header(lines: list[str]) -> dict[str, str]:
    """解析 key=value 头部（跳过注释/空行）。"""
    header: dict[str, str] = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            header[k.strip()] = v.strip()
    return header


def parse_pcf(text: str) -> PCFBasket:
    """解析 PCF 文本（标准格式或纯 CSV），返回 PCFBasket。"""
    lines = text.splitlines()
    header = _parse_header([ln for ln in lines if "[成分股]" not in ln])

    etf_code = header.get("ETF代码") or header.get("etf_code") or ""
    trade_date = header.get("清单日期") or header.get("trade_date") or ""
    effective = header.get("生效日期") or header.get("effective_date") or ""
    creation_unit = float(header.get("最小申赎单位") or header.get("creation_unit") or 0.0)
    cash_component = float(header.get("现金差额") or header.get("cash_component") or 0.0)

    if not etf_code or not trade_date:
        raise DataError("PCF 缺少 ETF代码/清单日期（header 需含 key=value）")

    tdate = pd.Timestamp(trade_date)
    edate = pd.Timestamp(effective) if effective else next_trading_day(tdate)

    constituents = _parse_constituents(lines)
    return PCFBasket(
        etf_code=etf_code,
        trade_date=tdate,
        effective_date=edate,
        creation_unit=creation_unit,
        cash_component=cash_component,
        constituents=constituents,
    )


def _parse_constituents(lines: list[str]) -> list[PCFConstituent]:
    """解析成分股表格（[成分股] 段或纯 CSV 表）。"""
    rows: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if "=" in s:
            continue  # header 行
        cells = [c.strip() for c in s.split(",")]
        if cells:
            rows.append(cells)

    if not rows:
        return []

    # 第一行若是表头则跳过
    header_row = rows[0]
    if header_row and header_row[0] in ("证券代码", "symbol", "代码"):
        rows = rows[1:]

    out: list[PCFConstituent] = []
    for cells in rows:
        if not cells or not cells[0]:
            continue
        symbol = str(cells[0]).zfill(6)
        name = cells[1] if len(cells) > 1 else ""
        quantity = float(cells[2]) if len(cells) > 2 and cells[2] else 0.0
        cash_sub = cells[3] if len(cells) > 3 and cells[3] else CS_ALLOWED
        price = None
        if len(cells) > 4 and cells[4]:
            try:
                price = float(cells[4])
            except ValueError:
                price = None
        out.append(
            PCFConstituent(
                symbol=symbol, name=name, quantity=quantity, cash_substitute=cash_sub, price=price
            )
        )
    return out


def parse_pcf_file(path: str | Path) -> PCFBasket:
    """读取 PCF 文件并解析。"""
    p = Path(path)
    if not p.exists():
        raise DataError(f"PCF 文件不存在: {p}")
    return parse_pcf(p.read_text(encoding="utf-8"))


def pcf_from_dataframe(
    df: pd.DataFrame,
    etf_code: str,
    trade_date: DateLike,
    effective_date: DateLike | None = None,
    creation_unit: float = 0.0,
    cash_component: float = 0.0,
    calendar: Iterable[DateLike] | None = None,
) -> PCFBasket:
    """由 DataFrame（A1 pcf 表 schema 或含 symbol/quantity 的任意表）构造 PCFBasket。"""
    df = df.copy()
    for c in (sc.COL_SYMBOL, sc.COL_QUANTITY):
        if c not in df.columns:
            raise DataError(f"构造 PCF 缺少列: {c}")
    tdate = pd.Timestamp(trade_date)
    edate = (
        pd.Timestamp(effective_date)
        if effective_date is not None
        else next_trading_day(tdate, calendar)
    )
    constituents = []
    for _, row in df.iterrows():
        constituents.append(
            PCFConstituent(
                symbol=str(row[sc.COL_SYMBOL]).zfill(6),
                name=str(row.get("name", "")),
                quantity=float(row[sc.COL_QUANTITY]),
                cash_substitute=str(row.get("cash_substitute", CS_ALLOWED)),
                price=float(row["price"]) if ("price" in row and pd.notna(row["price"])) else None,
            )
        )
    return PCFBasket(
        etf_code=etf_code,
        trade_date=tdate,
        effective_date=edate,
        creation_unit=float(creation_unit),
        cash_component=float(cash_component),
        constituents=constituents,
    )


__all__ = [
    "CS_ALLOWED",
    "CS_FORBIDDEN",
    "CS_REQUIRED",
    "PCFBasket",
    "PCFConstituent",
    "next_trading_day",
    "parse_pcf",
    "parse_pcf_file",
    "pcf_from_dataframe",
]
