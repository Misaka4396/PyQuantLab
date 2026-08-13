"""数据质量报告：统计缺失率、异常率、停牌/涨跌停占比，并输出 markdown。

质量指标基于"原始（未清洗）"数据计算，避免清洗掩盖数据问题：
- missing_rate   ：OHLCV 各列缺失值占比（整体）
- anomaly_rate   ：存在至少一种异常（非正价 / high<low / close 越界 / 单根极端收益）的行占比
- suspension_rate：停牌（成交量=0）占比
- limit_up/down  ：涨跌停 bar 计数
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data import schemas as sc
from data.data_loader import DEFAULT_DATA_ROOT, DataLoader, clean_ohlcv

PRICE_COLS = [sc.COL_OPEN, sc.COL_HIGH, sc.COL_LOW, sc.COL_CLOSE]
ALL_COLS = PRICE_COLS + sc.OHLCV_OPTIONAL


class QualityReporter:
    """数据质量检查器，基于 DataLoader 读取的原始数据生成质量报告。"""

    def __init__(self, data_root: Union[str, Path] = DEFAULT_DATA_ROOT, limit_pct: float = 0.10):
        self.loader = DataLoader(data_root)
        self.limit_pct = limit_pct

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def inspect_ohlcv(self, df: pd.DataFrame) -> Dict:
        """对单个 OHLCV 原始表计算质量指标，返回 dict。"""
        df = df.copy()
        if sc.COL_DATETIME in df.columns:
            df = df.set_index(sc.COL_DATETIME)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        n = len(df)
        # 缺失率
        missing = {}
        for c in ALL_COLS:
            missing[c] = float(df[c].isna().mean()) if c in df.columns else 1.0
        present = [c for c in ALL_COLS if c in df.columns]
        total_cells = n * len(ALL_COLS)
        total_missing = int(df[present].isna().sum().sum()) if present else total_cells
        missing_rate = total_missing / total_cells if total_cells else 0.0

        # 异常行（至少存在一种异常）
        anomaly = pd.Series(False, index=df.index)
        have = {c: c in df.columns for c in PRICE_COLS}
        if have[sc.COL_OPEN] and have[sc.COL_HIGH] and have[sc.COL_LOW] and have[sc.COL_CLOSE]:
            nonpos = (df[PRICE_COLS] <= 0).any(axis=1)
            hl_bad = df[sc.COL_HIGH] < df[sc.COL_LOW]
            out_of_hl = (df[sc.COL_CLOSE] > df[sc.COL_HIGH]) | (df[sc.COL_CLOSE] < df[sc.COL_LOW])
            ret = df[sc.COL_CLOSE].pct_change(fill_method=None)
            extreme = ret.abs().gt(self.limit_pct + 0.05).fillna(False)
            anomaly = nonpos | hl_bad | out_of_hl | extreme
        elif have[sc.COL_CLOSE]:
            anomaly = (df[sc.COL_CLOSE] <= 0) | df[sc.COL_CLOSE].isna()

        anomaly_rows = int(anomaly.sum())
        anomaly_rate = anomaly_rows / n if n else 0.0

        # 停牌与涨跌停（基于清洗后的派生列）
        cleaned = clean_ohlcv(df, limit_pct=self.limit_pct) if n else df
        if cleaned.empty:
            suspension_rate = 0.0
            limit_up = limit_down = 0
        else:
            suspension_rate = float(cleaned[sc.COL_IS_SUSPENDED].mean())
            limit_up = int((cleaned[sc.COL_LIMIT_STATUS] == sc.LIMIT_UP).sum())
            limit_down = int((cleaned[sc.COL_LIMIT_STATUS] == sc.LIMIT_DOWN).sum())

        start = df.index.min() if n else pd.NaT
        end = df.index.max() if n else pd.NaT
        return {
            "rows": n,
            "start": start,
            "end": end,
            "missing": missing,
            "missing_rate": missing_rate,
            "anomaly_rows": anomaly_rows,
            "anomaly_rate": anomaly_rate,
            "suspension_rate": suspension_rate,
            "limit_up": limit_up,
            "limit_down": limit_down,
        }

    def inspect_symbol(self, symbol: str) -> Dict:
        """读取某证券原始数据并计算质量指标。"""
        raw = self.loader.load_ohlcv(symbol, clean=False)
        stats = self.inspect_ohlcv(raw)
        stats["symbol"] = symbol
        return stats

    def build_report(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """批量生成质量报告；symbols 为空时扫描全部已落盘证券。"""
        if symbols is None:
            symbols = self.loader.list_ohlcv_symbols()
        return {s: self.inspect_symbol(s) for s in symbols}

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def to_markdown(self, report: Dict[str, Dict]) -> str:
        """把质量报告 dict 渲染为 markdown 文本。"""
        lines = [
            "# 数据质量报告",
            "",
            f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- 证券数量: {len(report)}",
            f"- 涨跌停阈值: {self.limit_pct:.0%}",
            "",
            "## 汇总",
            "",
            "| symbol | 行数 | 起始 | 结束 | 缺失率 | 异常率 | 停牌率 | 涨停数 | 跌停数 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for sym, s in report.items():
            start = "" if pd.isna(s.get("start")) else pd.Timestamp(s["start"]).strftime("%Y-%m-%d")
            end = "" if pd.isna(s.get("end")) else pd.Timestamp(s["end"]).strftime("%Y-%m-%d")
            lines.append(
                f"| {sym} | {s['rows']} | {start} | {end} | "
                f"{s['missing_rate']:.4%} | {s['anomaly_rate']:.4%} | "
                f"{s['suspension_rate']:.4%} | {s['limit_up']} | {s['limit_down']} |"
            )

        lines += ["", "## 分列缺失率", ""]
        lines.append("| symbol | " + " | ".join(ALL_COLS) + " |")
        lines.append("|---" * (len(ALL_COLS) + 1) + "|")
        for sym, s in report.items():
            vals = " | ".join(f"{s['missing'][c]:.4%}" for c in ALL_COLS)
            lines.append(f"| {sym} | {vals} |")
        return "\n".join(lines)

    def write_report(
        self,
        path: Optional[Union[str, Path]] = None,
        symbols: Optional[List[str]] = None,
    ) -> str:
        """生成并写出 markdown 报告，返回报告文件路径。"""
        report = self.build_report(symbols)
        md = self.to_markdown(report)
        out = Path(path) if path is not None else self.loader.data_root / "quality_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return str(out)
