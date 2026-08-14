"""A4 绩效报告生成器 + 旧版向量化报告（BacktestReport，供 UI 兼容）。

新增 ``ReportGenerator``：输入权益曲线 + 成交明细，一键输出：
- PNG 图（权益曲线 / 回撤 / 月度热力图 / 成本占比 四合一）
- HTML 报告（自包含，内嵌 base64 图 + 指标表 + 成交表）
- trade list CSV（平仓交易导出）

旧版 ``BacktestReport``（消费 core.types.BacktestResult）保留在文件末尾，
供 ui/pages 与 ui/pages_qt 的向量化回测页面继续使用。
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from backtest.metrics_enhanced import PerformanceAnalyzer, build_round_trips
from core.types import BacktestResult  # 旧版 BacktestReport 依赖

matplotlib.use("Agg")  # 无界面环境生成图（服务器/测试）
import matplotlib.pyplot as plt

# 中文标题/标签渲染：Windows 优先中文字体，缺失时回退默认（仅影响图内文字显示）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 报告生成器
# ---------------------------------------------------------------------------
class ReportGenerator:
    """一键绩效报告：权益/回撤/月度热力图/成本占比 + HTML + trade CSV。"""

    def __init__(self, analyzer: PerformanceAnalyzer, title: str = "策略绩效报告"):
        self.analyzer = analyzer
        self.title = title

    # ------------------------------ 绘图 ------------------------------
    def plot_figure(self) -> plt.Figure:
        """生成 2x2 组合图（权益、回撤、月度热力图、成本占比）。"""
        a = self.analyzer
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(self.title, fontsize=14, fontweight="bold")

        # 1) 权益曲线
        ax = axes[0, 0]
        ax.plot(a.equity.index, a.equity.values, color="#1f77b4", lw=1.2)
        ax.set_title("权益曲线")
        ax.set_ylabel("权益")
        ax.grid(alpha=0.3)

        # 2) 回撤
        ax = axes[0, 1]
        dd = a.drawdown_series()
        ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
        ax.set_title("回撤")
        ax.set_ylabel("回撤")
        ax.grid(alpha=0.3)

        # 3) 月度热力图
        ax = axes[1, 0]
        monthly = a.monthly_returns()
        if not monthly.empty:
            heat = _monthly_heatmap_matrix(monthly)
            if heat.size:
                vals = heat.to_numpy()
                max_abs = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
                im = ax.imshow(heat, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
                ax.set_xticks(range(heat.shape[1]))
                ax.set_xticklabels([f"{m}月" for m in range(1, heat.shape[1] + 1)], fontsize=8)
                ax.set_yticks(range(heat.shape[0]))
                ax.set_yticklabels(heat.index, fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                for i in range(heat.shape[0]):
                    for j in range(heat.shape[1]):
                        v = heat.iloc[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7)
        ax.set_title("月度收益热力图")

        # 4) 成本占比（固定费用分项）
        ax = axes[1, 1]
        cost = _cost_breakdown(a.fills)
        if cost:
            labels = list(cost.keys())
            values = list(cost.values())
            ax.bar(labels, values, color=["#ff7f0e", "#2ca02c", "#9467bd"])
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(0.5, 0.5, "无成交", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("成本占比（固定费用，元）")
        ax.grid(alpha=0.3, axis="y")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # ------------------------------ 输出 ------------------------------
    def export_trades(self, path: str | Path) -> str:
        """把 FIFO 平仓交易导出为 CSV，返回文件路径。"""
        trips = build_round_trips(self.analyzer.fills, self.analyzer.equity.index)
        if trips:
            df = pd.DataFrame(
                [
                    {
                        "symbol": t.symbol,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "quantity": t.quantity,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "win": t.win,
                        "holding_bars": t.holding_bars,
                    }
                    for t in trips
                ]
            )
        else:
            df = pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "pnl_pct",
                    "win",
                    "holding_bars",
                ]
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        return str(p)

    def build_html(self, png_b64: str) -> str:
        """构建自包含 HTML 报告（内嵌 base64 图 + 指标表 + 成交表）。"""
        a = self.analyzer
        metrics = a.to_dataframe()
        metrics_html = metrics.to_html(index=False, border=0, classes="metrics")
        trips = build_round_trips(a.fills, a.equity.index)
        trades_df = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "quantity": t.quantity,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "win": t.win,
                    "holding_bars": t.holding_bars,
                }
                for t in trips
            ]
        )
        trades_html = (
            trades_df.to_html(index=False, border=0, classes="trades")
            if trips
            else "<p>无平仓交易</p>"
        )

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{self.title}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 20px; }}
img {{ max-width: 100%; border: 1px solid #eee; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
table.metrics td, table.trades td, table.trades th {{ padding: 4px 10px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<img src="data:image/png;base64,{png_b64}" alt="报告图">
<h2>绩效指标</h2>
{metrics_html}
<h2>交易明细</h2>
{trades_html}
</body>
</html>"""

    def generate(
        self,
        output_dir: str | Path,
        name: str = "report",
        show: bool = False,
    ) -> dict[str, str]:
        """一键生成报告，返回各产物路径 dict：png / html / trades_csv。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig = self.plot_figure()
        png_path = out / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)

        html_path = out / f"{name}.html"
        html_path.write_text(self.build_html(png_b64), encoding="utf-8")

        csv_path = self.export_trades(out / f"{name}_trades.csv")

        if show:
            with contextlib.suppress(Exception):
                plt.show()

        return {
            "png": str(png_path),
            "html": str(html_path),
            "trades_csv": str(csv_path),
        }


# ---------------------------------------------------------------------------
# 绘图辅助
# ---------------------------------------------------------------------------
def _monthly_heatmap_matrix(monthly: pd.Series) -> pd.DataFrame:
    """把月度收益（DatetimeIndex，月末）转成年×月矩阵（行=年份，列=1..12）。"""
    s = monthly.copy()
    s.index = pd.to_datetime(s.index)
    df = pd.DataFrame({"year": s.index.year, "month": s.index.month, "ret": s.values})
    years = sorted(df["year"].unique())
    heat = pd.DataFrame(np.nan, index=years, columns=range(1, 13))
    for _, row in df.iterrows():
        heat.at[row["year"], row["month"]] = row["ret"]
    return heat


def _cost_breakdown(fills: pd.DataFrame) -> dict[str, float]:
    """固定费用分项合计：佣金 / 印花税 / 过户费。"""
    if fills is None or len(fills) == 0:
        return {}
    out = {}
    for col, label in (("commission", "佣金"), ("stamp_tax", "印花税"), ("transfer_fee", "过户费")):
        if col in fills.columns:
            v = float(pd.to_numeric(fills[col], errors="coerce").sum())
            if v > 0:
                out[label] = v
    return out


# ---------------------------------------------------------------------------
# 旧版向量化报告（供 UI 兼容，勿删）
# ---------------------------------------------------------------------------
class BacktestReport:
    def __init__(self, result: BacktestResult):
        self.result = result

    def summary_text(self) -> str:
        m = self.result.metrics
        lines = [
            f"总收益率: {m.total_return * 100:.2f}%",
            f"年化收益: {m.annualized_return * 100:.2f}%",
            f"夏普比率: {m.sharpe_ratio:.2f}",
            f"最大回撤: {m.max_drawdown * 100:.2f}%",
            f"胜率: {m.win_rate * 100:.1f}%",
            f"交易次数: {m.total_trades}",
        ]
        return "\n".join(lines)

    def metrics_table(self) -> pd.DataFrame:
        m = self.result.metrics
        data = {
            "总收益率": f"{m.total_return * 100:.2f}%",
            "年化收益": f"{m.annualized_return * 100:.2f}%",
            "年化波动率": f"{m.annualized_volatility * 100:.2f}%",
            "夏普比率": f"{m.sharpe_ratio:.2f}",
            "索提诺比率": f"{m.sortino_ratio:.2f}",
            "最大回撤": f"{m.max_drawdown * 100:.2f}%",
            "最大回撤天数": str(m.max_drawdown_duration),
            "卡玛比率": f"{m.calmar_ratio:.2f}",
            "Alpha": f"{m.alpha:.4f}",
            "Beta": f"{m.beta:.4f}",
            "VaR 95%": f"{m.var_95 * 100:.2f}%",
            "CVaR 95%": f"{m.cvar_95 * 100:.2f}%",
            "胜率": f"{m.win_rate * 100:.1f}%",
            "盈亏比": f"{m.profit_factor:.2f}",
            "交易次数": str(m.total_trades),
            "盈利次数": str(m.winning_trades),
            "亏损次数": str(m.losing_trades),
            "平均盈利": f"{m.avg_win * 100:.2f}%",
            "平均亏损": f"{m.avg_loss * 100:.2f}%",
        }
        df = pd.DataFrame(list(data.items()), columns=["指标", "数值"])
        return df

    def to_dict(self) -> dict:
        return {
            "config": {
                "initial_capital": self.result.config.initial_capital,
                "commission": self.result.config.commission,
                "slippage": self.result.config.slippage,
            },
            "metrics": {
                k: v for k, v in self.result.metrics.__dict__.items() if not k.startswith("_")
            },
            "total_trades": len(self.result.trades),
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
