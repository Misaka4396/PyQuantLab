"""C4 过拟合检测报告生成（markdown，含判定阈值、风险结论、上线建议）。

消费 ``ml.overfit.OverfitAssessment``，渲染为 markdown 文本或落盘 .md 文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ml.overfit import OverfitAssessment


def _fmt(v: Optional[float]) -> str:
    """数值格式化：float 保留 4 位，None 显示 —。"""
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_overfit_report(
    assessment: OverfitAssessment,
    title: str = "过拟合检测报告",
) -> str:
    """渲染过拟合检测 markdown 文本。"""
    a = assessment
    t = a.thresholds
    lines: list = []
    lines.append(f"# {title}")
    lines.append("")

    lines.append("## 风险结论")
    lines.append(f"- 风险等级：**{a.risk_level}**")
    lines.append(f"- 上线建议：{a.recommendation}")
    lines.append("")

    lines.append("## 判定阈值")
    lines.append("| 指标 | 低风险 | 中风险 | 高风险 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| Deflated Sharpe (DSR) | ≥ {t['dsr_safe']} "
        f"| [{t['dsr_high_risk']}, {t['dsr_safe']}) | < {t['dsr_high_risk']} |"
    )
    lines.append(
        f"| PBO | ≤ {t['pbo_low_risk']} "
        f"| ({t['pbo_low_risk']}, {t['pbo_high_risk']}] | > {t['pbo_high_risk']} |"
    )
    lines.append(
        f"| OOS/IS Sharpe 衰减 | ≥ {t['degradation_safe']} "
        f"| [{t['degradation_high_risk']}, {t['degradation_safe']}) "
        f"| < {t['degradation_high_risk']} |"
    )
    lines.append("")

    lines.append("## 指标明细")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| Deflated Sharpe (DSR) | {_fmt(a.dsr)} |")
    lines.append(f"| PBO（参数版） | {_fmt(a.pbo)} |")
    lines.append(f"| PBO（频率版） | {_fmt(a.pbo_freq)} |")
    lines.append(f"| OOS/IS Sharpe 衰减 | {_fmt(a.sharpe_degradation)} |")
    lines.append(f"| OOS/IS 年化收益衰减 | {_fmt(a.return_degradation)} |")
    lines.append(f"| IS Sharpe（年化） | {_fmt(a.is_sharpe)} |")
    lines.append(f"| OOS Sharpe（年化） | {_fmt(a.oos_sharpe)} |")
    lines.append(f"| Sharpe（每期） | {_fmt(a.sharpe_period)} |")
    lines.append(f"| Sharpe（年化） | {_fmt(a.sharpe_annual)} |")
    lines.append(f"| 偏度 γ3 | {_fmt(a.skew)} |")
    lines.append(f"| 峰度 γ4（full） | {_fmt(a.kurtosis)} |")
    lines.append(f"| 试验次数 N | {a.trials} |")
    lines.append("")

    lines.append("## 判定依据")
    for r in a.reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## 公式来源")
    lines.append("- DSR/PSR：Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*")
    lines.append("- PBO/CSCV：Bailey, Borwein, López de Prado & Zhu (2015), *The Probability of Backtest Overfitting*")
    return "\n".join(lines) + "\n"


def generate_overfit_report(
    assessment: OverfitAssessment,
    output_dir: Union[str, Path],
    name: str = "overfit_report",
    title: Optional[str] = None,
) -> str:
    """落盘 markdown 报告，返回文件路径。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.md"
    text = render_overfit_report(assessment, title=title or "过拟合检测报告")
    path.write_text(text, encoding="utf-8")
    return str(path)


__all__ = [
    "render_overfit_report",
    "generate_overfit_report",
]
