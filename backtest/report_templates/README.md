# 报告模板说明

本目录用于放置 A4 绩效报告的自定义模板与使用说明。当前报告由
`backtest/report.py` 的 `ReportGenerator` 一键生成，无需额外模板文件即可输出
PNG / HTML / trade CSV。

## 快速使用

```python
import pandas as pd
from backtest.metrics_enhanced import PerformanceAnalyzer
from backtest.report import ReportGenerator

# equity_curve：A2 引擎输出（含 equity 列，index=timestamp）
# fills：A2 引擎输出 fills_frame()
analyzer = PerformanceAnalyzer(equity_curve, fills)
gen = ReportGenerator(analyzer, title="示例策略")
paths = gen.generate("./reports", name="demo")
print(paths)  # {'png': ..., 'html': ..., 'trades_csv': ...}
```

## 输出内容

- `{name}.png`：四合一图（权益曲线 / 回撤 / 月度热力图 / 成本占比）
- `{name}.html`：自包含 HTML（内嵌 base64 图 + 指标表 + 成交明细表），可直接邮件/归档
- `{name}_trades.csv`：FIFO 配对后的平仓交易明细

## 自定义模板

- 若要修改图表布局，直接改 `ReportGenerator.plot_figure()`。
- 若要修改 HTML 结构/样式，改 `ReportGenerator.build_html()`（当前为内联 CSS 模板）。
- 若需要外部 HTML 模板文件（Jinja2 等），可将模板放到本目录，并在 `build_html`
  中加载渲染——当前实现为保持零额外依赖而采用内联模板。
