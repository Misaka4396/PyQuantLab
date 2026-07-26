"""结果分析页面 — 详细绩效与风险分析."""

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backtest.report import BacktestReport
from backtest.risk import RiskAnalyzer
from ui.charts_qt import equity_curve_chart, returns_histogram, rolling_chart

FONT_FAMILY = "Microsoft YaHei"


class ResultsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("结果分析")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        self.desc_label = QLabel("请先在「运行回测」页面执行回测。")
        self.desc_label.setFont(QFont(FONT_FAMILY, 12))
        self.desc_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.desc_label)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont(FONT_FAMILY, 14))
        self.tabs.addTab(self._create_overview_tab(), "概览")
        self.tabs.addTab(self._create_equity_tab(), "权益曲线")
        self.tabs.addTab(self._create_drawdown_tab(), "回撤分析")
        self.tabs.addTab(self._create_trades_tab(), "交易分析")
        self.tabs.addTab(self._create_risk_tab(), "风险指标")
        layout.addWidget(self.tabs, 1)

    def _create_overview_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.metrics_table = QTableWidget()
        layout.addWidget(self.metrics_table)
        self.overview_chart = QWidget()
        self.overview_chart_layout = QVBoxLayout(self.overview_chart)
        self.overview_chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.overview_chart)
        return w

    def _create_equity_tab(self):
        self.equity_container = QWidget()
        layout = QVBoxLayout(self.equity_container)
        layout.setContentsMargins(0, 0, 0, 0)
        return self.equity_container

    def _create_drawdown_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl1 = QLabel("历史回撤记录")
        lbl1.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        layout.addWidget(lbl1)
        self.dd_table = QTableWidget()
        layout.addWidget(self.dd_table)

        lbl2 = QLabel("滚动夏普比率 (252日)")
        lbl2.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        lbl2.setStyleSheet("margin-top: 8px;")
        layout.addWidget(lbl2)
        self.rolling_sharpe_container = QWidget()
        self.rolling_sharpe_layout = QVBoxLayout(self.rolling_sharpe_container)
        self.rolling_sharpe_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.rolling_sharpe_container)
        return w

    def _create_trades_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.trade_kpi_layout = QHBoxLayout()
        self.trade_kpis = {}
        kpi_names = {"总交易次数": "Total Trades", "胜率": "Win Rate",
                      "平均盈利": "Avg Win", "平均亏损": "Avg Loss", "盈亏比": "Profit Factor"}
        for name in ["总交易次数", "胜率", "平均盈利", "平均亏损", "盈亏比"]:
            kw = QWidget()
            kl = QVBoxLayout(kw)
            lbl = QLabel(name)
            lbl.setFont(QFont(FONT_FAMILY, 12))
            lbl.setStyleSheet("color: #a6adc8;")
            kl.addWidget(lbl)
            val = QLabel("—")
            val.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
            kl.addWidget(val)
            self.trade_kpis[name] = val
            self.trade_kpi_layout.addWidget(kw)
        layout.addLayout(self.trade_kpi_layout)

        lbl = QLabel("交易明细")
        lbl.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        layout.addWidget(lbl)
        self.trades_table = QTableWidget()
        layout.addWidget(self.trades_table)
        return w

    def _create_risk_tab(self):
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        tg = QGroupBox("尾部风险")
        tl = QHBoxLayout(tg)
        self.tail_labels = {}
        for name in ["偏度", "峰度", "VaR 99%", "最大日亏损"]:
            kw = QWidget()
            kl = QVBoxLayout(kw)
            lbl = QLabel(name)
            lbl.setFont(QFont(FONT_FAMILY, 12))
            lbl.setStyleSheet("color: #a6adc8;")
            kl.addWidget(lbl)
            val = QLabel("—")
            val.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
            kl.addWidget(val)
            self.tail_labels[name] = val
            tl.addWidget(kw)
        layout.addWidget(tg)

        sg = QGroupBox("压力测试")
        sl = QVBoxLayout(sg)
        self.stress_table = QTableWidget()
        sl.addWidget(self.stress_table)
        layout.addWidget(sg)

        vg = QGroupBox("滚动波动率 (21日)")
        vl = QVBoxLayout(vg)
        self.roll_vol_container = QWidget()
        self.roll_vol_layout = QVBoxLayout(self.roll_vol_container)
        self.roll_vol_layout.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.roll_vol_container)
        layout.addWidget(vg)

        w.setWidget(inner)
        return w

    def on_activated(self):
        from PyQt5.QtWidgets import QApplication
        for w in QApplication.instance().allWidgets():
            if hasattr(w, 'nav_list'):
                bp = w.pages[3]
                if hasattr(bp, 'results') and bp.results:
                    name = list(bp.results.keys())[0]
                    result = bp.results[name]
                    self._display(result)
                break

    def _display(self, result):
        m = result.metrics
        report = BacktestReport(result)
        returns = result.equity_curve["returns"]
        self.desc_label.setText(f"当前回测结果: {report.summary_text().replace(chr(10), ' | ')}")

        # 概览
        df = report.metrics_table()
        self.metrics_table.setRowCount(len(df))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        for row in range(len(df)):
            item0 = QTableWidgetItem(str(df.iloc[row, 0]))
            item0.setFont(QFont(FONT_FAMILY, 13))
            item1 = QTableWidgetItem(str(df.iloc[row, 1]))
            item1.setFont(QFont(FONT_FAMILY, 13))
            self.metrics_table.setItem(row, 0, item0)
            self.metrics_table.setItem(row, 1, item1)

        while self.overview_chart_layout.count():
            child = self.overview_chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        canvas = equity_curve_chart(result.equity_curve)
        self.overview_chart_layout.addWidget(canvas)

        # 权益曲线
        while self.equity_container.layout().count():
            child = self.equity_container.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        ec = equity_curve_chart(result.equity_curve)
        self.equity_container.layout().addWidget(ec)

        # 回撤
        dd_df = RiskAnalyzer.drawdown_analysis(result.equity_curve["equity"])
        if not dd_df.empty:
            self.dd_table.setRowCount(len(dd_df))
            self.dd_table.setColumnCount(5)
            self.dd_table.setHorizontalHeaderLabels(["起始日期", "结束日期", "持续天数", "最大回撤", "恢复程度"])
            for row in range(len(dd_df)):
                for col, key in enumerate(["start", "end", "duration", "max_drawdown", "recovery"]):
                    val = dd_df.iloc[row][key]
                    if isinstance(val, float):
                        val = f"{val:.2%}"
                    item = QTableWidgetItem(str(val))
                    item.setFont(QFont(FONT_FAMILY, 13))
                    self.dd_table.setItem(row, col, item)

        while self.rolling_sharpe_layout.count():
            child = self.rolling_sharpe_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        rs = rolling_chart(RiskAnalyzer.rolling_sharpe(returns), "滚动夏普比率 (252日)")
        self.rolling_sharpe_layout.addWidget(rs)

        # 交易分析
        self.trade_kpis["总交易次数"].setText(str(m.total_trades))
        self.trade_kpis["胜率"].setText(f"{m.win_rate*100:.1f}%")
        self.trade_kpis["平均盈利"].setText(f"{m.avg_win*100:.2f}%")
        self.trade_kpis["平均亏损"].setText(f"{m.avg_loss*100:.2f}%")
        self.trade_kpis["盈亏比"].setText(f"{m.profit_factor:.2f}")

        trades = result.trades
        if not trades.empty:
            self.trades_table.setRowCount(len(trades))
            cols = ["entry_date", "exit_date", "pnl", "pnl_pct", "win"]
            self.trades_table.setColumnCount(len(cols))
            self.trades_table.setHorizontalHeaderLabels(["入场日期", "出场日期", "盈亏", "盈亏 %", "盈利"])
            for row in range(len(trades)):
                for col, key in enumerate(cols):
                    val = trades.iloc[row][key]
                    if key == "pnl_pct":
                        val = f"{val*100:.2f}%"
                    elif key == "win":
                        val = "是" if val else "否"
                    item = QTableWidgetItem(str(val))
                    item.setFont(QFont(FONT_FAMILY, 13))
                    self.trades_table.setItem(row, col, item)

        # 风险
        tail = RiskAnalyzer.tail_risk(returns)
        for name, key in [("偏度", "skewness"), ("峰度", "kurtosis"),
                           ("VaR 99%", "var_99"), ("最大日亏损", "max_daily_loss")]:
            v = tail[key]
            if abs(v) < 1:
                fmt = f"{v:.4f}"
            elif abs(v) < 0.1:
                fmt = f"{v*100:.3f}%"
            else:
                fmt = f"{v*100:.2f}%"
            self.tail_labels[name].setText(fmt)

        stress = RiskAnalyzer.stress_test(returns)
        self.stress_table.setRowCount(len(stress))
        self.stress_table.setColumnCount(3)
        self.stress_table.setHorizontalHeaderLabels(["场景", "冲击幅度", "最终价值 ($)"])
        for row in range(len(stress)):
            for col, key in enumerate(["scenario", "shock", "final_value"]):
                val = stress.iloc[row][key]
                if key == "shock":
                    val = f"{val:.0%}"
                item = QTableWidgetItem(str(val))
                item.setFont(QFont(FONT_FAMILY, 13))
                self.stress_table.setItem(row, col, item)

        while self.roll_vol_layout.count():
            child = self.roll_vol_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        rv = rolling_chart(RiskAnalyzer.rolling_volatility(returns), "滚动波动率 (21日)")
        self.roll_vol_layout.addWidget(rv)
