"""组合模拟页面 — 组合优化与蒙特卡洛模拟."""

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portfolio.optimizer import PortfolioOptimizer
from portfolio.simulation import PortfolioSimulation
from ui.charts_qt import efficient_frontier_chart, monte_carlo_chart
from ui.worker import run_in_thread

FONT_FAMILY = "Microsoft YaHei"


class PortfolioPage(QWidget):
    def __init__(self):
        super().__init__()
        self.returns: pd.DataFrame = None
        self.opt_weights = None
        self._opt_thread = None
        self._mc_thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("组合模拟")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        opt_group = QGroupBox("组合优化")
        opt_layout = QHBoxLayout(opt_group)

        lbl1 = QLabel("优化方法:")
        lbl1.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        opt_layout.addWidget(lbl1)

        self.opt_combo = QComboBox()
        self.opt_combo.setFont(QFont(FONT_FAMILY, 14))
        self.opt_combo.addItems(["最大夏普比率", "最小方差", "风险平价", "等权重"])
        opt_layout.addWidget(self.opt_combo)

        self.opt_btn = QPushButton("开始优化")
        self.opt_btn.clicked.connect(self._optimize)
        opt_layout.addWidget(self.opt_btn)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        top_layout = QHBoxLayout()
        self.weights_table = QTableWidget()
        top_layout.addWidget(self.weights_table, 1)

        self.frontier_container = QWidget()
        self.frontier_layout = QVBoxLayout(self.frontier_container)
        self.frontier_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.frontier_container, 2)
        layout.addLayout(top_layout, 1)

        self.opt_metrics_layout = QHBoxLayout()
        self.opt_metrics = {}
        for name in ["预期收益", "预期波动率", "夏普比率"]:
            w = QWidget()
            wl = QVBoxLayout(w)
            lbl = QLabel(name)
            lbl.setFont(QFont(FONT_FAMILY, 12))
            lbl.setStyleSheet("color: #a6adc8;")
            wl.addWidget(lbl)
            val = QLabel("—")
            val.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
            wl.addWidget(val)
            self.opt_metrics[name] = val
            self.opt_metrics_layout.addWidget(w)
        self.opt_metrics_layout.addStretch()
        layout.addLayout(self.opt_metrics_layout)

        mc_group = QGroupBox("蒙特卡洛模拟")
        mc_layout = QHBoxLayout(mc_group)

        lbl_font = QFont(FONT_FAMILY, 13)

        lbl2 = QLabel("模拟次数:")
        lbl2.setFont(lbl_font)
        mc_layout.addWidget(lbl2)

        self.mc_sims = QSpinBox()
        self.mc_sims.setFont(QFont(FONT_FAMILY, 13))
        self.mc_sims.setRange(100, 10000)
        self.mc_sims.setValue(1000)
        self.mc_sims.setSingleStep(100)
        mc_layout.addWidget(self.mc_sims)

        lbl3 = QLabel("时间跨度 (天):")
        lbl3.setFont(lbl_font)
        mc_layout.addWidget(lbl3)

        self.mc_days = QSpinBox()
        self.mc_days.setFont(QFont(FONT_FAMILY, 13))
        self.mc_days.setRange(21, 1260)
        self.mc_days.setValue(252)
        mc_layout.addWidget(self.mc_days)

        lbl4 = QLabel("初始资金 ($):")
        lbl4.setFont(lbl_font)
        mc_layout.addWidget(lbl4)

        self.mc_capital = QSpinBox()
        self.mc_capital.setFont(QFont(FONT_FAMILY, 13))
        self.mc_capital.setRange(1000, 10000000)
        self.mc_capital.setValue(100000)
        self.mc_capital.setSingleStep(10000)
        mc_layout.addWidget(self.mc_capital)

        self.mc_btn = QPushButton("运行模拟")
        self.mc_btn.clicked.connect(self._run_mc)
        mc_layout.addWidget(self.mc_btn)
        mc_layout.addStretch()
        layout.addWidget(mc_group)

        self.mc_container = QWidget()
        self.mc_chart_layout = QVBoxLayout(self.mc_container)
        self.mc_chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mc_container, 1)

        self.mc_stats_layout = QHBoxLayout()
        self.mc_stats = {}
        for name in ["均值", "中位数", "P5 下限", "P95 上限"]:
            w = QWidget()
            wl = QVBoxLayout(w)
            lbl = QLabel(name)
            lbl.setFont(QFont(FONT_FAMILY, 12))
            lbl.setStyleSheet("color: #a6adc8;")
            wl.addWidget(lbl)
            val = QLabel("—")
            val.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
            wl.addWidget(val)
            self.mc_stats[name] = val
            self.mc_stats_layout.addWidget(w)
        self.mc_stats_layout.addStretch()
        layout.addLayout(self.mc_stats_layout)

    def on_activated(self):
        from PyQt5.QtWidgets import QApplication

        for w in QApplication.instance().allWidgets():
            if hasattr(w, "nav_list"):
                dp = w.pages[1]
                if hasattr(dp, "data") and dp.data is not None:
                    data = dp.data
                    if isinstance(data.columns, pd.MultiIndex):
                        tickers = list(data.columns.levels[0])
                        closes = pd.DataFrame({t: data[(t, "Close")] for t in tickers})
                    else:
                        closes = data
                    self.returns = closes.pct_change().dropna()
                break

    def _optimize(self):
        if self.returns is None or len(self.returns.columns) < 2:
            return
        self.opt_btn.setEnabled(False)
        self.opt_btn.setText("优化中...")
        method = self.opt_combo.currentText()
        returns = self.returns.copy()

        def _compute():
            opt = PortfolioOptimizer(returns)
            if method == "最大夏普比率":
                pw = opt.max_sharpe()
            elif method == "最小方差":
                pw = opt.min_variance()
            elif method == "风险平价":
                pw = opt.risk_parity()
            else:
                pw = opt.equal_weight()
            frontier = opt.efficient_frontier(30)
            return pw, frontier

        self._opt_thread = run_in_thread(
            self,
            _compute,
            on_finished=self._on_opt_done,
            on_error=lambda e: self._set_opt_enabled(True),
        )

    def _on_opt_done(self, result):
        pw, frontier = result
        self.opt_weights = pw

        self.weights_table.setRowCount(len(pw.assets))
        self.weights_table.setColumnCount(2)
        self.weights_table.setHorizontalHeaderLabels(["资产", "权重"])
        for row, (asset, wgt) in enumerate(zip(pw.assets, pw.weights, strict=False)):
            item0 = QTableWidgetItem(asset)
            item0.setFont(QFont(FONT_FAMILY, 13))
            self.weights_table.setItem(row, 0, item0)
            item1 = QTableWidgetItem(f"{wgt * 100:.1f}%")
            item1.setTextAlignment(Qt.AlignCenter)
            item1.setFont(QFont(FONT_FAMILY, 13))
            self.weights_table.setItem(row, 1, item1)

        self.opt_metrics["预期收益"].setText(f"{pw.expected_return * 100:.2f}%")
        self.opt_metrics["预期波动率"].setText(f"{pw.expected_volatility * 100:.2f}%")
        self.opt_metrics["夏普比率"].setText(f"{pw.sharpe_ratio:.2f}")

        while self.frontier_layout.count():
            child = self.frontier_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if not frontier.empty:
            canvas = efficient_frontier_chart(frontier, pw)
            self.frontier_layout.addWidget(canvas)

        self.opt_btn.setEnabled(True)
        self.opt_btn.setText("开始优化")

    def _set_opt_enabled(self, enabled):
        self.opt_btn.setEnabled(enabled)
        self.opt_btn.setText("开始优化")

    def _run_mc(self):
        if self.opt_weights is None or self.returns is None:
            return
        self.mc_btn.setEnabled(False)
        self.mc_btn.setText("模拟中...")
        weights = self.opt_weights.weights.copy()
        returns = self.returns.copy()
        num_sims = self.mc_sims.value()
        initial_value = self.mc_capital.value()
        days = self.mc_days.value()

        def _compute():
            sim = PortfolioSimulation(returns, num_simulations=num_sims)
            paths = sim.run(weights, initial_value=initial_value, days=days)
            percentiles = sim.compute_path_percentiles(paths)
            stats = sim.compute_terminal_stats(paths)
            return percentiles, stats

        self._mc_thread = run_in_thread(
            self,
            _compute,
            on_finished=self._on_mc_done,
            on_error=lambda e: self._set_mc_enabled(True),
        )

    def _on_mc_done(self, result):
        percentiles, stats = result

        while self.mc_chart_layout.count():
            child = self.mc_chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        canvas = monte_carlo_chart(percentiles)
        self.mc_chart_layout.addWidget(canvas)

        for name, key in [
            ("均值", "mean"),
            ("中位数", "median"),
            ("P5 下限", "p5"),
            ("P95 上限", "p95"),
        ]:
            val = stats.get(key, 0)
            self.mc_stats[name].setText(f"${val:,.0f}")

        self.mc_btn.setEnabled(True)
        self.mc_btn.setText("运行模拟")

    def _set_mc_enabled(self, enabled):
        self.mc_btn.setEnabled(enabled)
        self.mc_btn.setText("运行模拟")
