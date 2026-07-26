"""回测运行页面 — 配置并执行回测."""

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

from backtest.engine import BacktestEngine
from backtest.report import BacktestReport
from core.types import BacktestConfig
from strategy.registry import registry
from ui.charts_qt import equity_curve_chart
from ui.worker import run_in_thread

FONT_FAMILY = "Microsoft YaHei"


class BacktestPage(QWidget):
    def __init__(self):
        super().__init__()
        self.data: pd.DataFrame = None
        self.results = {}
        self._thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("运行回测")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        cfg_group = QGroupBox("回测参数")
        cfg = QHBoxLayout(cfg_group)

        lbl_font = QFont(FONT_FAMILY, 13)

        lbl1 = QLabel("初始资金 ($):")
        lbl1.setFont(lbl_font)
        cfg.addWidget(lbl1)

        self.capital_spin = QSpinBox()
        self.capital_spin.setFont(QFont(FONT_FAMILY, 13))
        self.capital_spin.setRange(1000, 10000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setSingleStep(10000)
        cfg.addWidget(self.capital_spin)

        lbl2 = QLabel("手续费 (%):")
        lbl2.setFont(lbl_font)
        cfg.addWidget(lbl2)

        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setFont(QFont(FONT_FAMILY, 13))
        self.commission_spin.setRange(0.0, 1.0)
        self.commission_spin.setValue(0.1)
        self.commission_spin.setSingleStep(0.01)
        self.commission_spin.setDecimals(3)
        cfg.addWidget(self.commission_spin)

        lbl3 = QLabel("滑点 (%):")
        lbl3.setFont(lbl_font)
        cfg.addWidget(lbl3)

        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setFont(QFont(FONT_FAMILY, 13))
        self.slippage_spin.setRange(0.0, 1.0)
        self.slippage_spin.setValue(0.05)
        self.slippage_spin.setSingleStep(0.01)
        self.slippage_spin.setDecimals(3)
        cfg.addWidget(self.slippage_spin)

        layout.addWidget(cfg_group)

        strat_layout = QHBoxLayout()

        lbl4 = QLabel("策略:")
        lbl4.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        strat_layout.addWidget(lbl4)

        self.strategy_combo = QComboBox()
        self.strategy_combo.setFont(QFont(FONT_FAMILY, 14))
        self.strategy_combo.addItems(registry.list_names())
        strat_layout.addWidget(self.strategy_combo)

        self.run_btn = QPushButton("运行回测")
        self.run_btn.clicked.connect(self._run_backtest)
        strat_layout.addWidget(self.run_btn)

        self.compare_btn = QPushButton("对比全部策略")
        self.compare_btn.clicked.connect(self._compare_all)
        strat_layout.addWidget(self.compare_btn)

        strat_layout.addStretch()
        layout.addLayout(strat_layout)

        self.kpi_layout = QHBoxLayout()
        self.kpi_labels = {}
        kpi_names = ["总收益率", "年化收益", "夏普比率", "最大回撤", "胜率", "交易次数"]
        for name in kpi_names:
            w = QWidget()
            wl = QVBoxLayout(w)
            title = QLabel(name)
            title.setFont(QFont(FONT_FAMILY, 11))
            title.setStyleSheet("color: #a6adc8;")
            value = QLabel("—")
            value.setFont(QFont(FONT_FAMILY, 20, QFont.Bold))
            wl.addWidget(title)
            wl.addWidget(value)
            self.kpi_labels[name] = value
            self.kpi_layout.addWidget(w)
        layout.addLayout(self.kpi_layout)

        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart_container, 1)

        self.comp_table = QTableWidget()
        self.comp_table.setVisible(False)
        layout.addWidget(self.comp_table)

    def on_activated(self):
        from PyQt5.QtWidgets import QApplication
        for w in QApplication.instance().allWidgets():
            if hasattr(w, 'nav_list'):
                dp = w.pages[1]
                if hasattr(dp, 'data') and dp.data is not None:
                    self.data = dp.data
                sp = w.pages[2]
                if hasattr(sp, 'strategy_combo'):
                    idx = self.strategy_combo.findText(sp.strategy_combo.currentText())
                    if idx >= 0:
                        self.strategy_combo.setCurrentIndex(idx)
                break

    def _get_config(self):
        return BacktestConfig(
            initial_capital=self.capital_spin.value(),
            commission=self.commission_spin.value() / 100,
            slippage=self.slippage_spin.value() / 100,
        )

    def _set_buttons_enabled(self, enabled):
        self.run_btn.setEnabled(enabled)
        self.compare_btn.setEnabled(enabled)
        if not enabled:
            self.run_btn.setText("运行中...")
        else:
            self.run_btn.setText("运行回测")

    def _run_backtest(self):
        if self.data is None:
            return
        self._set_buttons_enabled(False)
        name = self.strategy_combo.currentText()
        params = self._get_strategy_params_from_ui(name)
        config = self._get_config()

        def _compute():
            strategy = registry.create(name, **params)
            engine = BacktestEngine(config)
            return {name: engine.run(self.data, strategy)}

        self._thread = run_in_thread(
            self, _compute,
            on_finished=self._on_single_done,
            on_error=lambda e: self._set_buttons_enabled(True),
        )

    def _on_single_done(self, results):
        self.results = results
        name = list(results.keys())[0]
        self._display_result(results[name])
        self._update_comparison_table()
        self._set_buttons_enabled(True)

    def _compare_all(self):
        if self.data is None:
            return
        self._set_buttons_enabled(False)
        config = self._get_config()

        def _compute():
            engine = BacktestEngine(config)
            results = {}
            for name in registry.list_names():
                spec = registry.get_param_spec(name)
                default_params = {k: v["default"] for k, v in spec.items()}
                strategy = registry.create(name, **default_params)
                results[name] = engine.run(self.data, strategy)
            return results

        self._thread = run_in_thread(
            self, _compute,
            on_finished=self._on_compare_done,
            on_error=lambda e: self._set_buttons_enabled(True),
        )

    def _on_compare_done(self, results):
        self.results = results
        name = self.strategy_combo.currentText()
        if name in self.results:
            self._display_result(self.results[name])
        self._update_comparison_table()
        self._set_buttons_enabled(True)

    def _get_strategy_params_from_ui(self, name: str) -> dict:
        from PyQt5.QtWidgets import QApplication
        for w in QApplication.instance().allWidgets():
            if hasattr(w, 'nav_list'):
                sp = w.pages[2]
                if hasattr(sp, '_get_current_params'):
                    return sp._get_current_params()
        spec = registry.get_param_spec(name)
        return {k: v["default"] for k, v in spec.items()}

    def _display_result(self, result):
        m = result.metrics
        values = [
            f"{m.total_return * 100:.2f}%",
            f"{m.annualized_return * 100:.2f}%",
            f"{m.sharpe_ratio:.2f}",
            f"{m.max_drawdown * 100:.2f}%",
            f"{m.win_rate * 100:.1f}%",
            str(m.total_trades),
        ]
        for label, val in zip(self.kpi_labels.keys(), values):
            self.kpi_labels[label].setText(val)

        while self.chart_layout.count():
            child = self.chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        canvas = equity_curve_chart(result.equity_curve)
        self.chart_layout.addWidget(canvas)

    def _update_comparison_table(self):
        if len(self.results) <= 1:
            self.comp_table.setVisible(False)
            return
        self.comp_table.setVisible(True)
        headers = ["策略", "总收益率", "夏普比率", "最大回撤", "胜率", "交易次数"]
        self.comp_table.setColumnCount(len(headers))
        self.comp_table.setHorizontalHeaderLabels(headers)
        self.comp_table.setRowCount(len(self.results))
        for row, (name, r) in enumerate(self.results.items()):
            m = r.metrics
            vals = [
                name, f"{m.total_return*100:.2f}%", f"{m.sharpe_ratio:.2f}",
                f"{m.max_drawdown*100:.2f}%", f"{m.win_rate*100:.1f}%", str(m.total_trades),
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont(FONT_FAMILY, 13))
                self.comp_table.setItem(row, col, item)
        self.comp_table.horizontalHeader().setStretchLastSection(True)
