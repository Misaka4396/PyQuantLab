"""策略配置页面 — 选择与配置交易策略."""

import pandas as pd
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from strategy.registry import registry
from ui.charts_qt import price_chart
from ui.worker import run_in_thread

FONT_FAMILY = "Microsoft YaHei"


class StrategyPage(QWidget):
    def __init__(self):
        super().__init__()
        self.data: pd.DataFrame = None
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._do_update_preview)
        self._preview_thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("策略配置")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        desc = QLabel("选择交易策略并调整参数。信号将实时显示在走势图上。")
        desc.setFont(QFont(FONT_FAMILY, 12))
        desc.setStyleSheet("color: #a6adc8; margin-bottom: 12px;")
        layout.addWidget(desc)

        sel_layout = QHBoxLayout()
        lbl = QLabel("策略:")
        lbl.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        sel_layout.addWidget(lbl)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(registry.list_names())
        self.strategy_combo.setFont(QFont(FONT_FAMILY, 14))
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        sel_layout.addWidget(self.strategy_combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        self.desc_label = QLabel()
        self.desc_label.setFont(QFont(FONT_FAMILY, 12))
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #a6adc8; margin: 6px 0;")
        layout.addWidget(self.desc_label)

        self.params_group = QGroupBox("参数配置")
        self.params_layout = QVBoxLayout(self.params_group)
        self.param_widgets = {}
        layout.addWidget(self.params_group)

        stats_layout = QHBoxLayout()
        self.buy_label = QLabel("买入信号: —")
        self.buy_label.setFont(QFont(FONT_FAMILY, 14))
        self.sell_label = QLabel("卖出信号: —")
        self.sell_label.setFont(QFont(FONT_FAMILY, 14))
        stats_layout.addWidget(self.buy_label)
        stats_layout.addWidget(self.sell_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart_container, 1)

        self._on_strategy_changed(self.strategy_combo.currentText())

    def on_activated(self):
        from PyQt5.QtWidgets import QApplication

        for w in QApplication.instance().allWidgets():
            if hasattr(w, "nav_list"):
                data_page = w.pages[1]
                if hasattr(data_page, "data") and data_page.data is not None:
                    self.data = data_page.data
                    self._update_preview()
                break

    def _on_strategy_changed(self, name: str):
        if not name:
            return
        cls = registry.get(name)
        self.desc_label.setText(cls.get_description())

        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.param_widgets.clear()

        spec = cls.get_param_spec()
        row_layout = QHBoxLayout()
        col_count = 0

        for key, info in spec.items():
            w = QWidget()
            wl = QVBoxLayout(w)
            lbl = QLabel(info.get("label", key))
            lbl.setFont(QFont(FONT_FAMILY, 13))
            wl.addWidget(lbl)

            if info["type"] == "int":
                sw = QSpinBox()
                sw.setFont(QFont(FONT_FAMILY, 13))
                sw.setRange(info.get("min", 0), info.get("max", 1000))
                sw.setValue(info.get("default", 0))
                sw.setSingleStep(info.get("step", 1))
                sw.valueChanged.connect(self._update_preview)
            elif info["type"] == "choice":
                sw = QComboBox()
                sw.setFont(QFont(FONT_FAMILY, 13))
                sw.addItems(info.get("choices", []))
                sw.setCurrentText(str(info.get("default", "")))
                sw.currentTextChanged.connect(self._update_preview)
            else:
                sw = QSpinBox()
                sw.setFont(QFont(FONT_FAMILY, 13))
                sw.setRange(0, 1000)
                sw.valueChanged.connect(self._update_preview)

            wl.addWidget(sw)
            self.param_widgets[key] = sw
            row_layout.addWidget(w)
            col_count += 1

            if col_count >= 3:
                self.params_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
                col_count = 0

        if col_count > 0:
            self.params_layout.addLayout(row_layout)

    def _get_current_params(self) -> dict:
        spec = registry.get_param_spec(self.strategy_combo.currentText())
        params = {}
        for key, info in spec.items():
            w = self.param_widgets.get(key)
            if w is None:
                params[key] = info["default"]
            elif isinstance(w, QSpinBox):
                params[key] = w.value()
            elif isinstance(w, QComboBox):
                params[key] = w.currentText()
        return params

    def _update_preview(self):
        self._debounce_timer.start()

    def _do_update_preview(self):
        if self.data is None:
            return
        name = self.strategy_combo.currentText()
        if not name:
            return
        try:
            params = self._get_current_params()

            def _compute():
                strategy = registry.create(name, **params)
                signals = strategy.generate_signals(self.data)
                if isinstance(self.data.columns, pd.MultiIndex):
                    ticker = self.data.columns.levels[0][0]
                    close = self.data[(ticker, "Close")]
                else:
                    close = (
                        self.data["Close"] if "Close" in self.data.columns else self.data.iloc[:, 0]
                    )
                preview_len = min(300, len(self.data))
                chart_data = pd.DataFrame(
                    {"Close": close.tail(preview_len)}, index=self.data.index[-preview_len:]
                )
                signals_df = pd.DataFrame(
                    {"signal": signals.tail(preview_len)}, index=self.data.index[-preview_len:]
                )
                return chart_data, signals_df

            self._preview_thread = run_in_thread(
                self,
                _compute,
                on_finished=self._on_preview_computed,
            )
        except Exception:
            pass

    def _on_preview_computed(self, result):
        chart_data, signals_df = result
        buy_count = int((signals_df["signal"] == 1).sum())
        sell_count = int((signals_df["signal"] == -1).sum())
        self.buy_label.setText(f"买入信号: {buy_count}")
        self.sell_label.setText(f"卖出信号: {sell_count}")

        while self.chart_layout.count():
            child = self.chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        canvas = price_chart(chart_data, signals_df)
        self.chart_layout.addWidget(canvas)
