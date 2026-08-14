"""实时行情页面 — 实时报价与日内走势图."""

from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULT_TICKERS
from data.live_data import LiveDataFetcher
from ui.charts_qt import live_price_chart
from ui.worker import run_in_thread

FONT_FAMILY = "Microsoft YaHei"


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        self.fetcher = LiveDataFetcher()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self._active = False
        self._price_history = {}
        self._quote_thread = None
        self._chart_thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("实时行情")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        desc = QLabel("实时报价与日内数据，每 60 秒自动刷新。")
        desc.setFont(QFont(FONT_FAMILY, 12))
        desc.setStyleSheet("color: #a6adc8; margin-bottom: 14px;")
        layout.addWidget(desc)

        ctrl = QHBoxLayout()
        lbl_font = QFont(FONT_FAMILY, 13)

        lbl = QLabel("股票代码:")
        lbl.setFont(lbl_font)
        ctrl.addWidget(lbl)

        self.ticker_input = QComboBox()
        self.ticker_input.setEditable(True)
        self.ticker_input.addItems(DEFAULT_TICKERS)
        self.ticker_input.setCurrentText("")
        self.ticker_input.setPlaceholderText("输入股票代码...")
        self.ticker_input.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.ticker_input, 1)

        self.add_btn = QPushButton("添加")
        self.add_btn.clicked.connect(self._add_ticker)
        ctrl.addWidget(self.add_btn)

        self.remove_btn = QPushButton("移除")
        self.remove_btn.clicked.connect(self._remove_ticker)
        ctrl.addWidget(self.remove_btn)

        self.toggle_btn = QPushButton("开始刷新")
        self.toggle_btn.clicked.connect(self._toggle_live)
        self.toggle_btn.setStyleSheet("background: #a6e3a1; color: #1e1e2e;")
        ctrl.addWidget(self.toggle_btn)

        lbl2 = QLabel("自选列表:")
        lbl2.setFont(lbl_font)
        ctrl.addWidget(lbl2)

        self.symbols_list = QComboBox()
        self.symbols_list.addItems(DEFAULT_TICKERS)
        self.symbols_list.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.symbols_list)

        lbl3 = QLabel("周期:")
        lbl3.setFont(lbl_font)
        ctrl.addWidget(lbl3)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m", "1h"])
        self.interval_combo.setCurrentText("5m")
        self.interval_combo.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.interval_combo)

        layout.addLayout(ctrl)

        self.quote_table = QTableWidget()
        self.quote_table.setColumnCount(6)
        self.quote_table.setHorizontalHeaderLabels(
            ["代码", "最新价", "涨跌", "涨跌幅", "成交量", "时间"]
        )
        self.quote_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.quote_table.setAlternatingRowColors(True)
        layout.addWidget(self.quote_table)

        self.last_refresh_label = QLabel("上次刷新: —")
        self.last_refresh_label.setFont(QFont(FONT_FAMILY, 11))
        self.last_refresh_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.last_refresh_label)

        chart_label = QLabel("日内走势图")
        chart_label.setFont(QFont(FONT_FAMILY, 16, QFont.Bold))
        chart_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(chart_label)

        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart_container, 1)

    def on_activated(self):
        if not self._active:
            self._active = True
            self._refresh_quotes()

    def _toggle_live(self):
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_btn.setText("开始刷新")
            self.toggle_btn.setStyleSheet("background: #a6e3a1; color: #1e1e2e;")
        else:
            self.timer.start(60000)
            self.toggle_btn.setText("停止刷新")
            self.toggle_btn.setStyleSheet("background: #f38ba8; color: #1e1e2e;")
            self._refresh()

    def _add_ticker(self):
        sym = self.ticker_input.currentText().strip().upper()
        if sym and sym not in [
            self.symbols_list.itemText(i) for i in range(self.symbols_list.count())
        ]:
            self.symbols_list.addItem(sym)
            self.symbols_list.setCurrentText(sym)
            self._refresh()

    def _remove_ticker(self):
        idx = self.symbols_list.currentIndex()
        if self.symbols_list.count() > 1 and idx >= 0:
            self.symbols_list.removeItem(idx)

    def _refresh(self):
        self._refresh_quotes()
        self._refresh_chart()

    def _refresh_quotes(self):
        self.toggle_btn.setEnabled(False)
        symbols = [self.symbols_list.itemText(i) for i in range(self.symbols_list.count())]
        self._quote_thread = run_in_thread(
            self,
            self.fetcher.get_current_quotes,
            symbols,
            on_finished=self._on_quotes_loaded,
            on_error=lambda e: self.toggle_btn.setEnabled(True),
        )

    def _on_quotes_loaded(self, quotes):
        self.quote_table.setRowCount(len(quotes))
        for row, (sym, q) in enumerate(quotes.items()):
            items = [
                QTableWidgetItem(sym),
                QTableWidgetItem(f"${q.price:.2f}"),
                QTableWidgetItem(f"{q.change:+.2f}"),
                QTableWidgetItem(f"{q.change_pct:+.2%}"),
                QTableWidgetItem(f"{q.volume:,}"),
                QTableWidgetItem(q.timestamp.strftime("%H:%M:%S")),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont(FONT_FAMILY, 13))
                if col in (2, 3):
                    if q.change < 0:
                        item.setForeground(Qt.red)
                    else:
                        item.setForeground(Qt.darkGreen)
                self.quote_table.setItem(row, col, item)
        self.last_refresh_label.setText(f"上次刷新: {datetime.now().strftime('%H:%M:%S')}")
        self.toggle_btn.setEnabled(True)

    def _refresh_chart(self):
        symbols = [self.symbols_list.itemText(i) for i in range(self.symbols_list.count())]
        interval = self.interval_combo.currentText()

        def _fetch_intraday():
            return self.fetcher.get_intraday(symbols, period="1d", interval=interval)

        self._chart_thread = run_in_thread(
            self,
            _fetch_intraday,
            on_finished=self._on_chart_loaded,
        )

    def _on_chart_loaded(self, df):
        if df.empty:
            return
        while self.chart_layout.count():
            child = self.chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        canvas = live_price_chart(df)
        self.chart_layout.addWidget(canvas)
