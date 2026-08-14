"""数据管理页面 — 下载与查看股票数据."""

from datetime import datetime, timedelta

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDateEdit,
    QGroupBox,
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
from data.manager import DataManager
from ui.charts_qt import price_chart, returns_histogram
from ui.worker import run_in_thread

FONT_FAMILY = "Microsoft YaHei"


class DataPage(QWidget):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()
        self.data: pd.DataFrame = None
        self._thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("数据管理")
        header.setFont(QFont(FONT_FAMILY, 22, QFont.Bold))
        header.setStyleSheet("margin-bottom: 6px;")
        layout.addWidget(header)

        ctrl_group = QGroupBox("数据源")
        ctrl = QHBoxLayout(ctrl_group)

        lbl_font = QFont(FONT_FAMILY, 13)

        lbl1 = QLabel("股票代码:")
        lbl1.setFont(lbl_font)
        ctrl.addWidget(lbl1)

        self.ticker_input = QComboBox()
        self.ticker_input.setEditable(True)
        self.ticker_input.addItems(DEFAULT_TICKERS)
        self.ticker_input.setCurrentText(",".join(DEFAULT_TICKERS))
        self.ticker_input.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.ticker_input, 1)

        lbl2 = QLabel("起始日期:")
        lbl2.setFont(lbl_font)
        ctrl.addWidget(lbl2)

        self.start_date = QDateEdit()
        self.start_date.setDate(datetime.now().date() - timedelta(days=365 * 3))
        self.start_date.setCalendarPopup(True)
        self.start_date.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.start_date)

        lbl3 = QLabel("结束日期:")
        lbl3.setFont(lbl_font)
        ctrl.addWidget(lbl3)

        self.end_date = QDateEdit()
        self.end_date.setDate(datetime.now().date())
        self.end_date.setCalendarPopup(True)
        self.end_date.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.end_date)

        lbl4 = QLabel("周期:")
        lbl4.setFont(lbl_font)
        ctrl.addWidget(lbl4)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1d", "1wk", "1mo"])
        self.interval_combo.setFont(QFont(FONT_FAMILY, 13))
        ctrl.addWidget(self.interval_combo)

        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self._load_data)
        ctrl.addWidget(self.load_btn)

        layout.addWidget(ctrl_group)

        self.status_label = QLabel("尚未加载数据。")
        self.status_label.setFont(QFont(FONT_FAMILY, 12))
        self.status_label.setStyleSheet("color: #a6adc8; margin: 6px 0;")
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        charts_layout = QHBoxLayout()
        self.price_chart_container = QWidget()
        self.price_layout = QVBoxLayout(self.price_chart_container)
        self.price_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.addWidget(self.price_chart_container, 2)

        self.ret_chart_container = QWidget()
        self.ret_layout = QVBoxLayout(self.ret_chart_container)
        self.ret_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.addWidget(self.ret_chart_container, 1)

        layout.addLayout(charts_layout, 1)

    def on_activated(self):
        pass

    def _load_data(self):
        tickers_str = self.ticker_input.currentText().strip()
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if not tickers:
            self.status_label.setText("请输入至少一个股票代码。")
            return

        self.load_btn.setEnabled(False)
        self.load_btn.setText("加载中...")
        self.status_label.setText(f"正在下载 {', '.join(tickers)} 的数据...")

        sd = self.start_date.date().toPyDate()
        ed = self.end_date.date().toPyDate()
        interval = self.interval_combo.currentText()

        self._thread = run_in_thread(
            self,
            self.dm.get_data,
            tickers,
            str(sd),
            str(ed),
            interval,
            on_finished=self._on_data_loaded,
            on_error=self._on_data_error,
        )

    def _on_data_loaded(self, data):
        self.data = data
        cache_info = self.dm.cache.get_stats()
        self.status_label.setText(
            f"已加载 {data.shape[0]} 行 × {data.shape[1]} 列  |  "
            f"缓存: {cache_info['total_files']} 个文件"
        )
        self._update_table()
        self._update_charts()
        self.load_btn.setEnabled(True)
        self.load_btn.setText("加载数据")

    def _on_data_error(self, err_msg):
        self.status_label.setText(f"错误: {err_msg}")
        self.load_btn.setEnabled(True)
        self.load_btn.setText("加载数据")

    def _update_table(self):
        if self.data is None:
            return
        if isinstance(self.data.columns, pd.MultiIndex):
            ticker = self.data.columns.levels[0][0]
            df = self.data[ticker].tail(500)
        else:
            df = self.data.tail(500)

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for row in range(len(df)):
            for col in range(len(df.columns)):
                val = df.iloc[row, col]
                item = QTableWidgetItem(f"{val:.4f}" if isinstance(val, float) else str(val))
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont(FONT_FAMILY, 12))
                self.table.setItem(row, col, item)

    def _update_charts(self):
        if self.data is None:
            return
        if isinstance(self.data.columns, pd.MultiIndex):
            ticker = self.data.columns.levels[0][0]
            close = self.data[(ticker, "Close")]
        elif "Close" in self.data.columns:
            close = self.data["Close"]
        else:
            close = self.data.iloc[:, 0]

        for container in [self.price_layout, self.ret_layout]:
            while container.count():
                child = container.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        chart_data = pd.DataFrame({"Close": close}, index=self.data.index.tail(500))
        pc = price_chart(chart_data)
        self.price_layout.addWidget(pc)

        rets = close.pct_change().dropna()
        rc = returns_histogram(rets)
        self.ret_layout.addWidget(rc)
