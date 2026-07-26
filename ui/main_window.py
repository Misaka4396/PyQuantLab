"""PyQuantLab 主窗口 — PyQt5 桌面应用."""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QStatusBar,
)

from ui.pages_qt.data_page_qt import DataPage
from ui.pages_qt.strategy_page_qt import StrategyPage
from ui.pages_qt.backtest_page_qt import BacktestPage
from ui.pages_qt.results_page_qt import ResultsPage
from ui.pages_qt.portfolio_page_qt import PortfolioPage
from ui.pages_qt.live_page_qt import LivePage

FONT_FAMILY = "Microsoft YaHei"
FONT_SIZE = 13
FONT_SIZE_LARGE = 16
FONT_SIZE_TITLE = 20


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQuantLab — 量化交易平台")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # 全局默认字体
        font = QFont(FONT_FAMILY, FONT_SIZE)
        QApplication.setFont(font)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # 侧边栏
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)

        title = QLabel("PyQuantLab")
        title_font = QFont(FONT_FAMILY, 22)
        title_font.setBold(True)
        title.setFont(title_font)
        sidebar_layout.addWidget(title)

        subtitle = QLabel("量化交易平台")
        subtitle.setObjectName("subtitle")
        sub_font = QFont(FONT_FAMILY, 11)
        subtitle.setFont(sub_font)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addSpacing(12)

        self.nav_list = QListWidget()
        nav_font = QFont(FONT_FAMILY, 14)
        self.nav_list.setFont(nav_font)
        nav_items = [
            "📡 实时行情",
            "📥 数据管理",
            "📊 策略配置",
            "▶️ 运行回测",
            "📈 结果分析",
            "🏦 组合模拟",
        ]
        for item in nav_items:
            self.nav_list.addItem(item)
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self._on_nav)
        sidebar_layout.addWidget(self.nav_list)
        sidebar_layout.addStretch()

        version_label = QLabel("v1.0.0")
        version_label.setObjectName("version")
        ver_font = QFont(FONT_FAMILY, 10)
        version_label.setFont(ver_font)
        sidebar_layout.addWidget(version_label)

        # 页面栈
        self.stack = QStackedWidget()
        self.pages = [
            LivePage(),
            DataPage(),
            StrategyPage(),
            BacktestPage(),
            ResultsPage(),
            PortfolioPage(),
        ]
        for p in self.pages:
            self.stack.addWidget(p)

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        status_font = QFont(FONT_FAMILY, 12)
        self.status_bar.setFont(status_font)

        self.setStyleSheet(self._style())

    def _on_nav(self, index: int):
        self.stack.setCurrentIndex(index)
        self.pages[index].on_activated()

    def _style(self) -> str:
        return f"""
        QMainWindow {{ background: #1e1e2e; }}
        QWidget {{ background: #1e1e2e; color: #cdd6f4; font-family: "{FONT_FAMILY}"; font-size: {FONT_SIZE}px; }}
        #sidebar {{ background: #181825; border-right: 1px solid #313244; }}
        QListWidget {{ background: #181825; border: none; outline: none; font-size: 14px; }}
        QListWidget::item {{ padding: 12px 16px; border-radius: 8px; margin: 3px 8px; }}
        QListWidget::item:selected {{ background: #45475a; color: #cdd6f4; }}
        QListWidget::item:hover {{ background: #313244; }}
        QLabel#subtitle {{ color: #a6adc8; font-size: 11px; margin-bottom: 8px; }}
        QLabel#version {{ color: #585b70; font-size: 10px; }}
        QPushButton {{ background: #89b4fa; color: #1e1e2e; border: none; padding: 10px 22px;
                      border-radius: 8px; font-weight: bold; font-size: 14px; font-family: "{FONT_FAMILY}"; }}
        QPushButton:hover {{ background: #b4d0fb; }}
        QPushButton:pressed {{ background: #74a8f7; }}
        QComboBox {{ background: #313244; border: 1px solid #45475a; padding: 8px;
                    border-radius: 6px; min-width: 130px; font-size: 13px; font-family: "{FONT_FAMILY}"; }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{ font-size: 14px; font-family: "{FONT_FAMILY}"; }}
        QLineEdit {{ background: #313244; border: 1px solid #45475a; padding: 8px;
                    border-radius: 6px; font-size: 13px; font-family: "{FONT_FAMILY}"; }}
        QSpinBox, QDoubleSpinBox {{ background: #313244; border: 1px solid #45475a;
                                   padding: 8px; border-radius: 6px; font-size: 13px; font-family: "{FONT_FAMILY}"; }}
        QTableWidget {{ background: #1e1e2e; border: 1px solid #313244; gridline-color: #313244; font-size: 13px; font-family: "{FONT_FAMILY}"; }}
        QTableWidget::item {{ padding: 5px 10px; }}
        QHeaderView::section {{ background: #181825; border: none; padding: 8px;
                               border-bottom: 2px solid #45475a; font-size: 13px; font-weight: bold; font-family: "{FONT_FAMILY}"; }}
        QTabWidget::pane {{ border: 1px solid #313244; }}
        QTabBar::tab {{ background: #181825; padding: 10px 22px; margin-right: 3px; font-size: 14px; font-family: "{FONT_FAMILY}"; }}
        QTabBar::tab:selected {{ background: #313244; border-bottom: 3px solid #89b4fa; }}
        QProgressBar {{ border: 1px solid #313244; border-radius: 6px; text-align: center; font-size: 12px; }}
        QProgressBar::chunk {{ background: #89b4fa; border-radius: 4px; }}
        QStatusBar {{ background: #181825; color: #a6adc8; font-size: 12px; font-family: "{FONT_FAMILY}"; }}
        QGroupBox {{ border: 1px solid #313244; border-radius: 8px; margin-top: 14px;
                    padding-top: 18px; font-size: 14px; font-weight: bold; font-family: "{FONT_FAMILY}"; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 8px; }}
        QCheckBox {{ spacing: 8px; font-size: 13px; }}
        QSlider::groove:horizontal {{ height: 8px; background: #313244; border-radius: 4px; }}
        QSlider::handle:horizontal {{ background: #89b4fa; width: 18px; margin: -5px 0; border-radius: 9px; }}
        QDateEdit {{ background: #313244; border: 1px solid #45475a; padding: 8px; border-radius: 6px; font-size: 13px; font-family: "{FONT_FAMILY}"; }}
        QScrollArea {{ border: none; }}
        """


def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
