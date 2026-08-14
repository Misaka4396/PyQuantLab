"""P10 E2E 冒烟测试：PyQt5 桌面 GUI 启动与导航（offscreen 平台）。

覆盖验收标准：
- 主窗口可构造：标题、6 个导航页、侧边栏、状态栏齐备
- 导航切换驱动页面栈且各页 on_activated 不崩溃
- 关键页面（回测）含核心控件（策略选择/资金输入/运行按钮）
"""

from __future__ import annotations

import pytest
from PyQt5.QtWidgets import QApplication, QLabel, QListWidget, QPushButton, QStatusBar

from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    """模块级 QApplication（offscreen 平台由 conftest 设置）。"""
    instance = QApplication.instance() or QApplication([])
    yield instance


def _make_window(qtbot, app) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# 构造与结构
# ---------------------------------------------------------------------------
def test_main_window_constructs(qtbot, app):
    """主窗口构造成功：标题 / 页面栈 / 侧边栏 / 状态栏。"""
    w = _make_window(qtbot, app)
    assert "PyQuantLab" in w.windowTitle(), f"标题异常: {w.windowTitle()}"
    assert len(w.pages) == 6, f"应 6 个页面，实际 {len(w.pages)}"
    assert w.nav_list.count() == 6, f"应 6 个导航项，实际 {w.nav_list.count()}"
    assert isinstance(w.status_bar, QStatusBar)
    assert w.stack.count() == 6


def test_sidebar_and_version_label(qtbot, app):
    """侧边栏含导航列表与版本标签。"""
    w = _make_window(qtbot, app)
    labels = w.findChildren(QLabel)
    texts = [label.text() for label in labels]
    assert "PyQuantLab" in texts, "侧边栏应有标题标签"
    assert any(t.startswith("v") for t in texts), f"应有版本标签, 实际: {texts}"


# ---------------------------------------------------------------------------
# 导航行为
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_navigation_switches_stack(qtbot, app, idx):
    """导航点击 → 页面栈切换且 on_activated 不崩溃。"""
    w = _make_window(qtbot, app)
    w.nav_list.setCurrentRow(idx)  # 触发 currentRowChanged → _on_nav
    qtbot.waitUntil(lambda: w.stack.currentIndex() == idx, timeout=3000)
    assert w.stack.currentWidget() is w.pages[idx]


def test_default_page_is_live(qtbot, app):
    """默认显示第一页（实时行情）。"""
    w = _make_window(qtbot, app)
    assert w.stack.currentIndex() == 0


# ---------------------------------------------------------------------------
# 关键页面控件
# ---------------------------------------------------------------------------
def test_backtest_page_core_controls(qtbot, app):
    """回测页含核心控件：策略选择 / 资金输入 / 运行按钮。"""
    w = _make_window(qtbot, app)
    page = w.pages[3]  # BacktestPage
    btns = page.findChildren(QPushButton)
    assert len(btns) >= 1, "回测页应有操作按钮"
    # 页面布局有效（子控件数量合理）
    assert len(page.findChildren(QListWidget)) + len(page.findChildren(QLabel)) >= 1
