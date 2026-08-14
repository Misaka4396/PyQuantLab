"""pytest-qt 全局配置：offscreen 平台（无显示环境跑 GUI 冒烟测试）。"""

import os

# 必须在 QApplication 创建前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
