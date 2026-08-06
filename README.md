# PyQuantLab — Python 量化交易平台

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI: PyQt5](https://img.shields.io/badge/UI-PyQt5-orange.svg)]()
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()

一站式 **Python 量化交易平台**：覆盖从行情数据、策略研发、向量化回测、风险分析到组合优化的完整量化工作流。
提供 **PyQt5 原生桌面 GUI** 与 **Streamlit Web UI** 双界面，无需浏览器即可完成全部研究流程。

> ⚠️ 本项目仅供量化研究与教育学习使用，不构成任何投资建议。

---

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [扩展新策略](#扩展新策略)
- [项目结构](#项目结构)
- [修改日志](#修改日志)
- [License](#license)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 📡 实时行情 | 实时报价 + 分时图自动刷新（60s 轮询，间隔可配） |
| 🖥️ 原生桌面 GUI | PyQt5 实现，无需浏览器 |
| 🗄️ 数据管理 | yfinance 下载历史数据，本地 Parquet 缓存 |
| 🧠 策略引擎 | 内置 4 策略：均线交叉 / RSI / MACD / 布林带 |
| 🧩 插件架构 | 实现基类 + 注册即可添加新策略，UI 自动生成参数控件 |
| ⚡ 向量化回测 | 佣金 + 滑点建模，单策略/多策略对比 |
| 📊 绩效指标 | Sharpe / Sortino / Calmar / 最大回撤 / VaR-CVaR / alpha-beta |
| 🛡️ 风险分析 | 回撤分析、滚动指标、压力测试、尾部风险 |
| 🏦 组合优化 | 均值-方差（最大 Sharpe / 最小方差）、风险平价、等权 |
| 🎲 蒙特卡洛模拟 | Cholesky 分解前瞻组合模拟 |

## 快速开始

### 方式一：源码运行（桌面 GUI）

```bash
git clone https://github.com/Misaka4396/PyQuantLab.git
cd PyQuantLab
pip install -r requirements.txt
python launcher_qt.py
```

### 方式二：Web UI（Streamlit）

```bash
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 方式三：独立 EXE（免 Python）

```bash
pip install pyinstaller
pyinstaller PyQuantLab_Qt.spec --noconfirm
# 产物: dist/PyQuantLab/PyQuantLab.exe
```

## 使用指南

| 步骤 | 操作 |
|------|------|
| 1. 实时行情 | 输入代码监视实时价格，切换自动刷新与分时图周期 |
| 2. 数据管理 | 输入代码 + 日期范围下载历史数据，本地 Parquet 缓存 |
| 3. 策略配置 | 选择策略调整参数，买卖信号实时预览在价格图上 |
| 4. 运行回测 | 设置资金/佣金/滑点，单策略或全策略横向对比 |
| 5. 结果与风险 | 六大分析页：概览 / 权益曲线 / 回撤 / 交易分析 / 风险指标 / 压力测试 |
| 6. 组合模拟 | 优化权重（最大 Sharpe/最小方差/风险平价）+ 蒙特卡洛交互图表 |

## 扩展新策略

1. 创建 `strategy/my_strategy.py`，实现 `BaseStrategy`：

```python
from strategy.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    @classmethod
    def get_name(cls) -> str:
        return "My Strategy"

    @classmethod
    def get_param_spec(cls) -> dict:
        return {"param1": {"type": "int", "default": 10, "min": 1, "max": 100,
                           "step": 1, "label": "参数1", "help": "说明"}}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 返回: 1(买) / -1(卖) / 0(持有)
        pass
```

2. 在 `strategy/registry.py` 注册：

```python
from strategy.my_strategy import MyStrategy
registry.register(MyStrategy)
```

新策略自动出现在 UI 下拉框中，参数控件自动生成。

## 项目结构

```
PyQuantLab/
├── launcher_qt.py       # 桌面应用入口 (PyQt5)
├── launcher.py          # Streamlit Web 入口
├── app.py               # Streamlit 应用
├── config.py            # 全局配置
├── data/
│   ├── live_data.py     # 实时行情
│   ├── downloader.py    # yfinance 封装
│   ├── cache.py         # Parquet 缓存
│   ├── manager.py       # 数据编排
│   └── transform.py     # 技术指标
├── strategy/            # 策略基类 + 4 内置策略
├── backtest/            # 向量化回测 + 指标 + 风险
├── portfolio/           # 组合优化 + 蒙特卡洛
└── ui/
    ├── main_window.py   # PyQt5 主窗口
    ├── charts_qt.py     # Matplotlib 图表工厂
    ├── pages_qt/        # 6 个 PyQt5 页面组件
    ├── pages/           # 5 个 Streamlit 页面
    └── components/      # Streamlit 图表组件
```

## 修改日志

详见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT License — 详见 [LICENSE](LICENSE)