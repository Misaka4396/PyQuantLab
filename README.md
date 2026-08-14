# PyQuantLab — Python 量化交易平台

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI: PyQt5](https://img.shields.io/badge/UI-PyQt5-orange.svg)]()
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![version](https://img.shields.io/badge/version-1.2.6-blueviolet)]()
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)]()
[![tests](https://img.shields.io/badge/tests-164%20passed-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)]()

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
| 🗄️ 回测数据治理 | point-in-time 数据层（as_of 时间戳防前视）、幸存者偏差处理、数据质量报告 |
| ⚙️ 事件驱动引擎 | engine/ 事件循环（信号→订单→撮合→核算），多标的、插件回调、可复现 |
| 💰 成本与滑点 | 佣金/印花税/过户费/半价差/冲击成本/篮子加成，逐笔明细可审计 |
| 📈 绩效评估增强 | Sharpe/Sortino/Calmar/VaR-CVaR + 月度热力图 + HTML 报告 + trade 导出 |
| 🔄 ETF 套利 | IOPV/NAV/PCF 数据层、折溢价信号与阈值、篮子同步执行、申赎模拟、套利回测 |
| 🧠 ML/DL 建模 | 防泄漏特征流水线、Purged CV + Walk-forward、LightGBM 基线 + PyTorch LSTM/Transformer |
| 🛡️ 过拟合检测 | Deflated Sharpe / PBO / CSCV 过拟合评估与上线建议 |

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
        return {
            "param1": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "step": 1,
                "label": "参数1",
                "help": "说明",
            }
        }

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
├── cost_model.py        # 交易成本与滑点模型（A3）
├── cost_config.py       # 成本参数集中配置
├── data/                # 行情/缓存/编排 + A1 数据治理（pit/survivorship/data_loader/quality_report/schemas）
├── strategy/            # 策略基类 + 4 内置策略
├── engine/              # A2 事件驱动回测引擎（events/order/matching/portfolio/accounting）
├── backtest/            # 向量化回测 + 绩效指标 + A4 报告（metrics_enhanced/report）
├── portfolio/           # 组合优化 + 蒙特卡洛
├── etf/                 # B ETF 套利专项（数据层/信号/篮子执行/回测集成）
├── ml/                  # C ML/DL 专项（特征/CV/训练/过拟合检测）
├── core/                # 核心类型与异常
└── ui/
    ├── main_window.py   # PyQt5 主窗口
    ├── charts_qt.py     # Matplotlib 图表工厂
    ├── pages_qt/        # 6 个 PyQt5 页面组件
    ├── pages/           # 5 个 Streamlit 页面
    └── components/      # Streamlit 图表组件
```

### 量化研究增强层（v1.2.0 新增）

| 专项 | 模块 | 说明 |
|------|------|------|
| A 通用底座 | `data/pit.py` 等 | point-in-time 数据治理，杜绝未来函数与幸存者偏差 |
| A 通用底座 | `engine/` | 事件驱动回测引擎，为 ETF 篮子下单与 ML 滚动重训预留扩展点 |
| A 通用底座 | `cost_model.py` | 佣金/印花税/过户费/价差/冲击/篮子成本，逐笔明细可审计 |
| A 通用底座 | `backtest/metrics_enhanced.py` | 完整绩效指标 + 一键 HTML/PNG 报告 |
| B ETF 套利 | `etf/` | IOPV/NAV/PCF 对齐、折溢价信号、篮子同步执行、套利回测（理想 vs 真实执行双模式） |
| C ML/DL | `ml/` | 防泄漏特征、Purged CV + Walk-forward、LightGBM/PyTorch 训练、过拟合检测（DSR/PBO/CSCV） |

## 修改日志

详见 [CHANGELOG.md](CHANGELOG.md)。

## 质量与工程化

| 维度 | 状态 |
|------|------|
| 测试体系 | **164 用例**三层金字塔：单元 146 + 集成 8 + E2E 冒烟 10（`pytest tests/ -q`） |
| 覆盖率门禁 | **≥80%**（`pyproject.toml` fail_under=80） |
| Lint/Format | ruff（0 错误）+ pre-commit 提交前自动检查 |
| 提交规范 | Conventional Commits（commitizen 强制，commit-msg hook） |
| CI | GitHub Actions：lint → 测试 → 覆盖率门禁（`.github/workflows/ci.yml`） |
| API 文档 | [pdoc 在线文档](docs/api/index.html)，`scripts/build_docs.ps1` 一键重建 |

## 打包发布（exe）

桌面版与 ML 训练器为**独立 exe + 共享 DLL 目录**结构，发布时三者缺一不可：

```
dist/
├── PyQuantLab/          # 主程序 GUI（双击 PyQuantLab.exe）
├── PyQuantLab_ML/       # ML/DL 训练器（命令行 ml-trainer）
└── PyQuantLab_common/   # 共享 mkl 数学库（两 exe 共用，勿删）
```

- 主程序：`dist\PyQuantLab\PyQuantLab.exe` 双击运行
- ML 训练器：`dist\PyQuantLab_ML\PyQuantLab_ML.exe train/registry/overfit --help`
- 重新打包：`python -m PyInstaller PyQuantLab_Qt.spec --noconfirm`（主程序）/ `PyQuantLab_ML.spec`（ML）；共享目录用 `scripts/build_common.ps1`

## API 文档

[在线 API 文档](docs/api/index.html)（pdoc 生成：`powershell -File scripts/build_docs.ps1`）。

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（开发环境 / 代码规范 / 测试 / 提交规范 / PR 流程），并遵守 [行为准则](CODE_OF_CONDUCT.md)。

## License

MIT License — 详见 [LICENSE](LICENSE)
