# PyQuantLab — Claude Code 项目上下文

## 项目简介
PyQuantLab 是 Python 量化交易平台（PyQt5 桌面 GUI + Streamlit Web 双界面），现有模块：
- `data/`：实时行情、yfinance 下载、Parquet 缓存、数据编排、技术指标
- `strategy/`：策略基类 BaseStrategy + 4 内置策略（均线交叉/RSI/MACD/布林带），registry 注册制
- `backtest/`：向量化回测引擎 + 绩效指标 + 风险分析
- `portfolio/`：组合优化（均值-方差/风险平价/等权）+ 蒙特卡洛模拟
- `ui/`：PyQt5 主窗口 + 6 页面；Streamlit 页面
- `core/`：核心类型与异常

## 正在实施的新架构（2026-08，三套专项）
- **A 通用底座**（两套共用）：A1 回测数据层与治理（point-in-time、防前视/幸存者偏差）；A2 事件驱动回测引擎；A3 成本与滑点模型；A4 绩效评估与报告
- **B ETF 套利专项**：B1 ETF/IOPV/NAV/PCF 数据；B2 折溢价信号与阈值；B3 篮子同步执行 + 申赎模拟；B4 回测集成与评估
- **C ML/DL 专项**：C1 特征工程与标注（防泄漏）；C2 训练/验证（滚动回测 + Purged CV）；C3 模型训练与重训调度（LightGBM 基线 + PyTorch 深度学习）；C4 过拟合检测与评估

新模块目录约定：`engine/`（A2）、`cost_model.py`（A3）、`etf/`（B1-B4）、`ml/`（C1-C4）、`data/` 内扩展（A1）、`backtest/` 内扩展（A4）。

## 运行环境（Windows，PowerShell）
- **主 Python 环境**：`E:\Anaconda3-2026\python.exe`（Python 3.13，PyQt5 5.15.11、pandas 2.3.3、numpy 2.3.5、scipy、matplotlib、streamlit、plotly、pyarrow、pytest、torch CPU、lightgbm、scikit-learn、akshare、yfinance 均已安装）
- 备用环境：`E:\lianghua2\.venv\Scripts\python.exe`（无 PyQt5/torch）
- 终端是 PowerShell 5.1，不要用 `&` 调用符写法；pip 超时用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 代码规范
- 中文注释 + 类型提示；pandas/numpy 风格与现有模块一致
- 新功能模块必须与现有 UI 解耦：不修改 `ui/`、`launcher_qt.py`、`launcher.py`、`app.py` 的现有流程
- 数据一律本地 parquet 缓存优先，免费数据源优先（akshare/tushare/yfinance）
- 单元测试放 `tests/`，用 pytest
- 时间序列代码必须防前视：滚动窗口计算、禁全样本标准化、point-in-time 数据带 as_of 时间戳

## 关键命令
- 运行测试：`E:\Anaconda3-2026\python.exe -m pytest tests/ -x -q`
- 语法检查：`E:\Anaconda3-2026\python.exe -m compileall <module>`
