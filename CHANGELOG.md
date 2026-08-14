# PyQuantLab 修改日志

本项目使用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式。

## [v1.2.4] - 2026-08-13

### 新增：集成测试分层（P1-2）

- 新增 `tests/integration/` 目录，跨模块全链路集成测试（8 用例，合成数据 + 固定种子，离线确定性）
- `test_etf_arbitrage_pipeline.py`（B 链 4 用例）：A1 数据 → B1 折溢价 → B2 信号（机会窗口驱动开/平仓 + 前视防护）→ B3 篮子执行（A3 计费口径一致）→ B4 双模式回测（容量/压力/机会统计）
- `test_ml_pipeline.py`（C 链 4 用例）：C1 特征-标签对齐（t→t+h 可追溯）→ C2 Purged K-fold + Walk-forward 无泄漏 → C3 LightGBM 训练版本化 → C4 过拟合判定 + 报告落盘
- 全量回归：**154 测试通过**（146 单元 + 8 集成），覆盖率 84.70%

## [v1.2.3] - 2026-08-13

### 新增：开源治理（P1-1）

- 新增 `CONTRIBUTING.md`：开发环境 / 代码规范 / 测试要求 / Conventional Commits 提交规范 / trunk-based 分支与 PR 流程
- 新增 Issue 模板：`.github/ISSUE_TEMPLATE/bug_report.md`、`feature_request.md`
- 新增 `.github/PULL_REQUEST_TEMPLATE.md`：PR 检查清单（类型/关联 Issue/验证项）
- 新增 `CODE_OF_CONDUCT.md`：贡献者公约行为准则（中文版）
- README 增加「贡献」章节（链接 CONTRIBUTING/行为准则）

### 新增：工程化工具链（P0：ruff + pre-commit + CI）

- 新增 `pyproject.toml`：ruff（lint+format，line-length 100，中文项目忽略 RUF001/2/3）、mypy（宽松）、pytest、coverage 门禁（fail_under=80，范围限定 A/B/C 新专项）
- 全库 lint 修复：3102 错误 → 0（pyupgrade 现代化 Optional→`|`、unused-import、E741 变量名等）；86 文件 ruff format 统一格式
- 新增 `.pre-commit-config.yaml`：提交前自动 ruff --fix + format + 文件检查（经 ssh.github.com:443 拉取 hook，规避 github.com 封锁）
- 新增 `requirements-dev.txt`：pytest/pytest-cov/ruff/mypy/pre-commit；`requirements.txt` 补齐 lightgbm/scikit-learn/akshare
- 新增 `.github/workflows/ci.yml`：GitHub Actions（lint job：ruff check + format + mypy 报告；test job：torch CPU + pytest --cov 门禁 ≥80%）
- 修复 `core/types.py` 缺失 numpy 导入（F821）；`backtest/report.py` import 重构（Agg 模式保留 noqa）
- 实测：ruff check 0 错误、pre-commit 全过、**146 测试通过、覆盖率 84.34% ≥ 80% 门禁**

## [v1.2.2] - 2026-08-13

### 变更：共享 DLL 目录（方案 A）

- 新增 `dist/PyQuantLab_common/`：mkl 数学库（23 个 DLL，406MB）从两个 exe 移出，主程序与 ML 训练器**共享**同一份
- 新增 `runtime_hook_common.py`：exe 启动时经 `os.add_dll_directory` 加载共享目录，numpy/scipy 运行不受影响
- 新增 `scripts/build_common.ps1`：一键构建共享目录（从打包产物复制 mkl DLL）
- 体积：主程序 exe **347MB → 199MB**（-43%），ML 训练器 **419MB → 271MB**（-35%）
- 发布结构：`dist/PyQuantLab/` + `dist/PyQuantLab_ML/` + `dist/PyQuantLab_common/`（三者缺一不可）
- 验证：主程序 GUI 启动正常；ML 训练器 registry/overfit/train 全链路实测通过（mkl 自 common 加载）

## [v1.2.1] - 2026-08-13

### 变更：exe 发布拆分（方案 B）

- 原单 exe（722MB）拆分为双 exe，按需分发：
  - `dist/PyQuantLab/PyQuantLab.exe`：主程序（GUI + 回测 + ETF 数据/信号 + akshare），**347MB**（-52%）
  - `dist/PyQuantLab_ML/PyQuantLab_ML.exe`：ML/DL 训练器（torch CPU + lightgbm + sklearn，console CLI），**419MB**
- 新增 `ml_trainer_cli.py`：训练器 CLI 入口（`train` / `registry list|rollback` / `overfit` 三个子命令，Windows 控制台 UTF-8 输出）
- 新增 `PyQuantLab_ML.spec`：ML 训练器独立打包配置（排除 PyQt5/akshare/可视化冗余）
- `PyQuantLab_Qt.spec` 瘦身：移除 torch/lightgbm/sklearn 依赖，排除 QtWebEngine（-112MB）、panel/bokeh/botocore/cv2/numba 等 akshare 链冗余
- 验证：主程序 GUI 启动正常；ML 训练器 registry/overfit/train（LightGBM 全流程）实测通过

## [v1.2.0] - 2026-08-13

### 新增：A 通用底座（量化研究增强层）

- A1 回测数据层与数据治理：`data/pit.py`（point-in-time 对齐、as_of 时间戳防前视）、`data/survivorship.py`（幸存者偏差处理、退市/调出股生命周期追踪）、`data/data_loader.py`（统一加载入口、增量更新）、`data/quality_report.py`（数据质量报告）、`data/schemas.py` + `data/schemas.md`（parquet schema 文档）
- A2 事件驱动回测引擎：`engine/` 包（events/order/matching/portfolio/accounting/engine/config），支持多标的、next-bar 撮合无前视、插件回调机制、固定种子可复现
- A3 交易成本与滑点模型：`cost_model.py` + `cost_config.py`（佣金/印花税/过户费/半价差/冲击成本/篮子加成，逐笔明细可审计，参数集中配置）
- A4 绩效评估与报告：`backtest/metrics_enhanced.py`（收益/风险/交易/成本占比全指标）、`backtest/report.py` 扩展 `ReportGenerator`（权益/回撤/月度热力图/成本占比 + HTML 报告 + trade CSV 导出；旧版 `BacktestReport` 保留兼容 UI）

### 新增：B ETF 套利专项

- B1 ETF 套利数据层：`etf/etf_data.py`（IOPV/NAV/PCF 对齐、折溢价序列）、`etf/pcf_parser.py`（PCF 篮子解析）、`etf/iopv_estimator.py`（IOPV 合成估算）、`etf/demo_data/`
- B2 折溢价信号与阈值：`etf/etf_signal.py`（滚动 z-score 防前视、开仓阈值=成本+缓冲、强平规则）、`etf/threshold_config.py`、`etf/signal_grid.py`（滚动样本内网格优化）
- B3 篮子同步执行与申赎：`etf/basket_execution.py`（逐腿时延/部分成交模拟、敞口 tracking error、申赎开关默认关闭）、`etf/execution_config.py`
- B4 套利回测集成：`etf/etf_strategy.py` + `etf/etf_backtest.py`（挂载 A2 引擎、理想 vs 含同步风险双模式、容量分析、压力测试）

### 新增：C ML/DL 专项

- C1 特征工程与标注：`ml/features.py`（防泄漏特征流水线、滚动标准化、t→t+h 标签对齐、版本化存储）、`ml/leakage_audit.py`（三类泄漏审计）、`ml/features_schema.md`
- C2 训练/验证框架：`ml/cv.py`（Purged K-fold + embargo）、`ml/walk_forward.py`（滚动回测、重训调度、OOS 只报告一次）
- C3 模型训练与重训调度：`ml/models_lgb.py`（LightGBM 基线可复现）、`ml/models_torch.py`（PyTorch LSTM/Transformer）、`ml/torch_dataset.py`（时序窗口、禁跨时间 shuffle）、`ml/model_registry.py`（模型 sha256 版本绑定/回滚）、`ml/retrain_scheduler.py`、`ml/train.py`、`ml/run_config.py`
- C4 过拟合检测与评估：`ml/overfit.py`（Deflated Sharpe / PBO / CSCV）、`ml/feature_importance.py`（特征重要性跨 fold 稳定性）、`ml/overfit_report.py`（风险结论 + 上线建议）

### 其他

- `core/exceptions.py` 新增 `EngineError`
- `tests/` 新增 7 个测试文件（146 个用例全覆盖 A/B/C 三专项）

## [v1.1.0] - 2026-08-06

### 文档规范化
- 覆写 README.md：统一项目徽章、功能特性表、快速开始、使用指南、扩展策略、项目结构
- 新增 CHANGELOG.md 修改日志
- LICENSE 更新版权声明为 Misaka4396

### 工程化
- 补充 .gitignore 忽略 __pycache__/build/dist 等构建产物
- 通过 Git LFS 跟踪 exe 文件（见 .gitattributes）

## [v1.0.0] - 2026-07-26

### 初始版本
- 初始化提交: PyQuantLab 量化交易平台
- 配置 Git LFS 跟踪 exe 文件
- 添加 PyQuantLab 可执行文件 (Git LFS)
