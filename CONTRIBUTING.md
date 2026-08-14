# 贡献指南（Contributing）

感谢您对 **PyQuantLab** 的关注与贡献！本指南帮助您高效参与项目开发，请先阅读并遵守。

## 目录

- [开发环境](#开发环境)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [提交规范](#提交规范)
- [分支与 PR 流程](#分支与-pr-流程)
- [功能开发流程](#功能开发流程)

---

## 开发环境

1. 需要 **Python 3.10+**
2. 克隆仓库并安装依赖：

```bash
git clone https://github.com/Misaka4396/PyQuantLab.git
cd PyQuantLab
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt        # 开发工具链
pip install torch --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU 版
```

3. 安装 pre-commit（提交前自动 lint/format）：

```bash
pre-commit install
pre-commit run --all-files   # 手动全量检查
```

## 代码规范

- **格式与风格**：由 `ruff` 统一管理（配置见 `pyproject.toml`，行宽 100）
- **命名**：变量/函数 snake_case、类 PascalCase、常量 UPPER_CASE
- **类型**：公共函数/类必须类型注解（PEP 604 语法 `X | None`）
- **注释**：中文注释；docstring 说明模块/类/函数职责；禁止废话注释
- **检查命令**：

```bash
ruff check .          # lint（须 0 错误）
ruff format --check . # 格式（须全部已格式化）
mypy .                # 类型检查（宽松模式，报告不阻断）
```

> 新代码提交前必须通过 `pre-commit` 全部检查。

## 测试要求

- 单元测试放 `tests/`，集成测试放 `tests/integration/`，命名 `test_*.py`
- 测试函数命名 `test_<行为描述>`，确定性优先（合成数据 + 固定种子，禁止网络/时间依赖）
- 覆盖率门禁：**≥80%**（范围：A/B/C 新专项模块，见 `pyproject.toml`）

```bash
pytest tests/ -q                      # 全量测试
pytest tests/ --cov --cov-report=term # 覆盖率（fail_under=80）
```

## 提交规范

使用 **Conventional Commits** 格式：`type(scope): subject`

| type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `docs` | 文档变更 |
| `refactor` | 重构（无功能变化） |
| `test` | 测试变更 |
| `build`/`ci` | 构建/CI 配置 |
| `chore` | 杂项（工具链、依赖等） |

```text
✅ feat(engine): 支持限价单部分成交
✅ fix(etf): 修正折溢价开仓阈值计算
❌ 修复bug
❌ update code
```

正文可补充说明（动机/影响），版本变更同步更新 `CHANGELOG.md`（Keep a Changelog 格式）。

## 分支与 PR 流程

- **主干开发**（trunk-based）：所有开发基于 `main`，短周期分支 + 快速合并
- 分支命名：`feature/<名称>` / `fix/<名称>` / `docs/<名称>`
- PR 要求：
  1. 标题遵循 Conventional Commits
  2. 关联相关 Issue（`Fixes #123`）
  3. CI 全部通过（lint + 测试 + 覆盖率门禁）
  4. 至少 1 位维护者 approve
- 合并策略：**Squash merge**（保持 main 历史整洁）

## 功能开发流程

1. 从 `main` 切分支：`git checkout -b feature/my-feature`
2. 开发 + 测试（先写测试再实现，或 TDD）
3. 本地验证：`pre-commit run --all-files && pytest tests/ --cov`
4. 提交（遵循提交规范）→ 推送 → 创建 PR（用模板）
5. 合并后删除分支

---

再次感谢您的贡献！如有疑问请通过 Issue 提出。
