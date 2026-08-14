# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PyQuantLab ML/DL 训练器（方案 B 拆分产物，console CLI）。

打包 ml_trainer_cli.py → dist/PyQuantLab_ML/PyQuantLab_ML.exe
包含: torch(CPU)/lightgbm/sklearn + ml/ data/ core/ 模块；不含 PyQt5 GUI 与 akshare。
"""

import os

PROJECT_DIR = SPECPATH  # noqa: F821

# 收集项目 Python 模块（ml / data / core 三个研究目录，作为 datas 加载）
added_files = []
for sub in ("ml", "data", "core", "engine", "etf"):
    base = os.path.join(PROJECT_DIR, sub)
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                full = os.path.join(root, f)
                rel_dir = os.path.relpath(os.path.dirname(full), PROJECT_DIR)
                added_files.append((full, rel_dir))

a = Analysis(
    ['ml_trainer_cli.py'],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'torch',
        'torch.nn',
        'torch.utils.data',
        'torch.cuda',
        'lightgbm',
        'sklearn',
        'sklearn.model_selection',
        'joblib',
        'pandas',
        'numpy',
        'numpy.core',
        'numpy.linalg',
        'scipy',
        'scipy.stats',
        'matplotlib',
        'matplotlib.backends.backend_agg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(PROJECT_DIR, 'runtime_hook_common.py')],
    excludes=[
        # GUI 与 Web 全部排除（本产物为纯 CLI）
        'PyQt5',
        'PySide6',
        'PySide2',
        'PyQt6',
        'shiboken6',
        'shiboken2',
        'streamlit',
        'tornado',
        'plotly',
        'altair',
        'pydeck',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
        # 数据源（主程序负责 akshare）
        'akshare',
        'yfinance',
        'curl_cffi',
        'mini_racer',
        'peewee',
        # akshare 链冗余
        'panel',
        'bokeh',
        'holoviews',
        'param',
        'pyviz',
        'botocore',
        'boto3',
        'cv2',
        'skimage',
        'sphinx',
        'statsmodels',
        'numba',
        'llvmlite',
        'sqlalchemy',
        # 注: jinja2 是 torch 依赖，保留不排除
    ],
    noarchive=False,
)

# ---- 方案 A 共享 DLL：mkl 全套移出 exe 目录，运行时从 dist/PyQuantLab_common/ 加载 ----
a.binaries = [b for b in a.binaries if not (b[0].startswith("mkl_") and b[0].endswith(".dll"))]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    a.zipfiles,
    name='PyQuantLab_ML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PyQuantLab_ML',
)
