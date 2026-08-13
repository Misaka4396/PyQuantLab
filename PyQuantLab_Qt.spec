# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PyQuantLab Qt desktop version."""

import os
import site

PROJECT_DIR = SPECPATH  # noqa: F821
site_packages = site.getsitepackages()[0]

# Collect project Python modules
added_files = []
for root, dirs, files in os.walk(PROJECT_DIR):
    for f in files:
        if f.endswith('.py') and f != 'PyQuantLab_Qt.spec' and f != 'PyQuantLab.spec' \
                and 'runtime_hook.py' not in f:
            if '__pycache__' in root:
                continue
            full = os.path.join(root, f)
            rel_dir = os.path.relpath(os.path.dirname(full), PROJECT_DIR)
            added_files.append((full, rel_dir))

# Include Qt platform plugin (critical for PyQt5 exe)
qt_plugin_dir = None
for p in site.getsitepackages():
    candidate = os.path.join(p, 'PyQt5', 'Qt5', 'plugins')
    if os.path.isdir(candidate):
        qt_plugin_dir = candidate
        break

if qt_plugin_dir:
    added_files.append((os.path.join(qt_plugin_dir, 'platforms'), 'platforms'))
    added_files.append((os.path.join(qt_plugin_dir, 'styles'), 'styles'))

a = Analysis(
    ['launcher_qt.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'matplotlib.colors',
        'matplotlib.cm',
        'PIL',
        'PIL.Image',
        'PIL.ImageColor',
        'pandas',
        'numpy',
        'numpy.core',
        'numpy.linalg',
        'numpy.random',
        'yfinance',
        'scipy',
        'scipy.optimize',
        'scipy.optimize._minimize',
        'scipy.linalg',
        # ---- v1.2.0 ML/DL 专项（方案 B：完整自包含）----
        'torch',
        'torch.nn',
        'torch.utils.data',
        'torch.cuda',
        'lightgbm',
        'sklearn',
        'akshare',
        'akshare.stock',
        'curl_cffi',
        'mini_racer',
        'peewee',
        'websockets',
        'jsonpath',
        'xlrd',
        'multitasking',
        'joblib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'streamlit',
        'tornado',
        'altair',
        'pydeck',
        'plotly',
        'IPython',
        'jupyter',
        'notebook',
        'sqlalchemy',
        # 排除多余 Qt 绑定，防止与 PyQt5 冲突
        'PySide6',
        'PySide2',
        'PyQt6',
        'shiboken6',
        'shiboken2',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    a.zipfiles,
    name='PyQuantLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    icon='E:\\lianghua1\\xunzong.ico',
    console=False,
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
    name='PyQuantLab',
)
