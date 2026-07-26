# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PyQuantLab."""

import os
from pathlib import Path

import site
import streamlit

# Use SPECPATH (provided by PyInstaller) as the project root
PROJECT_DIR = SPECPATH  # noqa: F821

# Get streamlit's installed path for bundled assets
streamlit_dir = Path(streamlit.__file__).parent
site_packages = Path(site.getsitepackages()[0])

# Collect all necessary data files
added_files = [
    (os.path.join(PROJECT_DIR, 'app.py'), '.'),
    (os.path.join(PROJECT_DIR, 'config.py'), '.'),
]

# Add package metadata (.dist-info) for importlib.metadata
for pkg_name in ['streamlit', 'altair', 'pandas', 'numpy', 'plotly', 'scipy', 'yfinance',
                  'jinja2', 'tornado', 'pydeck', 'pillow', 'requests', 'urllib3',
                  'certifi', 'idna', 'charset_normalizer', 'rich', 'click', 'gitpython',
                  'gitdb', 'smmap', 'watchdog', 'cachetools', 'protobuf',
                  'markdown_it_py', 'mdurl', 'pygments', 'blinker', 'validators']:
    dist_info = site_packages / f'{pkg_name}-*.dist-info'
    for match in site_packages.glob(f'{pkg_name}-*.dist-info'):
        added_files.append((str(match), match.name))

# Add all project modules
for root, dirs, files in os.walk(PROJECT_DIR):
    for f in files:
        if f.endswith('.py') and f not in ['launcher.py', 'PyQuantLab.spec']:
            if '__pycache__' in root:
                continue
            full = os.path.join(root, f)
            rel_dir = os.path.relpath(os.path.dirname(full), PROJECT_DIR)
            added_files.append((full, rel_dir))

# Streamlit's built-in frontend assets
streamlit_static = streamlit_dir / 'static'
if streamlit_static.exists():
    added_files.append((str(streamlit_static), 'streamlit/static'))

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # Streamlit internals
        'streamlit',
        'streamlit.web',
        'streamlit.web.bootstrap',
        'streamlit.web.server',
        'streamlit.web.server.websocket_headers',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.state',
        'streamlit.runtime.caching',
        'streamlit.commands',
        'streamlit.elements',
        'streamlit.proto',
        'streamlit.watcher',
        'streamlit.config',
        # Plotly
        'plotly',
        'plotly.express',
        'plotly.graph_objects',
        'plotly.io',
        'plotly.validators',
        # yfinance
        'yfinance',
        'yfinance.ticker',
        'yfinance.utils',
        # pandas / numpy
        'pandas',
        'pandas._libs',
        'numpy',
        'numpy.core',
        'numpy.linalg',
        'numpy.random',
        # scipy
        'scipy',
        'scipy.optimize',
        'scipy.optimize._minimize',
        'scipy.linalg',
        # Others
        'tornado',
        'tornado.web',
        'tornado.websocket',
        'jinja2',
        'altair',
        'pydeck',
        'PIL',
        # PyInstaller hooks
        'streamlit.web.server.server',
        'pkg_resources.py2_warn',
        'pkg_resources.markers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(PROJECT_DIR, 'runtime_hook.py')],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'sqlalchemy',
        'scipy.stats._stats_mstats_common',
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
    name='PyQuantLab',
)
