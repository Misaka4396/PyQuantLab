"""共享 DLL 加载 hook（方案 A：主程序与 ML 训练器共用 mkl 等重型二进制）。

发布包结构::

    dist/
    ├── PyQuantLab/            # 主程序（onedir）
    ├── PyQuantLab_ML/         # ML 训练器（onedir）
    └── PyQuantLab_common/     # 共享 DLL（mkl_*.dll 等，两 exe 共用）

本 hook 在解释器启动早期执行：将共享目录加入 DLL 搜索路径，
使 numpy/scipy 等运行时能加载 mkl 数学库（打包时已从各自 _internal 移除）。
"""

import contextlib
import os
import sys

_COMMON_DIRNAME = "PyQuantLab_common"


def _locate_common() -> str | None:
    """定位共享目录：exe 位于 dist/<App>/ 下，共享目录为 dist/PyQuantLab_common/。"""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    # 兼容 onedir（exe 在 dist/<App>/）与直接运行（源码环境跳过）
    dist_root = os.path.dirname(exe_dir)
    candidate = os.path.join(dist_root, _COMMON_DIRNAME)
    if os.path.isdir(candidate):
        return candidate
    # 备用：exe 同级目录下的 common
    fallback = os.path.join(exe_dir, _COMMON_DIRNAME)
    if os.path.isdir(fallback):
        return fallback
    return None


def _install():
    common = _locate_common()
    if common is None:
        return  # 无共享目录（如源码环境），跳过
    with contextlib.suppress(Exception):
        os.add_dll_directory(common)  # Python 3.8+ 官方 DLL 搜索机制
    # 兼容老式 LoadLibrary 搜索顺序
    os.environ["PATH"] = common + os.pathsep + os.environ.get("PATH", "")


_install()
