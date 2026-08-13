"""pytest 全局配置：把仓库根目录加入 sys.path，保证 data/core/config 可导入。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
