"""C3 模型注册/版本管理：模型与数据/特征/超参版本绑定，可回滚。

存储布局：``root/``
- ``_registry.json``：版本索引（版本号 → 元数据，含当前版本指针）。
- ``v{version}.joblib``：sklearn/LightGBM 模型（joblib 序列化）。
- ``v{version}.pt``：PyTorch 模型（torch.save）。

元数据绑定：``data_version`` / ``feature_version`` / ``hyperparams``（含哈希），
保证"模型 ↔ 数据 ↔ 特征 ↔ 超参"可追溯；``rollback`` 加载历史版本并把当前指针切回。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def hyperparams_hash(hyperparams: dict) -> str:
    """超参哈希（模型与超参版本绑定用）。"""
    payload = json.dumps(hyperparams, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ModelMeta:
    """单个模型版本的元数据。"""

    version: int
    model_type: str
    task: str
    data_version: int
    feature_version: int
    hyperparams: dict = field(default_factory=dict)
    seed: int = 0
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    hyperparams_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "task": self.task,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "hyperparams": self.hyperparams,
            "seed": self.seed,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "hyperparams_hash": self.hyperparams_hash,
        }


class ModelRegistry:
    """模型版本仓库（保存 / 加载 / 回滚 / 列表）。"""

    def __init__(self, root: str | Path = "./data_cache/models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "_registry.json"

    # ------------------------------------------------------------------
    def save(self, model: Any, meta: dict) -> int:
        """保存模型与元数据，返回新版本号（并设为当前版本）。"""
        index = self._load_index()
        version = max((int(k) for k in index["versions"]), default=0) + 1

        model_type = str(meta.get("model_type", type(model).__name__))
        file_path = self._save_model(model, version, model_type)

        record = ModelMeta(
            version=version,
            model_type=model_type,
            task=str(meta.get("task", "")),
            data_version=int(meta.get("data_version", 0)),
            feature_version=int(meta.get("feature_version", 0)),
            hyperparams=dict(meta.get("hyperparams", {})),
            seed=int(meta.get("seed", 0)),
            metrics=dict(meta.get("metrics", {})),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        record.hyperparams_hash = hyperparams_hash(record.hyperparams)

        index["versions"][str(version)] = {"file": file_path.name, **record.to_dict()}
        index["current"] = version
        self._write_index(index)
        return version

    # ------------------------------------------------------------------
    def load(self, version: int) -> tuple[Any, dict]:
        """加载指定版本，返回 (模型, 元数据 dict)。"""
        index = self._load_index()
        key = str(version)
        if key not in index["versions"]:
            raise KeyError(f"版本不存在: {version}")
        rec = index["versions"][key]
        model = self._load_model(self.root / rec["file"])
        return model, rec

    def rollback(self, version: int) -> tuple[Any, dict]:
        """回滚到历史版本：加载并把当前版本指针切回。"""
        model, meta = self.load(version)
        index = self._load_index()
        index["current"] = int(version)
        self._write_index(index)
        return model, meta

    def current_version(self) -> int | None:
        """返回当前版本号（无则 None）。"""
        cur = self._load_index().get("current")
        return int(cur) if cur is not None else None

    def latest_version(self) -> int | None:
        """返回最新（最大）版本号。"""
        versions = list(self._load_index()["versions"].keys())
        return max(int(v) for v in versions) if versions else None

    def list_versions(self) -> pd.DataFrame:
        """全部版本的元数据表。"""
        index = self._load_index()
        rows = []
        for v in sorted(index["versions"], key=int):
            rec = index["versions"][v]
            rows.append(
                {
                    "version": rec["version"],
                    "model_type": rec["model_type"],
                    "task": rec["task"],
                    "data_version": rec["data_version"],
                    "feature_version": rec["feature_version"],
                    "seed": rec["seed"],
                    "hyperparams_hash": rec["hyperparams_hash"],
                    "metrics": rec["metrics"],
                    "created_at": rec["created_at"],
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _save_model(self, model: Any, version: int, model_type: str) -> Path:
        if _is_torch_module(model):
            path = self.root / f"v{version}.pt"
            import torch

            torch.save(model.state_dict(), path)
            return path
        path = self.root / f"v{version}.joblib"
        joblib.dump(model, path)
        return path

    def _load_model(self, path: Path):
        if path.suffix == ".pt":
            return torch_load_state(path)
        return joblib.load(path)

    # ------------------------------------------------------------------
    def _load_index(self) -> dict:
        if not self._index_path.exists():
            return {"versions": {}, "current": None}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict) -> None:
        self._index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _is_torch_module(model: Any) -> bool:
    try:
        import torch

        return isinstance(model, torch.nn.Module)
    except Exception:
        return False


def torch_load_state(path: Path):
    """加载 torch state_dict 为普通 dict（不依赖具体模型类，便于回滚审计）。"""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "ModelMeta",
    "ModelRegistry",
    "hyperparams_hash",
]
