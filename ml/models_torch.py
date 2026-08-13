"""C3 PyTorch 模型（LSTM / Transformer）+ 严格时序切分训练循环（CPU 可跑）。

设计要点：
- 严格时序切分：训练/验证由 ``ml.torch_dataset.train_val_split_by_time`` 按时间先后
  切分；``DataLoader(shuffle=False)``，**禁止跨时间 shuffle**。
- CPU 训练，模型规模适中（hidden_size 默认 32，层数 1），单测 < 60s。
- 固定种子（torch/numpy），保证可复现。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.run_config import DL_LSTM, DL_TRANSFORMER
from ml.torch_dataset import TimeSeriesDataset, train_val_split_by_time


def set_seed(seed: int) -> None:
    """固定 torch / numpy 随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


class LSTMModel(nn.Module):
    """LSTM 时序模型：取最后一步隐状态做预测。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # out: (B, seq_len, hidden)
        return self.fc(out[:, -1, :])  # 取最后一步


class TransformerModel(nn.Module):
    """Transformer 时序模型：输入投影 + encoder，取最后一步做预测。"""

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.fc(h[:, -1, :])


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    output_size: int = 1,
) -> nn.Module:
    """按名称构造 PyTorch 模型。"""
    if model_name == DL_LSTM:
        return LSTMModel(input_size, hidden_size, num_layers, dropout, output_size)
    if model_name == DL_TRANSFORMER:
        return TransformerModel(
            input_size, d_model=hidden_size, nhead=4,
            num_layers=num_layers, dropout=dropout, output_size=output_size,
        )
    raise ValueError(f"未知模型类型: {model_name!r}")


def _eval_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            pred = model(xb)
            total += loss_fn(pred, yb).item() * len(xb)
            n += len(xb)
    return total / n if n else 0.0


def train_torch_model(
    model: nn.Module,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, list]:
    """训练循环（MSE 损失，回归口径；分类可改用 BCEWithLogits）。

    训练集 ``shuffle=False``：时序数据禁止跨时间 shuffle。
    """
    set_seed(seed)
    device = torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history["train_loss"].append(running / len(train_dataset))
        history["val_loss"].append(_eval_loss(model, val_loader, loss_fn, device))
    return history


def predict_torch_model(
    model: nn.Module,
    dataset: TimeSeriesDataset,
    batch_size: int = 32,
) -> np.ndarray:
    """用训练好的模型对数据集做预测（按时间顺序，无 shuffle）。"""
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def train_torch_from_arrays(
    X,
    y,
    model_name: str = DL_LSTM,
    seq_len: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Dict:
    """端到端：时序切分 → 构建 Dataset → 训练，返回模型 + 历史。"""
    Xtr, ytr, Xva, yva = train_val_split_by_time(X, y, train_frac)
    train_ds = TimeSeriesDataset(Xtr, ytr, seq_len)
    val_ds = TimeSeriesDataset(Xva, yva, seq_len)

    input_size = int(np.asarray(Xtr).shape[1])
    model = build_model(model_name, input_size, hidden_size, num_layers, dropout)
    history = train_torch_model(
        model, train_ds, val_ds, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    return {"model": model, "history": history, "train_dataset": train_ds, "val_dataset": val_ds}


__all__ = [
    "LSTMModel",
    "TransformerModel",
    "build_model",
    "train_torch_model",
    "predict_torch_model",
    "train_torch_from_arrays",
    "set_seed",
]
