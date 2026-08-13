"""PyQuantLab ML/DL 训练器 CLI（独立打包入口，方案 B 拆分产物）。

用法::

    ml-trainer train --data features.parquet --label fwd_ret_5 --time ts \\
                     --model lgb --task classification --epochs 5 --out models/
    ml-trainer registry list
    ml-trainer registry rollback 3
    ml-trainer overfit --is is_equity.csv --oos oos_equity.csv --trials 20 --out reports/

子命令：
- train    LightGBM 基线 / PyTorch LSTM / Transformer 训练（严格时序切分，模型版本化）
- registry 模型版本管理（list / rollback）
- overfit  过拟合检测报告（DSR / PBO / OOS-IS 对比，输出 markdown）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.overfit import assess_overfitting
from ml.overfit_report import generate_overfit_report
from ml.run_config import TrainConfig
from ml.train import train


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"不支持的格式: {p.suffix}（仅支持 parquet/csv）")


def _resolve_columns(df: pd.DataFrame, features, label, time_col):
    """特征/标签/时间列解析。"""
    cols = list(df.columns)
    if not label:
        low = {str(c).lower(): c for c in cols}
        for key in ("label", "target", "y", "fwd_ret", "forward"):
            hit = [c for k, c in low.items() if k == key or k.startswith(key)]
            if hit:
                label = hit[0]
                break
    if label is None or label not in cols:
        raise ValueError(f"标签列 {label!r} 不存在；可用列: {cols}")
    if time_col and time_col not in cols:
        raise ValueError(f"时间列 {time_col!r} 不存在；可用列: {cols}")
    feat_cols = [c for c in (features or cols) if c not in (label, time_col)]
    feat_cols = [c for c in feat_cols if c in cols]
    if not feat_cols:
        raise ValueError("特征列为空（请用 --features 指定）")
    return feat_cols, label, time_col


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    df = _load_frame(args.data)
    feat_cols, label, time_col = _resolve_columns(df, args.features, args.label, args.time)
    X = df[feat_cols].to_numpy()
    y = df[label].to_numpy()
    times = pd.to_datetime(df[time_col]) if time_col else None

    cfg = TrainConfig(
        seed=args.seed,
        task=args.task,
        use_lightgbm=args.model in ("lgb", "all"),
        use_dl=args.model in ("lstm", "transformer", "all"),
        use_rl=False,
        dl_model=args.model if args.model in ("lstm", "transformer") else "lstm",
        dl_epochs=args.epochs,
        model_dir=args.out,
    )
    print(f"[train] 样本={len(X)} 特征={len(feat_cols)} 模型={args.model} 任务={args.task} 种子={args.seed}")
    result = train(cfg, X, y, times=times)
    print("[train] 完成。模型版本:")
    for m, v in zip(result.get("models", []), result.get("versions", [])):
        name = m if isinstance(m, str) else type(m).__name__
        print(f"  - {name} v{v}")
    print(f"[train] 模型目录: {args.out}")
    return 0


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def cmd_registry(args: argparse.Namespace) -> int:
    from ml.model_registry import ModelRegistry

    reg = ModelRegistry(args.dir)
    if args.action == "list":
        index = reg._load_index()
        versions = index.get("versions", {})
        if not versions:
            print("[registry] 无已保存模型")
            return 0
        for ver in sorted(versions, key=int):
            rec = versions[ver]
            print(
                f"  v{ver}  {rec.get('model_type','?'):<20} task={rec.get('task','?'):<15} "
                f"data_v={rec.get('data_version','?')} feat_v={rec.get('feature_version','?')} "
                f"seed={rec.get('seed','?')} metrics={json.dumps(rec.get('metrics',{}), ensure_ascii=False)}"
            )
        print(f"[registry] 当前版本: {index.get('current_version')}")
        return 0
    if args.action == "rollback":
        ver = reg.rollback(int(args.version))
        print(f"[registry] 已回滚到 v{ver}")
        return 0
    raise ValueError(f"未知操作: {args.action}")


# ---------------------------------------------------------------------------
# overfit
# ---------------------------------------------------------------------------
def cmd_overfit(args: argparse.Namespace) -> int:
    def _load_seq(path: str):
        df = _load_frame(path)
        return df.iloc[:, 0].to_numpy()

    is_seq = _load_seq(args.is_seq) if args.is_seq else None
    oos_seq = _load_seq(args.oos_seq) if args.oos_seq else None
    assessment = assess_overfitting(
        is_equity=is_seq,
        oos_equity=oos_seq,
        trials=args.trials,
    )
    out_path = generate_overfit_report(assessment, args.out, name=args.name)
    print(f"[overfit] 风险等级: {assessment.risk_level} | 上线建议: {assessment.recommendation}")
    for key, val in (("DSR", assessment.dsr), ("PBO", assessment.pbo),
                     ("OOS/IS Sharpe衰减", assessment.sharpe_degradation)):
        print(f"[overfit] {key} = {val:.4f}" if val is not None else f"[overfit] {key} = —")
    print(f"[overfit] 报告已生成: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-trainer", description="PyQuantLab ML/DL 训练器")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="训练模型（LightGBM / LSTM / Transformer）")
    tr.add_argument("--data", required=True, help="特征数据 parquet/csv")
    tr.add_argument("--features", nargs="*", default=None, help="特征列（默认除 label/time 外全部）")
    tr.add_argument("--label", default=None, help="标签列")
    tr.add_argument("--time", default=None, help="时间列")
    tr.add_argument("--model", choices=["lgb", "lstm", "transformer", "all"], default="lgb")
    tr.add_argument("--task", choices=["classification", "regression"], default="classification")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=3, help="DL 训练轮数")
    tr.add_argument("--out", default="./data_cache/models", help="模型目录")
    tr.set_defaults(func=cmd_train)

    rg = sub.add_parser("registry", help="模型版本管理")
    rg.add_argument("action", choices=["list", "rollback"])
    rg.add_argument("version", nargs="?", type=int, help="rollback 目标版本")
    rg.add_argument("--dir", default="./data_cache/models", help="模型目录")
    rg.set_defaults(func=cmd_registry)

    of = sub.add_parser("overfit", help="过拟合检测报告")
    of.add_argument("--is", dest="is_seq", default=None, help="IS 权益/收益序列 csv")
    of.add_argument("--oos", dest="oos_seq", default=None, help="OOS 权益/收益序列 csv")
    of.add_argument("--trials", type=int, default=1, help="DSR 试验次数 N")
    of.add_argument("--out", default="./reports", help="报告输出目录")
    of.add_argument("--name", default="overfit_report")
    of.set_defaults(func=cmd_overfit)

    return p


def main(argv=None) -> int:
    # Windows 控制台 UTF-8 输出（避免中文乱码）
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001 - CLI 顶层错误捕获
        print(f"[错误] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
