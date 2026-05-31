from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "code" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from model import StockTransformer  # noqa: E402
from predict import build_inference_sequences, preprocess_predict_data  # noqa: E402
from train import (  # noqa: E402
    RankingDataset,
    WeightedRankingLoss,
    collate_fn,
    create_ranking_dataset_vectorized,
    preprocess_data,
    set_seed,
    train_ranking_model,
)


@dataclass(frozen=True)
class BacktestWindow:
    fold: int
    signal_date: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    train_data_end: pd.Timestamp
    future_dates: list[pd.Timestamp]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a purged weekly walk-forward backtest with the official "
            "StockTransformer pipeline."
        )
    )
    parser.add_argument("--data", default="data/train.csv", help="Historical csv path.")
    parser.add_argument(
        "--output",
        default="backtesting/results/backtest_results.csv",
        help="Per-fold result csv path.",
    )
    parser.add_argument(
        "--summary-output",
        default="backtesting/results/backtest_summary.json",
        help="Summary json path.",
    )
    parser.add_argument("--start", default=None, help="Earliest evaluation start date.")
    parser.add_argument("--end", default=None, help="Latest evaluation end date.")
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Use the last N valid weekly folds. Set 0 to use all folds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs per fold. Keep small for quick experiments.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Evaluation horizon. The official helpers currently support 5.",
    )
    parser.add_argument(
        "--purge",
        type=int,
        default=5,
        help=(
            "Minimum trading-day gap between a training sample end date and "
            "the signal date. The effective purge is at least horizon."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=160,
        help="Skip folds with fewer visible trading days before training.",
    )
    parser.add_argument(
        "--allow-holiday-weeks",
        action="store_true",
        help="Allow evaluation windows whose 5 future dates are not Mon-Fri consecutive.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one fold fails.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available.")
    return torch.device(device_name)


def apply_runtime_config(args: argparse.Namespace) -> None:
    if args.horizon != 5:
        raise ValueError(
            "The current official label/dataset helpers are hard-coded to a 5-day horizon."
        )
    if args.sequence_length is not None:
        config["sequence_length"] = args.sequence_length
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    config["num_epochs"] = args.epochs


def load_price_data(path: Path) -> tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(path)
    if len(df.columns) < 3:
        raise ValueError(f"Expected at least 3 columns in {path}.")

    stock_col = df.columns[0]
    date_col = df.columns[1]
    open_col = df.columns[2]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad_rows = int(df[date_col].isna().sum())
        raise ValueError(f"Found {bad_rows} rows with invalid dates in {path}.")

    df = df.sort_values([stock_col, date_col]).reset_index(drop=True)
    return df, stock_col, date_col, open_col


def make_windows(
    df: pd.DataFrame,
    date_col: str,
    start: str | None,
    end: str | None,
    folds: int,
    horizon: int,
    purge: int,
    min_train_days: int,
    require_consecutive_calendar: bool,
) -> list[BacktestWindow]:
    dates = pd.Series(df[date_col].drop_duplicates()).sort_values().reset_index(drop=True)
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    effective_purge = max(purge, horizon)
    windows: list[BacktestWindow] = []

    for signal_idx in range(len(dates) - horizon):
        signal_date = pd.Timestamp(dates.iloc[signal_idx])
        future_dates = [
            pd.Timestamp(dates.iloc[signal_idx + offset])
            for offset in range(1, horizon + 1)
        ]

        if require_consecutive_calendar:
            future_days = np.array(future_dates, dtype="datetime64[D]")
            if not np.all(np.diff(future_days).astype(np.int64) == 1):
                continue

        eval_start = future_dates[0]
        eval_end = future_dates[-1]
        if start_ts is not None and eval_start < start_ts:
            continue
        if end_ts is not None and eval_end > end_ts:
            continue

        sample_end_idx = signal_idx - effective_purge
        train_data_end_idx = sample_end_idx + horizon
        if sample_end_idx < 0 or train_data_end_idx < 0:
            continue
        if train_data_end_idx + 1 < min_train_days:
            continue

        windows.append(
            BacktestWindow(
                fold=len(windows) + 1,
                signal_date=signal_date,
                eval_start=eval_start,
                eval_end=eval_end,
                train_data_end=pd.Timestamp(dates.iloc[train_data_end_idx]),
                future_dates=future_dates,
            )
        )

    if folds > 0:
        windows = windows[-folds:]
        windows = [
            BacktestWindow(
                fold=i + 1,
                signal_date=w.signal_date,
                eval_start=w.eval_start,
                eval_end=w.eval_end,
                train_data_end=w.train_data_end,
                future_dates=w.future_dates,
            )
            for i, w in enumerate(windows)
        ]

    return windows


def clean_and_scale(
    train_data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_data = train_data.copy()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    if train_data.empty:
        raise ValueError("No training rows remain after feature cleaning.")
    train_data.loc[:, features] = scaler.fit_transform(train_data[features])
    return train_data, scaler


def train_one_fold(
    train_raw: pd.DataFrame,
    stockid2idx: dict[Any, int],
    device: torch.device,
    seed: int,
) -> tuple[StockTransformer, StandardScaler, list[str]]:
    set_seed(seed)
    train_data, features = preprocess_data(train_raw, is_train=True, stockid2idx=stockid2idx)
    train_data, scaler = clean_and_scale(train_data, features)

    sequences, targets, relevance, stock_indices = create_ranking_dataset_vectorized(
        train_data,
        features,
        config["sequence_length"],
    )
    if len(sequences) == 0:
        raise ValueError("No ranking samples were created for this fold.")

    dataset = RankingDataset(sequences, targets, relevance, stock_indices)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    model = StockTransformer(
        input_dim=len(features),
        config=config,
        num_stocks=len(stockid2idx),
    ).to(device)

    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config["top5_weight"],
        pairwise_weight=config["pairwise_weight"],
        base_weight=config.get("base_weight", 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=max(1, config["num_epochs"]),
    )

    for epoch in range(config["num_epochs"]):
        train_loss, train_metrics = train_ranking_model(
            model,
            loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer=None,
        )
        scheduler.step()
        metric_text = ", ".join(
            f"{name}={value:.4f}" for name, value in train_metrics.items()
        )
        print(
            f"    epoch {epoch + 1}/{config['num_epochs']} "
            f"loss={train_loss:.4f}"
            + (f", {metric_text}" if metric_text else "")
        )

    return model, scaler, features


def predict_top_k(
    model: StockTransformer,
    scaler: StandardScaler,
    features: list[str],
    visible_raw: pd.DataFrame,
    stockid2idx: dict[Any, int],
    stock_col: str,
    signal_date: pd.Timestamp,
    top_k: int,
    device: torch.device,
) -> tuple[list[Any], list[float]]:
    processed, _ = preprocess_predict_data(visible_raw, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed.loc[:, features] = scaler.transform(processed[features])

    stock_ids = sorted(visible_raw[stock_col].drop_duplicates().tolist())
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        config["sequence_length"],
        stock_ids,
        signal_date,
    )

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1][:top_k]
    selected = [sequence_stock_ids[i] for i in order]
    selected_scores = [float(scores[i]) for i in order]
    return selected, selected_scores


def future_returns(
    df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    open_col: str,
    future_dates: list[pd.Timestamp],
) -> dict[Any, float]:
    future_set = set(future_dates)
    future_df = df[df[date_col].isin(future_set)].copy()
    returns: dict[Any, float] = {}

    for stock_id, group in future_df.groupby(stock_col, sort=False):
        group = group.sort_values(date_col)
        if len(group) < len(future_dates):
            continue
        start_open = float(group.iloc[0][open_col])
        end_open = float(group.iloc[-1][open_col])
        if abs(start_open) <= 1e-12:
            continue
        returns[stock_id] = (end_open - start_open) / start_open

    return returns


def score_selection(
    selected: list[Any],
    returns: dict[Any, float],
    top_k: int,
) -> dict[str, float]:
    if len(returns) < top_k:
        raise ValueError(f"Only {len(returns)} stocks have complete future returns.")

    all_returns = np.array(list(returns.values()), dtype=np.float64)
    random_return = float(np.mean(all_returns))
    optimal_return = float(np.mean(np.sort(all_returns)[::-1][:top_k]))

    weight = 1.0 / top_k
    portfolio_return = 0.0
    missing = 0
    selected_returns = []
    for stock_id in selected[:top_k]:
        stock_return = returns.get(stock_id)
        if stock_return is None:
            missing += 1
            continue
        selected_returns.append(float(stock_return))
        portfolio_return += weight * float(stock_return)

    denominator = optimal_return - random_return
    visible_proxy = (
        (portfolio_return - random_return) / denominator
        if abs(denominator) > 1e-12
        else 0.0
    )

    return {
        "portfolio_return": portfolio_return,
        "random_return": random_return,
        "optimal_return": optimal_return,
        "visible_proxy": float(visible_proxy),
        "selected_mean_return": float(np.mean(selected_returns)) if selected_returns else 0.0,
        "missing_selected": float(missing),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("status") == "ok"]
    if not ok_rows:
        return {"folds": len(results), "ok_folds": 0}

    visible = np.array([row["visible_proxy"] for row in ok_rows], dtype=np.float64)
    portfolio = np.array([row["portfolio_return"] for row in ok_rows], dtype=np.float64)
    return {
        "folds": len(results),
        "ok_folds": len(ok_rows),
        "mean_visible_proxy": float(np.mean(visible)),
        "median_visible_proxy": float(np.median(visible)),
        "std_visible_proxy": float(np.std(visible)),
        "min_visible_proxy": float(np.min(visible)),
        "positive_proxy_ratio": float(np.mean(visible > 0)),
        "mean_portfolio_return": float(np.mean(portfolio)),
        "median_portfolio_return": float(np.median(portfolio)),
    }


def write_outputs(
    results: list[dict[str, Any]],
    output_path: Path,
    summary_output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
    summary = summarize_results(results)
    summary_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_backtest(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    data_path = resolve_path(args.data)
    output_path = resolve_path(args.output)
    summary_output_path = resolve_path(args.summary_output)
    device = choose_device(args.device)

    df, stock_col, date_col, open_col = load_price_data(data_path)
    windows = make_windows(
        df=df,
        date_col=date_col,
        start=args.start,
        end=args.end,
        folds=args.folds,
        horizon=args.horizon,
        purge=args.purge,
        min_train_days=args.min_train_days,
        require_consecutive_calendar=not args.allow_holiday_weeks,
    )
    if not windows:
        raise ValueError("No valid backtest windows were found. Relax start/end or folds.")

    print(f"data: {data_path}")
    print(f"device: {device}")
    print(f"folds: {len(windows)}")
    print(
        "config: "
        f"sequence_length={config['sequence_length']}, "
        f"batch_size={config['batch_size']}, "
        f"epochs={config['num_epochs']}, "
        f"learning_rate={config['learning_rate']}"
    )

    results: list[dict[str, Any]] = []
    for window in windows:
        print(
            f"\n=== Fold {window.fold}/{len(windows)} | "
            f"signal={window.signal_date.date()} | "
            f"eval={window.eval_start.date()}~{window.eval_end.date()} ==="
        )
        try:
            train_raw = df[df[date_col] <= window.train_data_end].copy()
            visible_raw = df[df[date_col] <= window.signal_date].copy()
            stock_ids = sorted(visible_raw[stock_col].drop_duplicates().tolist())
            stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

            model, scaler, features = train_one_fold(
                train_raw=train_raw,
                stockid2idx=stockid2idx,
                device=device,
                seed=args.seed + window.fold,
            )
            selected, selected_scores = predict_top_k(
                model=model,
                scaler=scaler,
                features=features,
                visible_raw=visible_raw,
                stockid2idx=stockid2idx,
                stock_col=stock_col,
                signal_date=window.signal_date,
                top_k=args.top_k,
                device=device,
            )
            returns = future_returns(
                df=df,
                stock_col=stock_col,
                date_col=date_col,
                open_col=open_col,
                future_dates=window.future_dates,
            )
            scores = score_selection(selected, returns, args.top_k)

            row = {
                "fold": window.fold,
                "status": "ok",
                "signal_date": window.signal_date.date().isoformat(),
                "eval_start": window.eval_start.date().isoformat(),
                "eval_end": window.eval_end.date().isoformat(),
                "train_data_end": window.train_data_end.date().isoformat(),
                "selected_stocks": "|".join(map(str, selected)),
                "selected_scores": "|".join(f"{score:.8f}" for score in selected_scores),
                **scores,
                "error": "",
            }
            print(
                "    selected="
                f"{row['selected_stocks']} | "
                f"portfolio={scores['portfolio_return']:.6f} | "
                f"proxy={scores['visible_proxy']:.6f}"
            )
        except Exception as exc:
            row = {
                "fold": window.fold,
                "status": "failed",
                "signal_date": window.signal_date.date().isoformat(),
                "eval_start": window.eval_start.date().isoformat(),
                "eval_end": window.eval_end.date().isoformat(),
                "train_data_end": window.train_data_end.date().isoformat(),
                "selected_stocks": "",
                "selected_scores": "",
                "portfolio_return": np.nan,
                "random_return": np.nan,
                "optimal_return": np.nan,
                "visible_proxy": np.nan,
                "selected_mean_return": np.nan,
                "missing_selected": np.nan,
                "error": str(exc),
            }
            print(f"    fold failed: {exc}")
            if args.stop_on_error:
                raise

        results.append(row)
        write_outputs(results, output_path, summary_output_path)

    summary = summarize_results(results)
    print("\n=== Backtest Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nresults: {output_path}")
    print(f"summary: {summary_output_path}")


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    run_backtest(args)


if __name__ == "__main__":
    main()
