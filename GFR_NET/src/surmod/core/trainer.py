"""Simplified training loop with mask support."""

import os
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm
import yaml
from pathlib import Path


def train_model(
    model,
    dataloader,
    config,
    process_batch,
    save_path=None,
    trial=None,
    report_interval=None,
    fold=None,
    timestamp=datetime.now().strftime("%Y%m%d_%H%M")
):
    """
    Train a model. Accepts a DataLoader object.
    Supports optional axis masks if present in the data.
    """
    verbose = config["common"].get("verbose", True)
    cuda_nb = config["common"].get("cuda_nb", 0)
    max_time = config["common"].get("max_time", float("inf"))

    training_cfg = config["training"]
    epochs = training_cfg.get("epochs", 1500)
    patience = training_cfg.get("patience", epochs)
    patience_start = training_cfg.get("patience_start", 500)
    batch_size = training_cfg.get("batch_size", 512)
    lr = training_cfg.get("lr", 0.0003)
    weight_decay = training_cfg.get("w_decay", 1e-5)
    metric = training_cfg.get("metric", "weighted_mse")

    # Device selection
    if torch.cuda.is_available() and cuda_nb < torch.cuda.device_count():
        device = torch.device(f"cuda:{cuda_nb}")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # Get current split from dataloader
    data = dataloader.get_data()

    x_train = data["X_train"].to(device)
    y_train = data["Y_train"].to(device)
    w_train = data["W_train"].to(device)
    x_val = data["X_val"].to(device)
    y_val = data["Y_val"].to(device)
    w_val = data["W_val"].to(device)
    axis_norm = data["axis_norm"].to(device)
    axis_mask_train = data["axis_mask_train"].to(device)
    axis_mask_val = data["axis_mask_val"].to(device)

    # Keep CPU copies of axis (so future users only need raw geometry to predict)
    axis = data.get("axis").cpu() if data.get("axis") is not None else None
    axis_norm_cpu = axis_norm.cpu()

    # Get normalizers (stored on DataLoader, not in the returned split dict)
    x_normalizer = dataloader.get_x_normalizer()
    y_normalizer = dataloader.get_y_normalizer()
    n_geom = x_train.shape[1]  # number of geometry input features

    # ====================== VERBOSE STARTUP SUMMARY ======================
    if verbose:
        print("\n" + "=" * 72)
        print(" TRAINING STARTUP SUMMARY")
        print("=" * 72)
        print(f" Model : {getattr(model, 'name', type(model).__name__)}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" Trainable params : {n_params:,}")
        print(f" Device : {device}")
        print("-" * 72)
        print(f" Metric : {metric}")
        print(f" Epochs : {epochs:,} | Batch size : {batch_size:,}")
        print(f" Learning rate : {lr} | Weight decay: {weight_decay}")
        print(f" Patience : {patience} (starts after epoch {patience_start})")
        print("-" * 72)
        n_train_geom = x_train.shape[0]
        n_val_geom = x_val.shape[0]
        n_test_geom = data.get("X_test").shape[0] if data.get("X_test") is not None else 0
        n_freq = axis_norm.shape[0]
        n_train_points = int(axis_mask_train.sum().item())
        n_val_points = int(axis_mask_val.sum().item())
        total_labeled = n_train_points + n_val_points
        print(f" Geometries : Train= {n_train_geom} | Val= {n_val_geom} | Test= {n_test_geom}")
        print(f" Frequency points : {n_freq}")
        print(f" Labeled points : Train= {n_train_points} | Val= {n_val_points}")
        print(f" Total labeled : {total_labeled} (train + val, after masking)")
        print("=" * 72 + "\n")

    train_ds = TensorDataset(x_train, y_train, w_train, axis_mask_train)
    val_ds = TensorDataset(x_val, y_val, w_val, axis_mask_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Paths  always use the *basename* of the data file so absolute paths
    # (common in automated tests) do not break the results/ hierarchy.
    fold_id = trial.number if trial is not None else fold

    model_name = config['common']['model_name']
    prefix = config["common"].get("save_name") or ""
    data_stem = Path(config["common"]["data_file"]).stem
    base = os.path.join(
        save_path,
        "results",
        data_stem,
        f"{prefix}_{model_name}_{timestamp}",
    )

    if model_name in ["active_gfr_net", "active_gfr"] or fold_id is not None:
        base = os.path.join(base + '_tuning', f"trials/trial_{fold_id}")
        config["common"]["trial_nb"] = fold_id

    os.makedirs(base, exist_ok=True)

    model_path = os.path.join(base, f"{model.name}.pth")
    history_path = os.path.join(base, f"train_metrics.csv")
    config_path = os.path.join(base, "config.yaml")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    # Optimizer & Scheduler (always Cosine Warm Restarts)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=training_cfg.get("cosine_T0", 500), T_mult=2
    )

    history = []
    best_loss = float("inf")
    no_improve = 0
    training_start = time.time()

    progress = (
        tqdm(range(epochs), desc="Training", leave=True, dynamic_ncols=True)
        if verbose else range(epochs)
    )

    for epoch in progress:
        # === Training ===
        model.train()
        train_loss_sum = 0.0
        train_total_weight = 0.0

        for batch in train_loader:
            bx, by, bw, mask = batch

            optimizer.zero_grad()
            loss, train_loss_sum, train_total_weight, _ = process_batch(
                model=model,
                bx=bx,
                by=by,
                bw=bw,
                axis_norm=axis_norm,
                train_sse=train_loss_sum,
                train_total_weight=train_total_weight,
                device=device,
                config=config,
                mask=mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # === Validation ===
        model.eval()
        val_loss_sum = 0.0
        val_total_weight = 0.0

        with torch.no_grad():
            for batch in val_loader:
                bx, by, bw, mask = batch

                _, val_loss_sum, val_total_weight, _ = process_batch(
                    model=model,
                    bx=bx,
                    by=by,
                    bw=bw,
                    axis_norm=axis_norm,
                    train_sse=val_loss_sum,
                    train_total_weight=val_total_weight,
                    device=device,
                    config=config,
                    mask=mask,
                )

        train_loss = train_loss_sum / train_total_weight if train_total_weight > 1e-8 else float("nan")
        val_loss = val_loss_sum / val_total_weight if val_total_weight > 1e-8 else float("nan")

        elapsed_min = (time.time() - training_start) / 60

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "time_minutes": elapsed_min
        })

        # Save best checkpoint (now includes axis + axis_norm)
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            if model_path is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "n_geom": n_geom,
                        "x_normalizer": x_normalizer.state_dict(),
                        "y_normalizer": y_normalizer.state_dict(),
                        "axis": axis,
                        "axis_norm": axis_norm_cpu,
                        "best_val_loss": best_loss,
                        "best_epoch": epoch,
                    },
                    model_path,
                )
        elif epoch > patience_start:
            no_improve += 1

        if verbose:
            progress.set_postfix({
                f"Train {metric}": f"{train_loss:.5f}",
                f"Val {metric}": f"{val_loss:.5f}",
                f"Best {metric}": f"{best_loss:.5f}",
            })

        # Optuna support
        if trial is not None and report_interval and (epoch + 1) % report_interval == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()  # noqa: F821  only reached when trial is provided

        if no_improve >= patience or elapsed_min * 60 >= max_time:
            break

    # Save history as CSV next to weights
    if history_path is not None:
        pd.DataFrame(history).to_csv(history_path, index=False)

    return best_loss
