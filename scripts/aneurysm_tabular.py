#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def to_numeric_age(x: str) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # common formats: '64', '64Y', '064Y'
    s = s.replace('Y', '').replace('y', '')
    try:
        return float(s)
    except Exception:
        return np.nan


def sex_to_num(x: str) -> float:
    s = str(x).strip().lower()
    if s in ("m", "male"):
        return 1.0
    if s in ("f", "female"):
        return 0.0
    return np.nan


def sample_every_n_keep_k(df: pd.DataFrame, n: int = 12, k: int = 3) -> pd.DataFrame:
    # Keep first k rows in every block of n rows, in current order
    if n <= 0 or k <= 0:
        return df
    idx = np.arange(len(df))
    keep_mask = (idx % n) < k
    return df.loc[keep_mask].reset_index(drop=True)


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(y: np.ndarray, train_frac=0.7, val_frac=0.15, seed=42) -> Split:
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def _split(indices):
        n = len(indices)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val
        return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]

    tr_p, va_p, te_p = _split(idx_pos)
    tr_n, va_n, te_n = _split(idx_neg)

    train = np.concatenate([tr_p, tr_n])
    val = np.concatenate([va_p, va_n])
    test = np.concatenate([te_p, te_n])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return Split(train=train, val=val, test=test)


class LogisticModel(nn.Module):
    def __init__(self, in_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple AUC implementation (assumes binary labels 0/1)
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    # ranks of positives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # Use rank sum method
    ranks = np.arange(1, len(y_score) + 1)
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def train_modality_group(df: pd.DataFrame, modality: str, out_dir: str, device: str = "cpu") -> Dict:
    dfm = df[df["Modality"] == modality].reset_index(drop=True)
    # Select features and target
    X_age = dfm["PatientAge_num"].to_numpy().astype(np.float32)
    X_sex = dfm["PatientSex_num"].to_numpy().astype(np.float32)
    y = dfm["Aneurysm Present"].astype(int).to_numpy()

    # Handle missing: drop rows with NaN in features
    mask = ~np.isnan(X_age) & ~np.isnan(X_sex)
    X_age, X_sex, y = X_age[mask], X_sex[mask], y[mask]
    dfm = dfm.loc[mask].reset_index(drop=True)

    if len(y) < 10 or len(np.unique(y)) < 2:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"metrics_{modality}.json"), "w") as f:
            json.dump({"error": f"Insufficient data for modality {modality}"}, f)
        return {"modality": modality, "error": "insufficient_data"}

    # Normalize age
    age_mean, age_std = float(X_age.mean()), float(X_age.std() + 1e-6)
    X_age = (X_age - age_mean) / age_std
    X = np.stack([X_age, X_sex], axis=1).astype(np.float32)

    split = stratified_split(y, train_frac=0.7, val_frac=0.15, seed=42)

    def subset(indices):
        return X[indices], y[indices]

    Xtr, ytr = subset(split.train)
    Xva, yva = subset(split.val)
    Xte, yte = subset(split.test)

    # Datasets
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).float()), batch_size=64, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva).float()), batch_size=256)
    te_loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte).float()), batch_size=256)

    # Model
    model = LogisticModel(in_features=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float('inf')
    patience, bad = 20, 0

    for epoch in range(200):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())

        # val
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_targets = []
            for xb, yb in va_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_logits.append(logits.cpu().numpy())
                val_targets.append(yb.numpy())
            val_logits = np.concatenate(val_logits)
            val_targets = np.concatenate(val_targets)
            val_loss = float(crit(torch.from_numpy(val_logits), torch.from_numpy(val_targets)).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "age_mean": age_mean,
                "age_std": age_std,
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Test metrics
    def eval_loader(loader) -> Tuple[float, float, float]:
        logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                logits_all.append(logits.cpu().numpy())
                y_all.append(yb.numpy())
        logits_all = np.concatenate(logits_all)
        probs = 1 / (1 + np.exp(-logits_all))
        y_all = np.concatenate(y_all)
        auc = roc_auc_score_manual(y_all, probs)
        acc = float(((probs >= 0.5).astype(int) == y_all.astype(int)).mean())
        loss = float(crit(torch.from_numpy(logits_all), torch.from_numpy(y_all)).item())
        return auc, acc, loss

    auc_tr, acc_tr, loss_tr = eval_loader(tr_loader)
    auc_va, acc_va, loss_va = eval_loader(va_loader)
    auc_te, acc_te, loss_te = eval_loader(te_loader)

    # Save artifacts
    mdir = os.path.join(out_dir, f"modality_{modality}")
    os.makedirs(mdir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "age_mean": age_mean,
        "age_std": age_std,
        "in_features": 2,
    }, os.path.join(mdir, "model.pt"))

    metrics = {
        "modality": modality,
        "counts": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "train": {"auc": auc_tr, "acc": acc_tr, "loss": loss_tr},
        "val": {"auc": auc_va, "acc": acc_va, "loss": loss_va},
        "test": {"auc": auc_te, "acc": acc_te, "loss": loss_te},
    }
    with open(os.path.join(mdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Write predictions for entire subset with split tags
    def add_preds(indices: np.ndarray, split_name: str) -> pd.DataFrame:
        Xs = torch.from_numpy(X[indices])
        with torch.no_grad():
            logits = model(Xs).numpy()
        probs = 1 / (1 + np.exp(-logits))
        return pd.DataFrame({
            "SeriesInstanceUID": dfm.loc[indices, "SeriesInstanceUID"].values,
            "Modality": dfm.loc[indices, "Modality"].values,
            "PatientAge": dfm.loc[indices, "PatientAge"].values,
            "PatientSex": dfm.loc[indices, "PatientSex"].values,
            "label": y[indices],
            "prob": probs,
            "split": split_name,
        })

    pred_df = pd.concat([
        add_preds(split.train, "train"),
        add_preds(split.val, "val"),
        add_preds(split.test, "test"),
    ]).reset_index(drop=True)
    pred_df.to_csv(os.path.join(mdir, "predictions.csv"), index=False)

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--localizers_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--block_size", type=int, default=12, help="block size for sampling")
    p.add_argument("--keep_per_block", type=int, default=3, help="rows to keep per block")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_train = pd.read_csv(args.train_csv)
    df_loc = pd.read_csv(args.localizers_csv)

    # Merge on SeriesInstanceUID (left join to keep labels)
    df = df_train.merge(df_loc[["SeriesInstanceUID", "SOPInstanceUID", "location"]], on="SeriesInstanceUID", how="left")

    # Sampling: keep first K of each N rows
    df = df.sort_values(by=["SeriesInstanceUID", "SOPInstanceUID"], na_position="last")
    df = sample_every_n_keep_k(df, n=args.block_size, k=args.keep_per_block)

    # Prepare features
    df["PatientAge_num"] = df["PatientAge"].apply(to_numeric_age)
    df["PatientSex_num"] = df["PatientSex"].apply(sex_to_num)

    # Sanity: ensure label exists
    if "Aneurysm Present" not in df.columns:
        raise ValueError("Expected 'Aneurysm Present' as binary label in train.csv")

    # Train per modality
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "rows_after_sampling": int(len(df))}
    metrics_all = {}
    for modality in sorted(df["Modality"].dropna().unique().tolist()):
        metrics = train_modality_group(df, modality, args.output_dir, device=device)
        metrics_all[modality] = metrics

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({**summary, "metrics": metrics_all}, f, indent=2)


if __name__ == "__main__":
    main()

