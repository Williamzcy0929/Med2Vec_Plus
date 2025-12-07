#!/usr/bin/env python
import argparse, os, pickle, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score, auc
)

ASPECTS = ["dx", "proc", "treat"]

def load_vocab_sizes(train_dir: Path, vocab_file: str = "vocab.pkl") -> dict:
    with open(train_dir / vocab_file, "rb") as f:
        vocabs = pickle.load(f)
    return {a: len(vocabs[a]["id2code"]) for a in ASPECTS}

def iter_patients_from_seqs(seqs: list):
    buf = []
    for v in seqs:
        if isinstance(v, list) and len(v)==1 and v[0]==-1:
            if buf:
                yield buf
            buf = []
        else:
            buf.append(v if isinstance(v, list) else [])
    if buf:
        yield buf

def load_all_targets_for_aspect(processed_root: Path, aspect: str):
    acc_targets = []
    for split in ["train", "valid", "test"]:
        seq_path = processed_root / split / f"{aspect}.seqs.pkl"
        if not seq_path.exists():
            continue
        with open(seq_path, "rb") as f:
            seqs = pickle.load(f)
        for visits in iter_patients_from_seqs(seqs):
            if len(visits) < 2:
                continue
            for t in range(len(visits) - 1):
                acc_targets.append(visits[t + 1])  # next-visit label set
    return acc_targets  # list of list[int]

def compute_prevalence(processed_root: Path) -> dict:
    train_dir = processed_root / "train"
    V = load_vocab_sizes(train_dir)
    out = {}
    for a in ASPECTS:
        targets = load_all_targets_for_aspect(processed_root, a)
        n_pairs = len(targets)
        pos = sum(len(lst) for lst in targets)
        denom = max(1, n_pairs) * V[a]
        p = pos / denom if denom > 0 else float("nan")
        out[a] = {"prevalence": p, "n_pairs": n_pairs, "V": V[a], "pos_total": pos}
    return out

def plot_prevalence_bar(prevalence: dict, out_path: Path):
    aspects = list(prevalence.keys())
    vals = [prevalence[a]["prevalence"] for a in aspects]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_title("Prevalence of Positive Labels")
    ax.bar(aspects, vals)
    ax.set_yscale("log")
    ax.set_ylabel("Prevalence (Log Scale)")

    for i, v in enumerate(vals):
        txt = f"{v:.4%}" if v > 0 else "0%"
        ax.text(i, v, txt,
                ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)

def simulate_scores(N: int, prevalence: float, seed: int = 707):
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < prevalence).astype(int)
    n_pos = y.sum()
    if n_pos < 5:  # ensure a few positives
        y[:5] = 1
    # Separability: positives ~ Beta(5,2), negatives ~ Beta(2,5)
    pos_scores = rng.beta(5, 2, size=int(y.sum()))
    neg_scores = rng.beta(2, 5, size=int((1 - y).sum()))
    scores = np.zeros_like(y, dtype=float)
    scores[y == 1] = pos_scores
    scores[y == 0] = neg_scores
    return y, scores

def plot_pr_roc_simulation(prevalence: dict, out_path: Path, N: int = 100_000):
    fig, axes = plt.subplots(len(ASPECTS), 2, figsize=(10, 3.2 * len(ASPECTS)))
    if len(ASPECTS) == 1:
        axes = np.array([axes])

    for i, a in enumerate(ASPECTS):
        p = prevalence[a]["prevalence"]
        p = max(p, 1e-4)  # avoid extreme zeros
        y, s = simulate_scores(N, p, seed=707 + i)

        # PR curve
        pr_ax = axes[i, 0]
        precision, recall, _ = precision_recall_curve(y, s)
        auprc = average_precision_score(y, s)
        pr_ax.plot(recall, precision)
        pr_ax.hlines(p, 0, 1)  # random baseline
        pr_ax.set_xlim(0, 1); pr_ax.set_ylim(0, 1)
        pr_ax.set_xlabel("Recall"); pr_ax.set_ylabel("Precision")
        pr_ax.set_title(f"{a.upper()} PR curve | AUPRC={auprc:.3f} (baseline={p:.3%})")

        # ROC curve
        roc_ax = axes[i, 1]
        fpr, tpr, _ = roc_curve(y, s)
        auroc = roc_auc_score(y, s)
        roc_ax.plot(fpr, tpr)
        roc_ax.plot([0, 1], [0, 1])
        roc_ax.set_xlim(0, 1); roc_ax.set_ylim(0, 1)
        roc_ax.set_xlabel("FPR"); roc_ax.set_ylabel("TPR")
        roc_ax.set_title(f"{a.upper()} ROC curve | AUROC={auroc:.3f}")

    fig.suptitle("AUPRC vs AUROC under the same prevalence", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)

def main():
    ap = argparse.ArgumentParser(description="Why use AUPRC instead of AUROC (class imbalance visualization)")
    ap.add_argument("--processed_root", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--out_dir", type=str, default="out/eda")
    ap.add_argument("--sim_N", type=int, default=100000)
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev = compute_prevalence(processed_root)
    # Save numeric summary
    pd.DataFrame.from_dict(prev, orient="index").to_json(out_dir / "auprc_vs_auroc_prevalence.json", indent=2)

    # Plot prevalence bars (random AUPRC baseline)
    plot_prevalence_bar(prev, out_dir / "auprc_why_prevalence.png")

    # Simulated PR vs ROC at the same prevalence
    plot_pr_roc_simulation(prev, out_dir / "auprc_vs_auroc_simulation.png", N=args.sim_N)

    print("[why-auprc] Saved:")
    print(" -", out_dir / "auprc_why_prevalence.png")
    print(" -", out_dir / "auprc_vs_auroc_simulation.png")
    print(" -", out_dir / "auprc_vs_auroc_prevalence.json")

if __name__ == "__main__":
    main()