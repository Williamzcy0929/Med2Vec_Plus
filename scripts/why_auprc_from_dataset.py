#!/usr/bin/env python
import argparse, os, pickle
from pathlib import Path
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score
)

ASPECTS = ["dx", "proc", "treat"]

def load_vocab_size(train_dir: Path, aspect: str, vocab_file: str = "vocab.pkl") -> int:
    """Return vocabulary size for an aspect from train/vocab.pkl."""
    with open(train_dir / vocab_file, "rb") as f:
        vocabs = pickle.load(f)
    # Expect structure: {'dx': {'id2code': [...]}, ...}
    return len(vocabs[aspect]["id2code"])

def iter_patients_from_seqs(seqs: list):
    """
    Yield per-patient list of visits.

    Input seqs is a flat list where [-1] is used as a sentinel between patients.
    """
    buf = []
    for v in seqs:
        if isinstance(v, list) and len(v) == 1 and v[0] == -1:
            if buf:
                yield buf
            buf = []
        else:
            # Normalize current-visit list of ints (drop negative codes)
            if isinstance(v, (list, np.ndarray)):
                buf.append([int(c) for c in list(v) if int(c) >= 0])
            else:
                buf.append([])
    if buf:
        yield buf

def collect_next_visit_pairs_from_split(processed_root: Path, aspect: str, split: str) -> List[List[int]]:
    """
    Collect all next-visit label sets (current -> next pairs) for a split.

    Returns a list of lists; each element is the multi-label set for the next visit.
    """
    path = processed_root / split / f"{aspect}.seqs.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        seqs = pickle.load(f)
    pairs = []
    for visits in iter_patients_from_seqs(seqs):
        if len(visits) < 2:
            continue
        for t in range(len(visits) - 1):
            pairs.append(visits[t + 1])  # labels for next visit
    return pairs

def dataset_frequency_baseline(processed_root: Path, aspect: str, max_test_pairs: int = None):
    """
    Compute PR/ROC using a simple train-frequency baseline, evaluated on test.

    We mainly use this to illustrate:
      - how sparse the label space is (class imbalance)
      - why AUPRC is more informative than AUROC under severe imbalance.
    """
    train_dir = processed_root / "train"
    V = load_vocab_size(train_dir, aspect)
    assert V > 0, f"Empty vocab for {aspect}"

    # ----- Estimate marginal frequencies p_j over next-visit labels on train -----
    train_pairs = collect_next_visit_pairs_from_split(processed_root, aspect, "train")
    n_tr = len(train_pairs)
    if n_tr == 0:
        raise RuntimeError(f"No train pairs for {aspect}")
    freq = np.zeros(V, dtype=np.int64)
    for y in train_pairs:
        for j in y:
            if 0 <= j < V:
                freq[j] += 1
    p = freq / float(n_tr)  # shape [V]

    # ----- Build test label matrix and constant score matrix -----
    test_pairs = collect_next_visit_pairs_from_split(processed_root, aspect, "test")
    if max_test_pairs is not None and len(test_pairs) > max_test_pairs:
        rng = np.random.default_rng(707)
        idx = rng.choice(len(test_pairs), size=max_test_pairs, replace=False)
        test_pairs = [test_pairs[i] for i in idx]
    n_te = len(test_pairs)
    if n_te == 0:
        raise RuntimeError(f"No test pairs for {aspect}")

    mlb = MultiLabelBinarizer(classes=np.arange(V))
    Y = mlb.fit_transform(test_pairs).astype(np.int8)   # [N, V] binary labels
    S = np.broadcast_to(p[None, :], Y.shape)            # [N, V] scores (same row p)

    y_true = Y.ravel().astype(np.int8)
    y_score = S.ravel()

    # ----- Imbalance statistics -----
    # Average number of positive labels per next visit
    avg_labels_per_pair = float(Y.sum() / float(n_te))
    # Ratio of average positives to the total number of possible labels
    sparsity_ratio = avg_labels_per_pair / float(V)
    # Overall positive fraction across all labels and visits
    prevalence = float(y_true.mean())
    # These two should be numerically equal; keep both names for clarity
    assert np.allclose(prevalence, sparsity_ratio, atol=1e-10)

    # ----- Metrics and curves (micro-averaged) -----
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)

    # Sanity checks
    assert 0.0 <= auprc <= 1.0
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= prevalence <= 1.0

    return {
        "V": int(V),
        "n_train_pairs": int(n_tr),
        "n_test_pairs": int(n_te),
        # imbalance stats
        "prevalence": prevalence,                       # overall positive fraction
        "avg_labels_per_pair": avg_labels_per_pair,     # E[|Y_t|]
        "sparsity_ratio": sparsity_ratio,               # avg_labels_per_pair / V
        # curves and metrics
        "precision": precision, "recall": recall, "auprc": float(auprc),
        "fpr": fpr, "tpr": tpr, "auroc": float(auroc),
    }

def plot_curves(curves_by_aspect: dict, out_png: Path):
    """
    Plot ROC curves (AUROC) per aspect.

    We no longer draw PR curves. Each subplot shows:
      - the ROC curve with AUROC in the title
      - the prevalence annotated below the curve (inside the axes).
    """
    n = len(curves_by_aspect)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.0 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (a, cv) in enumerate(curves_by_aspect.items()):
        prev = cv["prevalence"]

        ax_roc = axes[i]
        ax_roc.plot(cv["fpr"], cv["tpr"], label="ROC curve")
        # Diagonal line = random classifier
        ax_roc.plot([0, 1], [0, 1], linestyle="dashed", label="Random")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title(f"{a.upper()} ROC | AUROC={cv['auroc']:.3f} | AUPRC={cv['sparsity_ratio']:.3f}")
        ax_roc.grid(alpha=0.3)

        # Annotate prevalence inside the axes, near the bottom-left
        ax_roc.text(
            0.05, 0.05,
            f"Prevalence = {prev:.3%}",
            transform=ax_roc.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

        ax_roc.legend(loc="lower right", fontsize=8)

    fig.suptitle("AUROC under Severe Class Imbalance", y=0.995, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)

def main():
    ap = argparse.ArgumentParser(
        description="Explore why AUPRC is more informative than AUROC under class imbalance"
    )
    ap.add_argument("--processed_root", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--out_png", type=str, default="out/eda/auprc_vs_auroc_frequency_baseline.png")
    ap.add_argument("--summary_json", type=str, default="out/eda/auprc_vs_auroc_frequency_baseline.json")
    ap.add_argument("--max_test_pairs", type=int, default=None,
                    help="Optional cap on the number of test pairs to speed up evaluation")
    ap.add_argument("--aspects", type=str, default="dx,proc,treat",
                    help="Comma-separated list of aspects to evaluate")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    aspects = [s.strip() for s in args.aspects.split(",") if s.strip()]

    curves_by_aspect = {}
    summary = {}
    for a in aspects:
        cv = dataset_frequency_baseline(processed_root, a, max_test_pairs=args.max_test_pairs)
        curves_by_aspect[a] = cv
        summary[a] = {
            "V": cv["V"],
            "n_train_pairs": cv["n_train_pairs"],
            "n_test_pairs": cv["n_test_pairs"],
            "prevalence": cv["prevalence"],
            "avg_labels_per_pair": cv["avg_labels_per_pair"],
            "sparsity_ratio": cv["sparsity_ratio"],  # equal to prevalence; kept for clarity
            "auprc": cv["auprc"],
            "auroc": cv["auroc"],
        }

    plot_curves(curves_by_aspect, Path(args.out_png))

    # Save numeric summary for later inspection in text/markdown
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[why-auprc] Saved:")
    print(" -", args.out_png)
    print(" -", args.summary_json)
    for a, s in summary.items():
        print(
            f"   [{a}] V={s['V']} | train_pairs={s['n_train_pairs']} | "
            f"test_pairs={s['n_test_pairs']} | "
            f"avg_labels/next_visit={s['avg_labels_per_pair']:.3f} | "
            f"ratio(avg_labels/V)={s['sparsity_ratio']:.6f} | "
            f"prevalence={s['prevalence']:.6f} | "
            f"AUPRC={s['auprc']:.4f} | AUROC={s['auroc']:.4f}"
        )

if __name__ == "__main__":
    main()