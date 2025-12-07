#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# --- helpers to iterate patient-wise from seqs with sentinel [-1]
def iter_patients_from_seqs(seqs):
    """
    Iterate over a flat sequence list and yield per-patient visit lists.

    The input `seqs` is a flat list where a sentinel [ -1 ] indicates
    the boundary between two patients.
    """
    buf = []
    for v in seqs:
        if isinstance(v, list) and len(v) == 1 and v[0] == -1:
            # End of one patient
            if buf:
                yield buf
            buf = []
        else:
            # v is a visit; ensure we store a list
            buf.append(v if isinstance(v, list) else [])
    if buf:
        # Last patient
        yield buf


def load_vocab(processed_root: Path, vocab_file: str = "vocab.pkl"):
    """
    Load vocabularies and return a dict of vocab sizes per aspect.
    Expects structure: vocabs[aspect]["id2code"].
    """
    with open(processed_root / "train" / vocab_file, "rb") as f:
        vocabs = pickle.load(f)
    V = {a: len(vocabs[a]["id2code"]) for a in ["dx", "proc", "treat"]}
    return vocabs, V


def load_split_arrays(split_dir: Path, files: dict):
    """
    Load arrays for a split given a mapping from keys to relative file paths.
    Supports .pkl and .npy.
    """
    out = {}
    for k, rel in files.items():
        p = split_dir / rel
        if p.suffix.endswith("pkl"):
            with open(p, "rb") as f:
                out[k] = pickle.load(f)
        elif p.suffix == ".npy":
            out[k] = np.load(p)
        else:
            raise ValueError(f"Unsupported file type: {p}")
    return out


def count_global_freq(patients_visits, V: int):
    """
    Compute global code frequency over all patients and visits.

    patients_visits: list of patients,
        where each patient is a list of visits,
        and each visit is a list of code indices.
    V: vocabulary size for this aspect.
    """
    cnt = np.zeros(V, dtype=np.int64)
    for visits in patients_visits:
        for codes in visits:
            for j in codes:
                if 0 <= j < V:
                    cnt[j] += 1
    return cnt


def build_patient_features_for_split(
    split_dir: Path,
    vocabs,
    V,
    top_idx_dx: dict,
    top_idx_proc: dict,
    top_idx_treat: dict,
    binary: bool,
    include_demo: bool,
):
    """
    Build patient-level feature vectors for one split (train/valid/test).

    Features consist of:
      - dx: selected diagnosis codes (binary or count)
      - proc: selected procedure codes (binary or count)
      - treat: selected treatment codes (binary or count)
      - demo (optional): demographic features from demo.npy (aligned by patient)
    """
    files = {
        "dx": "dx.seqs.pkl",
        "proc": "proc.seqs.pkl",
        "treat": "treat.seqs.pkl",
        "demo": "demo.npy",
    }
    arrs = load_split_arrays(split_dir, files)

    dx_patients = list(iter_patients_from_seqs(arrs["dx"]))
    proc_patients = list(iter_patients_from_seqs(arrs["proc"]))
    treat_patients = list(iter_patients_from_seqs(arrs["treat"]))

    assert len(dx_patients) == len(proc_patients) == len(treat_patients), \
        f"Patient counts mismatch: dx={len(dx_patients)} proc={len(proc_patients)} treat={len(treat_patients)}"

    # demo.npy: each row is a fixed-length feature vector for one patient
    has_demo = isinstance(arrs.get("demo", None), np.ndarray) and arrs["demo"].ndim == 2
    D_demo = arrs["demo"].shape[1] if has_demo and include_demo else 0

    # total feature dimensionality
    D = (
        len(top_idx_dx)
        + len(top_idx_proc)
        + len(top_idx_treat)
        + (D_demo if include_demo and has_demo else 0)
    )

    X, meta = [], []
    for i in range(len(dx_patients)):
        dx_vis = dx_patients[i]
        proc_vis = proc_patients[i]
        treat_vis = treat_patients[i]
        n_visits = min(len(dx_vis), len(proc_vis), len(treat_vis))

        fv = np.zeros(D, dtype=np.float32)
        off = 0

        # Diagnosis features
        if binary:
            mark = np.zeros(len(top_idx_dx), dtype=np.float32)
            for t in range(n_visits):
                for j in dx_vis[t]:
                    if j in top_idx_dx:
                        mark[top_idx_dx[j]] = 1.0
            fv[off:off + len(mark)] = mark
            off += len(mark)
        else:
            cnt = np.zeros(len(top_idx_dx), dtype=np.float32)
            for t in range(n_visits):
                for j in dx_vis[t]:
                    if j in top_idx_dx:
                        cnt[top_idx_dx[j]] += 1.0
            fv[off:off + len(cnt)] = cnt
            off += len(cnt)

        # Procedure features
        if binary:
            mark = np.zeros(len(top_idx_proc), dtype=np.float32)
            for t in range(n_visits):
                for j in proc_vis[t]:
                    if j in top_idx_proc:
                        mark[top_idx_proc[j]] = 1.0
            fv[off:off + len(mark)] = mark
            off += len(mark)
        else:
            cnt = np.zeros(len(top_idx_proc), dtype=np.float32)
            for t in range(n_visits):
                for j in proc_vis[t]:
                    if j in top_idx_proc:
                        cnt[top_idx_proc[j]] += 1.0
            fv[off:off + len(cnt)] = cnt
            off += len(cnt)

        # Treatment features
        if binary:
            mark = np.zeros(len(top_idx_treat), dtype=np.float32)
            for t in range(n_visits):
                for j in treat_vis[t]:
                    if j in top_idx_treat:
                        mark[top_idx_treat[j]] = 1.0
            fv[off:off + len(mark)] = mark
            off += len(mark)
        else:
            cnt = np.zeros(len(top_idx_treat), dtype=np.float32)
            for t in range(n_visits):
                for j in treat_vis[t]:
                    if j in top_idx_treat:
                        cnt[top_idx_treat[j]] += 1.0
            fv[off:off + len(cnt)] = cnt
            off += len(cnt)

        # Optional demographic features (aligned per patient)
        if include_demo and has_demo:
            demo_vec = arrs["demo"][i].astype(np.float32)
            fv[off:off + D_demo] = demo_vec[:D_demo]
            off += D_demo

        X.append(fv)
        meta.append({
            "split": split_dir.name,   # still recorded for reference
            "n_visits": int(n_visits),
            "idx": i
        })

    X = np.stack(X, axis=0) if X else np.zeros((0, D), dtype=np.float32)
    return X, meta


def main():
    ap = argparse.ArgumentParser(
        description="PCA on cohort to visually inspect latent clusters"
    )
    ap.add_argument("--processed_root", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--topk_dx", type=int, default=1000)
    ap.add_argument("--topk_proc", type=int, default=500)
    ap.add_argument("--topk_treat", type=int, default=500)
    ap.add_argument("--binary", type=int, default=1,
                    help="Use binary indicators (1) or counts (0) for code features.")
    ap.add_argument("--include_demo", type=int, default=1,
                    help="Include demographic features from demo.npy if available.")
    ap.add_argument("--n_clusters", type=int, default=0,
                    help="Number of KMeans clusters on PCA space (0 = no clustering).")
    ap.add_argument("--out_png", type=str, default="out/pca_cohort.png",
                    help="Output PNG for the multi-panel PCA plot.")
    ap.add_argument("--out_csv", type=str, default="out/pca_cohort_coords.csv")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    os.makedirs(Path(args.out_png).parent, exist_ok=True)

    vocabs, V = load_vocab(processed_root)

    # 1) Collect train patients and compute global frequency to pick indices per aspect
    def load_patients_for_aspect(split_dir: Path, aspect: str):
        seq_path = split_dir / f"{aspect}.seqs.pkl"
        with open(seq_path, "rb") as f:
            seqs = pickle.load(f)
        return list(iter_patients_from_seqs(seqs))

    train_dir = processed_root / "train"
    dx_train = load_patients_for_aspect(train_dir, "dx")
    proc_train = load_patients_for_aspect(train_dir, "proc")
    treat_train = load_patients_for_aspect(train_dir, "treat")

    # Global frequency per aspect
    dx_freq = count_global_freq(dx_train, V["dx"])
    proc_freq = count_global_freq(proc_train, V["proc"])
    treat_freq = count_global_freq(treat_train, V["treat"])

    # Build original-id -> compressed_index maps
    # Here K <= 0 or K >= vocab size means: use the whole vocabulary
    def topk_index_map(freq: np.ndarray, K: int):
        if K <= 0 or K >= len(freq):
            idx = np.arange(len(freq))
        else:
            idx = np.argsort(-freq)[:K]
        return {int(j): i for i, j in enumerate(idx.tolist())}

    top_idx_dx = topk_index_map(dx_freq, args.topk_dx)
    top_idx_proc = topk_index_map(proc_freq, args.topk_proc)
    top_idx_treat = topk_index_map(treat_freq, args.topk_treat)

    # 2) Build patient-level features for each split and concatenate
    X_list, meta_list = [], []
    for split in ["train", "valid", "test"]:
        split_dir = processed_root / split
        Xs, meta = build_patient_features_for_split(
            split_dir,
            vocabs,
            V,
            top_idx_dx,
            top_idx_proc,
            top_idx_treat,
            binary=bool(args.binary),
            include_demo=bool(args.include_demo),
        )
        X_list.append(Xs)
        meta_list.extend(meta)
        print(f"[{split}] patients={Xs.shape[0]} dim={Xs.shape[1]}")

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, 0), dtype=np.float32)

    # 3) Standardize features and apply PCA to 3 components
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X) if X.shape[0] > 0 else X

    # We want the first 3 principal components so that we can plot:
    # (PC1 vs PC2), (PC1 vs PC3), and (PC2 vs PC3).
    pca = PCA(n_components=3, random_state=707)
    if Xz.shape[0] > 0 and Xz.shape[1] >= 3:
        Z = pca.fit_transform(Xz)          # shape: (n_samples, 3)
        evr = pca.explained_variance_ratio_
    else:
        # Fallback if dimensionality is too small
        Z = np.zeros((Xz.shape[0], 3), dtype=np.float32)
        evr = [0.0, 0.0, 0.0]

    print(
        "PCA explained variance ratio: "
        f"PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f}"
    )

    # 4) Optional KMeans clustering in 3D PCA space (for discovering latent clusters)
    if args.n_clusters and args.n_clusters > 0 and Z.shape[0] >= args.n_clusters:
        km = KMeans(n_clusters=args.n_clusters, random_state=707, n_init="auto")
        labels = km.fit_predict(Z)
    else:
        labels = np.array([-1] * Z.shape[0], dtype=int)

    # 5) Save CSV (with 3 PCs) and plot three pairwise views:
    #    (PC1 vs PC2), (PC1 vs PC3), (PC2 vs PC3) in one multi-panel figure.
    import pandas as pd

    df = pd.DataFrame({
        "pc1": Z[:, 0] if Z.shape[0] else [],
        "pc2": Z[:, 1] if Z.shape[0] else [],
        "pc3": Z[:, 2] if Z.shape[0] else [],
        "split": [m["split"] for m in meta_list],
        "n_visits": [m["n_visits"] for m in meta_list],
        "cluster": labels.tolist(),
    })
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # --- Plotting: 3 subplots for PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Which PC pairs to plot and their labels
    pc_pairs = [
        (0, 1, "PC1", "PC2"),
        (0, 2, "PC1", "PC3"),
        (1, 2, "PC2", "PC3"),
    ]

    # Whether we actually have clustering info
    have_clusters = (labels >= 0).any()

    for ax, (i_pc, j_pc, xlab, ylab) in zip(axes, pc_pairs):
        if have_clusters:
            # Color each cluster differently
            uniq_clusters = sorted(c for c in np.unique(labels) if c >= 0)
            for k in uniq_clusters:
                m = (labels == k)
                if m.sum() == 0:
                    continue
                ax.scatter(
                    Z[m, i_pc],
                    Z[m, j_pc],
                    s=np.clip(df.loc[m, "n_visits"].values, 5, 30),
                    c=f"C{k % 10}",
                    alpha=0.5,
                    label=f"cluster {k} (n={m.sum()})" if (i_pc, j_pc) == (0, 1) else None,
                )

            # Overlay cluster centers for this pair of PCs
            for k in uniq_clusters:
                m = (labels == k)
                if m.sum() == 0:
                    continue
                cx = Z[m, i_pc].mean()
                cy = Z[m, j_pc].mean()
                ax.scatter(
                    [cx],
                    [cy],
                    marker="X",
                    s=140,
                    c="k",
                    edgecolor="white",
                    linewidths=1.0,
                )
        else:
            # No clustering: plot all patients as one group
            ax.scatter(
                Z[:, i_pc],
                Z[:, j_pc],
                s=np.clip(df["n_visits"].values, 5, 30),
                alpha=0.5,
            )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(False)

    # Only show legend in the first subplot to avoid clutter
    if have_clusters:
        axes[0].legend(frameon=False)

    fig.suptitle(
        "PCA on Cohort "
        f"(EVR: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}, PC3={evr[2]:.2%})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180, bbox_inches="tight")
    print("Saved:", args.out_png, "and", args.out_csv)


if __name__ == "__main__":
    main()