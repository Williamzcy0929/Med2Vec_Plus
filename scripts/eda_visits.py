#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def visits_per_subject(adm: pd.DataFrame) -> pd.Series:
    # Count unique admissions (visits) per subject
    return adm.groupby("subject_id")["hadm_id"].nunique()

def main():
    ap = argparse.ArgumentParser(description="EDA: number of visits distribution (MIMIC vs cohort)")
    ap.add_argument("--mimic_dir", required=True, type=str, help="Root of MIMIC-IV (with hosp/)")
    ap.add_argument("--cohort_dir", type=str, default="out/cohort_discharge_only", help="Directory containing cohort_subjects.csv")
    ap.add_argument("--out_dir", type=str, default="out/eda", help="Where to write figures and summaries")
    args = ap.parse_args()

    hosp = Path(args.mimic_dir) / "hosp"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adm = pd.read_csv(hosp / "admissions.csv.gz", usecols=["subject_id", "hadm_id"])
    v_all = visits_per_subject(adm)

    # Cohort (optional): read subject list and subset
    cohort_path = Path(args.cohort_dir) / "cohort_subjects.csv"
    if cohort_path.exists():
        cohort = pd.read_csv(cohort_path, usecols=["subject_id"])
        cohort_ids = set(cohort["subject_id"].tolist())
        v_cohort = v_all[v_all.index.isin(cohort_ids)]
    else:
        v_cohort = pd.Series([], dtype=int)

    # Compute ≥2 visits proportion on the full dataset
    prop_ge2 = (v_all >= 2).mean()

    # Build clipped histograms (1..9 and "10+")
    def to_bins(s: pd.Series):
        x = np.clip(s.to_numpy(), 1, 10)
        bins = np.arange(1, 12)  # 1..11, with 10 representing 10+
        hist, _ = np.histogram(x, bins=bins)
        labels = [str(i) for i in range(1, 10)] + ["10+"]
        return labels, hist

    labels_all, hist_all = to_bins(v_all)
    labels_c, hist_c = (labels_all, np.zeros_like(hist_all)) if v_cohort.empty else to_bins(v_cohort)

    # Plot side-by-side bars
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels_all))
    w = 0.4
    ax.bar(x - w/2, hist_all, width=w, label="All MIMIC-IV")
    if not v_cohort.empty:
        ax.bar(x + w/2, hist_c, width=w, label="Cohort")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_all, rotation=0)
    ax.set_xlabel("Visits per Patient")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Visit Distribution (≥2 Visits in All MIMIC-IV: {prop_ge2:.1%})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "visits_distribution.png", dpi=180)

    # Write a short summary
    summary = {
        "patients_total": int(v_all.shape[0]),
        "patients_ge2": int((v_all >= 2).sum()),
        "prop_ge2": float(prop_ge2),
        "cohort_subjects": (int(v_cohort.shape[0]) if not v_cohort.empty else 0)
    }
    pd.Series(summary).to_json(out_dir / "visits_summary.json", indent=2)
    print("[visits] Saved visits_distribution.png and visits_summary.json to", out_dir)

if __name__ == "__main__":
    main()