#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_discharge_detail_with_hadm(mimic_dir: Path) -> pd.DataFrame:
    # discharge_detail may lack hadm_id; join via discharge.csv.gz on note_id if needed
    det = pd.read_csv(mimic_dir / "note" / "discharge_detail.csv.gz", low_memory=False)
    det.columns = [c.lower() for c in det.columns]
    if "hadm_id" not in det.columns:
        if "note_id" not in det.columns:
            raise ValueError("discharge_detail: missing both hadm_id and note_id.")
        base = pd.read_csv(
            mimic_dir / "note" / "discharge.csv.gz",
            usecols=["note_id", "subject_id", "hadm_id"],
        )
        base.columns = [c.lower() for c in base.columns]
        det = det.merge(base, on="note_id", how="left", validate="many_to_one")
    # keep non-empty text column (field_value or text)
    text_col = "field_value" if "field_value" in det.columns else (
        "text" if "text" in det.columns else None
    )
    if text_col is None:
        raise ValueError("discharge_detail: cannot find text column (field_value/text).")
    det[text_col] = det[text_col].astype(str).str.strip()
    det = det[
        (det["hadm_id"].notna())
        & (det[text_col].str.len() > 0)
        & (det[text_col].str.lower() != "nan")
    ]
    return det[["hadm_id"]].drop_duplicates()


def hadm_with_noteevents(mimic_dir: Path, chunksize: int = 500_000) -> pd.DataFrame:
    # Collect hadm_ids that have at least one non-empty note text from noteevents
    path = mimic_dir / "note" / "noteevents.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    cols = ["hadm_id", "text"]
    acc = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
        c = chunk.dropna(subset=["hadm_id"]).copy()
        c["text"] = c["text"].astype(str).str.strip()
        c = c[c["text"].str.len() > 0]
        if not c.empty:
            acc.append(c[["hadm_id"]].drop_duplicates())
    if acc:
        return pd.concat(acc, axis=0).drop_duplicates()
    return pd.DataFrame(columns=["hadm_id"])


def compute_notes_coverage(
    mimic_dir: Path,
    source: str,
    cohort_subjects: Optional[str] = None,
) -> dict:
    hosp = mimic_dir / "hosp"
    # read both hadm_id and subject_id so we can filter by either
    adm = pd.read_csv(hosp / "admissions.csv.gz", usecols=["hadm_id", "subject_id"])

    # If a cohort CSV is provided, restrict admissions to that cohort
    if cohort_subjects is not None:
        coh = pd.read_csv(cohort_subjects)
        if "subject_id" in coh.columns:
            subj_set = set(coh["subject_id"].unique())
            adm = adm[adm["subject_id"].isin(subj_set)]
            cohort_type = "subject_id"
        elif "hadm_id" in coh.columns:
            hadm_set = set(coh["hadm_id"].unique())
            adm = adm[adm["hadm_id"].isin(hadm_set)]
            cohort_type = "hadm_id"
        else:
            raise ValueError(
                "cohort_subjects CSV must contain either 'subject_id' or 'hadm_id' column."
            )
    else:
        cohort_type = None

    # admissions within scope (overall or cohort)
    hadm_in_scope = adm["hadm_id"].dropna().unique()
    total_visits = int(len(hadm_in_scope))

    if source == "discharge_detail":
        notes = load_discharge_detail_with_hadm(mimic_dir)
    elif source == "noteevents":
        notes = hadm_with_noteevents(mimic_dir)
    else:
        raise ValueError("notes_source must be 'discharge_detail' or 'noteevents'.")

    # restrict notes to the same scope
    notes = notes[notes["hadm_id"].isin(hadm_in_scope)]
    with_note = int(notes["hadm_id"].nunique())
    without_note = total_visits - with_note

    return {
        "total_visits": total_visits,
        "visits_with_note": with_note,
        "visits_without_note": without_note,
        "fraction_without_note": float(without_note / max(1, total_visits)),
        "notes_source": source,
        "cohort_subjects": cohort_subjects,
        "cohort_type": cohort_type,
    }


def load_routed_notes(processed_root: Path) -> pd.DataFrame:
    # Concatenate notes.jsonl from train/valid/test; drop patient sentinels {"_": -1}
    acc = []
    for split in ["train", "valid", "test"]:
        path = processed_root / split / "notes.jsonl"
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        if "_" in df.columns:
            df = df[df["_"].isna()]
        acc.append(df)
    if not acc:
        return pd.DataFrame()
    df = pd.concat(acc, axis=0, ignore_index=True)
    for c in ["dx", "proc", "treat", "ap"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)
    return df


def aspect_missingness(df_notes: pd.DataFrame) -> pd.DataFrame:
    if df_notes.empty:
        return pd.DataFrame(
            {
                "aspect": ["dx", "proc", "treat"],
                "missing_ratio": [np.nan, np.nan, np.nan],
            }
        )
    res = []
    denom = len(df_notes)
    for a in ["dx", "proc", "treat"]:
        miss = (df_notes[a].str.strip().str.len() == 0).sum()
        res.append(
            {
                "aspect": a,
                "missing": int(miss),
                "total": int(denom),
                "missing_ratio": float(miss / max(1, denom)),
            }
        )
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser(
        description="EDA: notes coverage and aspect-level missingness"
    )
    ap.add_argument(
        "--mimic_dir",
        required=True,
        type=str,
        help="Root of MIMIC-IV (with hosp/, note/)",
    )
    ap.add_argument(
        "--processed_root",
        required=True,
        type=str,
        help="data/processed/mimic4_splits",
    )
    ap.add_argument(
        "--notes_source",
        choices=["discharge_detail", "noteevents"],
        default="discharge_detail",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="out/eda",
    )
    ap.add_argument(
        "--cohort_subjects",
        type=str,
        default=None,
        help="Optional CSV with cohort definition; must contain subject_id or hadm_id.",
    )
    args = ap.parse_args()

    mimic_dir = Path(args.mimic_dir)
    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part A: notes coverage on all MIMIC-IV or restricted cohort
    cov = compute_notes_coverage(mimic_dir, args.notes_source, args.cohort_subjects)
    fig, ax = plt.subplots(figsize=(5, 3))
    vals = [cov["visits_with_note"], cov["visits_without_note"]]
    labels = ["With Notes", "No Matched Notes"]
    ax.bar(labels, vals)
    title_prefix = "Notes Missingness"
    if cov["cohort_subjects"] is not None:
        title_prefix += " (Cohort)"
    else:
        title_prefix += " (Overall)"
    ax.set_title(
        f"{title_prefix}\nNo Matched Note Fraction: {cov['fraction_without_note']:.1%}"
    )
    ax.set_ylabel("Number of Visits")
    fig.tight_layout()
    fig.savefig(out_dir / "notes_coverage.png", dpi=180)

    # Part B: aspect-level missingness after routing (dx/proc/treat)
    routed = load_routed_notes(processed_root)
    miss_df = aspect_missingness(routed)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.bar(miss_df["aspect"], miss_df["missing_ratio"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Missing Ratio")
    ax2.set_title("Aspect-Level Notes Missingness")
    fig2.tight_layout()
    fig2.savefig(out_dir / "aspect_missingness.png", dpi=180)

    # Write JSON summary
    summary = {
        "coverage": cov,
        "aspect_missingness": miss_df.to_dict(orient="records"),
    }
    pd.Series(summary).to_json(out_dir / "notes_eda_summary.json", indent=2)
    print(
        "[notes] Saved notes_coverage.png, aspect_missingness.png, and "
        "notes_eda_summary.json to",
        out_dir,
    )


if __name__ == "__main__":
    main()