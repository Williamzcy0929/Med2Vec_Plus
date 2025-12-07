#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build note sections per-visit (dx/proc/treat/ap) from *_detail tables and
overwrite data/processed/mimic4_splits/*/notes.jsonl accordingly.

Design goals:
- Use discharge_detail and (optionally) radiology_detail section-level rows.
- Route sections into {dx, proc, treat, ap} for Med2Vec+ cross-attention.
- Reproduce the same subject-level split policy as prepare_mimic_iv.py:
  subjects shuffled with seed 17, then 70/10/20 to train/valid/test.
- If --cohort_subjects is provided, restrict the subject pool to this cohort.
- Keep a .bak backup of existing notes.jsonl before overwriting.

This script does NOT change model code or other files.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd


# ---------------------------
# Section routing heuristics
# ---------------------------

DISCHARGE_TO_DX = {
    "discharge diagnosis", "discharge diagnoses", "diagnosis", "diagnoses",
    "final diagnosis", "principal diagnosis", "secondary diagnoses"
}
DISCHARGE_TO_PROC = {
    "procedure", "procedures", "operations", "operations performed",
    "operative report", "surgery", "surgeries"
}
DISCHARGE_TO_TREAT = {
    "medications", "medications on admission", "medications on discharge",
    "home medications", "current medications", "discharge medications",
    "medication on discharge", "meds", "discharge meds"
}
DISCHARGE_TO_AP = {
    "assessment", "plan", "assessment & plan", "assessment and plan",
    "hospital course", "course", "summary", "reason for admission",
    "condition on discharge", "disposition", "follow up"
}

# Radiology: route "impression" -> dx; "findings" -> ap (or proc if user chooses)
RAD_TO_DX = {"impression", "conclusion"}
RAD_FINDINGS = {"findings", "result"}

# Accept multiple common column names from *_detail tables
CAND_SECTION_COLS = ["section", "section_name", "section_header", "category", "label", "title", "SECTION", "Section"]
CAND_TEXT_COLS = ["text", "section_text", "note_text", "TEXT"]


def choose_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Pick the first existing column name among candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def normalize(s: str) -> str:
    return (s or "").strip()


def route_discharge(section_name: str) -> str:
    """Map a discharge section header to dx/proc/treat/ap."""
    name = (section_name or "").strip().lower()
    # exact or contains matching
    if any(k in name for k in DISCHARGE_TO_DX):
        return "dx"
    if any(k in name for k in DISCHARGE_TO_PROC):
        return "proc"
    if any(k in name for k in DISCHARGE_TO_TREAT):
        return "treat"
    if any(k in name for k in DISCHARGE_TO_AP):
        return "ap"
    # default to ap
    return "ap"


def route_radiology(section_name: str, findings_to: str = "ap") -> str:
    """Map a radiology section header to dx/ap (or proc if configured)."""
    name = (section_name or "").strip().lower()
    if any(k in name for k in RAD_TO_DX):
        return "dx"
    if any(k in name for k in RAD_FINDINGS):
        return findings_to  # "ap" (default) or "proc"
    # any other radiology subheader -> ap
    return "ap"


def load_detail_table(path: Path) -> pd.DataFrame:
    """Robustly read a *_detail.csv(.gz) into a DataFrame with canonical columns."""
    if not path.exists():
        raise FileNotFoundError(f"Detail file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    # detect required columns
    sec_col = choose_col(df, CAND_SECTION_COLS)
    txt_col = choose_col(df, CAND_TEXT_COLS)
    for req, name in [("subject_id", "subject_id"), ("hadm_id", "hadm_id")]:
        if req not in df.columns:
            raise ValueError(f"Required column missing in {path}: {name}")
    if sec_col is None or txt_col is None:
        raise ValueError(
            f"Cannot detect section/text columns in {path}. "
            f"Got columns: {list(df.columns)[:10]} ..."
        )
    out = df.loc[:, ["subject_id", "hadm_id", sec_col, txt_col]].copy()
    out.rename(columns={sec_col: "section", txt_col: "text"}, inplace=True)
    # drop whitespace-only text
    out["text"] = out["text"].astype(str)
    out = out[out["text"].str.strip().str.len() > 0]
    return out


def aggregate_sections(
    discharge_detail: Optional[Path],
    radiology_detail: Optional[Path],
    radiology_findings_to: str = "ap",
    max_chars_per_field: Optional[int] = None,
) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Build { (subject_id, hadm_id): {dx: ..., proc: ..., treat: ..., ap: ...} }.
    Concatenate multiple section chunks with separators.
    """
    by_visit: Dict[Tuple[int, int], Dict[str, List[str]]] = defaultdict(
        lambda: {"dx": [], "proc": [], "treat": [], "ap": []}
    )

    def add_chunk(sid: int, hadm: int, slot: str, header: str, text: str, source: str):
        prefix = f"[{source.upper()} | {header}]\n"
        by_visit[(sid, hadm)][slot].append(prefix + text.strip())

    # Discharge detail
    if discharge_detail is not None:
        df = load_detail_table(discharge_detail)
        for _, row in df.iterrows():
            sid = int(row["subject_id"])
            hadm = int(row["hadm_id"])
            sec = str(row["section"])
            slot = route_discharge(sec)
            add_chunk(sid, hadm, slot, sec, str(row["text"]), "discharge")

    # Radiology detail
    if radiology_detail is not None:
        df = load_detail_table(radiology_detail)
        for _, row in df.iterrows():
            sid = int(row["subject_id"])
            hadm = int(row["hadm_id"])
            sec = str(row["section"])
            slot = route_radiology(sec, findings_to=radiology_findings_to)
            add_chunk(sid, hadm, slot, sec, str(row["text"]), "radiology")

    # Collapse lists to single strings
    result: Dict[Tuple[int, int], Dict[str, str]] = {}
    for key, slots in by_visit.items():
        merged = {}
        for k, lst in slots.items():
            text = "\n\n----\n\n".join(lst)
            if max_chars_per_field is not None and len(text) > max_chars_per_field:
                text = text[:max_chars_per_field]
            merged[k] = text
        result[key] = merged
    return result


def build_subject_splits(
    mimic_dir: Path,
    cohort_subjects_csv: Optional[Path],
    seed: int = 17,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    """
    Reproduce the subject splits used by prepare_mimic_iv.py:
    - shuffle subject_ids with fixed seed
    - 70%/10%/20% for train/valid/test
    - optionally restrict to cohort_subjects
    """
    hosp = mimic_dir / "hosp"
    adm = pd.read_csv(hosp / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime"])
    if cohort_subjects_csv is not None:
        subs = pd.read_csv(cohort_subjects_csv, usecols=["subject_id"])
        sid_pool = list(sorted(set(subs["subject_id"].tolist())))
        adm = adm[adm["subject_id"].isin(sid_pool)]
    sujs = adm["subject_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sujs)
    n = len(sujs)
    n_tr = int(train_ratio * n)
    n_va = int(valid_ratio * n)
    train_ids = sujs[:n_tr]
    valid_ids = sujs[n_tr : n_tr + n_va]
    test_ids = sujs[n_tr + n_va :]
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def write_notes_jsonl_for_split(
    mimic_dir: Path,
    split_subjects: List[int],
    visit_texts: Dict[Tuple[int, int], Dict[str, str]],
    out_jsonl: Path,
):
    """
    For each subject in split:
      - sort their admissions by admittime,
      - write one JSON per visit: {"dx":..., "proc":..., "treat":..., "ap":..., "subject_id":..., "hadm_id":...}
      - after the last visit of the subject, write a sentinel line: {"_": -1}
    """
    hosp = mimic_dir / "hosp"
    adm = pd.read_csv(
        hosp / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
    )
    adm = adm[adm["subject_id"].isin(split_subjects)]
    adm = adm.sort_values(["subject_id", "admittime"])

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Keep a .bak backup if a notes.jsonl exists
    if out_jsonl.exists():
        out_jsonl.rename(out_jsonl.with_suffix(out_jsonl.suffix + ".bak"))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sid, g in adm.groupby("subject_id"):
            g = g.sort_values("admittime")
            for _, row in g.iterrows():
                key = (int(row["subject_id"]), int(row["hadm_id"]))
                slots = visit_texts.get(
                    key,
                    {"dx": "", "proc": "", "treat": "", "ap": ""},  # empty if no detail found
                )
                rec = {
                    "dx": normalize(slots.get("dx", "")),
                    "proc": normalize(slots.get("proc", "")),
                    "treat": normalize(slots.get("treat", "")),
                    "ap": normalize(slots.get("ap", "")),
                    # Extra keys are safe: our Dataset ignores them
                    "subject_id": int(row["subject_id"]),
                    "hadm_id": int(row["hadm_id"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # patient separator sentinel (required by the current Dataset)
            f.write(json.dumps({"_": -1}) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build routed notes.jsonl from *_detail tables")
    ap.add_argument("--mimic_dir", type=str, required=True, help="Root of MIMIC-IV (with hosp/, note/)")
    ap.add_argument("--discharge_detail", type=str, required=True, help="Path to note/discharge_detail.csv.gz")
    ap.add_argument("--radiology_detail", type=str, default=None, help="Optional path to note/radiology_detail.csv.gz")
    ap.add_argument("--radiology_findings_to", type=str, choices=["ap", "proc"], default="ap",
                    help="Route radiology 'Findings' to 'ap' (default) or 'proc'")
    ap.add_argument("--processed_root", type=str, default="data/processed/mimic4_splits",
                    help="Where train/valid/test folders live")
    ap.add_argument("--cohort_subjects", type=str, default=None,
                    help="Optional CSV produced by select_cohort.py (restricts subjects)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max_chars_per_field", type=int, default=None,
                    help="Optional char cap per field (dx/proc/treat/ap) to control size")
    args = ap.parse_args()

    mimic_dir = Path(args.mimic_dir)
    processed_root = Path(args.processed_root)
    discharge_detail = Path(args.discharge_detail) if args.discharge_detail else None
    radiology_detail = Path(args.radiology_detail) if args.radiology_detail else None
    cohort_csv = Path(args.cohort_subjects) if args.cohort_subjects else None

    # 1) Aggregate section-level rows into per-visit routed text blobs
    print("Aggregating section texts from detail tables ...")
    visit_texts = aggregate_sections(
        discharge_detail=discharge_detail,
        radiology_detail=radiology_detail,
        radiology_findings_to=args.radiology_findings_to,
        max_chars_per_field=args.max_chars_per_field,
    )
    print(f"  Built routed text for {len(visit_texts)} (subject_id, hadm_id) visits.")

    # 2) Rebuild subject splits (same policy as prepare_mimic_iv.py)
    print("Building subject splits (seed={}, 70/10/20) ...".format(args.seed))
    splits = build_subject_splits(
        mimic_dir=mimic_dir,
        cohort_subjects_csv=cohort_csv,
        seed=args.seed,
        train_ratio=0.7,
        valid_ratio=0.1,
    )

    # 3) Overwrite notes.jsonl per split
    for split in ["train", "valid", "test"]:
        out_jsonl = processed_root / split / "notes.jsonl"
        print(f"Writing {out_jsonl} ...")
        write_notes_jsonl_for_split(
            mimic_dir=mimic_dir,
            split_subjects=splits[split],
            visit_texts=visit_texts,
            out_jsonl=out_jsonl,
        )
    print("Done. New routed notes.jsonl files are ready.")
    print("If needed, original notes.jsonl files are saved as *.bak next to the new ones.")


if __name__ == "__main__":
    main()