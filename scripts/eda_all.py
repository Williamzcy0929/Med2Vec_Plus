#!/usr/bin/env python
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run all EDA figures (visits + notes)")
    ap.add_argument("--mimic_dir", required=True, type=str)
    ap.add_argument("--processed_root", required=True, type=str)
    ap.add_argument("--cohort_dir", type=str, default="out/cohort_discharge_only")
    ap.add_argument("--notes_source", choices=["discharge_detail","noteevents"], default="discharge_detail")
    ap.add_argument("--out_dir", type=str, default="out/eda")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    eda_vis = root / "eda_visits.py"
    eda_notes = root / "eda_missing_notes.py"

    # Run visits EDA
    cmd1 = [
        sys.executable, str(eda_vis),
        "--mimic_dir", args.mimic_dir,
        "--cohort_dir", args.cohort_dir,
        "--out_dir", args.out_dir,
    ]
    print("+", " ".join(cmd1)); subprocess.check_call(cmd1)

    # Run notes EDA
    cmd2 = [
        sys.executable, str(eda_notes),
        "--mimic_dir", args.mimic_dir,
        "--processed_root", args.processed_root,
        "--notes_source", args.notes_source,
        "--out_dir", args.out_dir,
    ]
    print("+", " ".join(cmd2)); subprocess.check_call(cmd2)
    print("[eda] All done. See outputs in", args.out_dir)

if __name__ == "__main__":
    main()