#!/usr/bin/env python
import argparse, subprocess, sys, os, pathlib

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description="Run Note-Aware Multi-Aspect Med2Vec end-to-end without editing configs")
    ap.add_argument("--mimic_dir", type=str, required=True)
    ap.add_argument("--notes_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--cohort_dir", type=str, default="out/cohort_v1")
    args = ap.parse_args()

    notes = args.notes_file or os.path.join(args.mimic_dir, "note", "noteevents.csv.gz")
    pathlib.Path(args.cohort_dir).mkdir(parents=True, exist_ok=True)

    sh(f"python scripts/select_cohort.py --mimic_dir {args.mimic_dir} --notes_file {notes} --out_dir {args.cohort_dir}")
    sh(f"python scripts/prepare_mimic_iv.py --mimic_dir {args.mimic_dir} --notes_file {notes} --cohort_subjects {args.cohort_dir}/cohort_subjects.csv --out_dir {args.out_dir} --use_train_vocab_only")
    sh("python -m src.med2vec_plus.train --config configs/default.yaml")
    sh("python -m src.med2vec_plus.evaluate --config configs/default.yaml --ckpt ckpts/best.pt --split test")

if __name__ == '__main__':
    main()
