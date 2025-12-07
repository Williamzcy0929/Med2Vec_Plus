#!/usr/bin/env python
import argparse, subprocess, sys, os, pathlib

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description="Run classical ML baselines without editing configs")
    ap.add_argument("--mimic_dir", type=str, required=True)
    ap.add_argument("--notes_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--cohort_dir", type=str, default="out/cohort_v1")
    ap.add_argument("--ckpt", type=str, default="ckpts_med2vec_only/best.pt")
    args = ap.parse_args()

    notes = args.notes_file or os.path.join(args.mimic_dir, "note", "noteevents.csv.gz")
    pathlib.Path(args.cohort_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.out_dir):
        sh(f"python scripts/select_cohort.py --mimic_dir {args.mimic_dir} --notes_file {notes} --out_dir {args.cohort_dir}")
        sh(f"python scripts/prepare_mimic_iv.py --mimic_dir {args.mimic_dir} --notes_file {notes} --cohort_subjects {args.cohort_dir}/cohort_subjects.csv --out_dir {args.out_dir} --use_train_vocab_only")

    if not os.path.exists(args.ckpt):
        sh("python -m src.med2vec_plus.train --config configs/med2vec_only.yaml")

    sh(f"python -m src.med2vec_plus.evaluate --config configs/med2vec_only.yaml --ckpt {args.ckpt} --split test --run_sklearn_baselines")

if __name__ == '__main__':
    main()
