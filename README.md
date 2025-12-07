# med2vec_plus (full pipeline)

This repository contains:
- Cohort selection tool for MIMIC-IV (CSV and Postgres engines).
- Preprocessing from MIMIC-IV to Med2Vec-compatible inputs.
- Note-Aware Multi-Aspect Med2Vec model and a Med2Vec-compatible baseline.
- Classical ML baselines (RF/GB/MLP/XGBoost).
- CUDA/MPS device support and synthetic data for quick tests.

## Simulated data quickstart (no external data required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Med2Vec baseline on synthetic data
python scripts/test_sim_med2vec_baseline.py

# Note-Aware Multi-Aspect Med2Vec on synthetic data
python scripts/test_sim_plus_model.py

# Classical ML baselines (RF/GB/MLP/XGBoost) on synthetic data
python scripts/test_sim_ml_baselines.py

# Run all three
python scripts/test_sim_all.py
```

## End-to-end on MIMIC-IV (CSV engine)

```bash
python scripts/select_cohort.py   --mimic_dir /data/mimic-iv-2.2   --notes_file /data/mimic-iv-2.2/note/noteevents.csv.gz   --min_visits 3 --require_dx --require_proc --require_treat --require_note   --sample_n 10000 --seed 707   --engine csv   --out_dir ./out/cohort_v1

python scripts/prepare_mimic_iv.py   --mimic_dir /data/mimic-iv-2.2   --notes_file /data/mimic-iv-2.2/note/noteevents.csv.gz   --cohort_subjects ./out/cohort_v1/cohort_subjects.csv   --out_dir data/processed/mimic4_splits   --use_train_vocab_only

# Med2Vec baseline
python -m src.med2vec_plus.train --config configs/med2vec_only.yaml
python -m src.med2vec_plus.evaluate --config configs/med2vec_only.yaml --ckpt ckpts_med2vec_only/best.pt --split test

# Note-Aware Multi-Aspect
python -m src.med2vec_plus.train --config configs/default.yaml
python -m src.med2vec_plus.evaluate --config configs/default.yaml --ckpt ckpts/best.pt --split test

# Classical ML baselines
python -m src.med2vec_plus.evaluate --config configs/med2vec_only.yaml --ckpt ckpts_med2vec_only/best.pt --split test --run_sklearn_baselines
```
