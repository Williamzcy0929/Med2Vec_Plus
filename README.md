# Med2Vec+

**Med2Vec+** is a representation learning model that has the capacity of **EHR codes+Clinical Notes** on **MIMIC-IV**. It reproduces Med2Vec-style representation while extending Med2Vec with clinical notes via aspect-level attention.  

Med2Vec+ and its model weights are also published on [Hugging Face](https://huggingface.co/Williamzcy0929/Med2Vec_Plus)

## What It Does

- Builds **routed clinical notes** (diagnosis/procedure/treatment/assessment-plan) from MIMIC-IV discharge summaries
- Merges with MIMIC-IV *hosp* tables to create **per-visit code sequences** and demographics
- Selects cohorts and writes **train/valid/test** splits with proper temporal ordering
- Trains **Med2Vec+** (temporal GRU or Transformer) with optional HuggingFace encoders for notes
- Supports **post-hoc calibration** (temperature scaling, Platt scaling, isotonic regression)
- Provides embedding and metric visualizations

---

## Table of Contents

- [Model at a Glance](#model-at-a-glance)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Prerequisites](#data-prerequisites)
- [Quick Start](#quick-start-end-to-end)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Evaluation & Calibration](#evaluation--calibration)
- [MIMIC-IV Variables Used](#mimic-iv-variables-used)

---

## Med2Vec+ Model

### Inputs per Patient

A sequence of visits where each visit contains:
- Three **code sets**: diagnosis (DX) / procedures (PROC) / treatments (TREAT)
- Patient **demographics** (age, gender, ethnicity, etc.)
- **Routed note segments**: diagnosis/procedure/treatment/assessment-plan text

### Architecture

- **Code encoders**: Learnable embeddings for diagnosis/procedure/treatment codes (optional shared or separate)
- **Text encoder**: BERT-family encoder (e.g., `emilyalsentzer/Bio_ClinicalBERT`) to embed routed note aspects
- **Aspect-aligned cross-attention**: Note embeddings query each code aspect so the model "reads" only relevant text for that aspect
- **Temporal module**: **GRU** or **Transformer** over visit sequences
- **Prediction heads**: Next-visit multilabel prediction for diagnosis/procedure/treatment (sigmoid over vocabularies); optional risk head

### Outputs

For each time step *t*, the model predicts probabilities over code vocabularies for the **next visit**:

- ŷ<sub>dx</sub> ∈ [0,1]<sup>V<sub>dx</sub></sup>
- ŷ<sub>proc</sub> ∈ [0,1]<sup>V<sub>proc</sub></sup>
- ŷ<sub>treat</sub> ∈ [0,1]<sup>V<sub>treat</sub></sup>

---

## Repository Layout

```
.
├── configs/
│   ├── default.yaml              # Main config for Med2Vec+
│   └── med2vec_only.yaml         # Code-only baseline config
├── scripts/
│   ├── discharge_only_pipeline.py    # Build routed notes, cohort, splits
│   ├── prepare_mimic_iv.py           # Build code sequences & vocabularies
│   └── visualize_code_embeddings.py  # Code embedding t-SNE/UMAP
└── src/med2vec_plus/
    ├── train.py                  # Model trainer
    ├── evaluate.py               # Model evaluation
    ├── calibrate.py              # Temperature/Platt/isotonic calibration
    ├── data/                     # Dataset, collate, vocabulary utilities
    ├── models/                   # Model architecture, temporal modules
    └── utils.py                  # Helper functions
```

---

## Installation

**Requirements**: Python 3.10–3.12

```bash
# Clone repository
git clone https://github.com/<your_org>/med2vec_plus_pipeline.git
cd med2vec_plus_pipeline

# Option A: Conda environment
conda create -n med2vecplus python=3.11 -y
conda activate med2vecplus

# Option B: Virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Prerequisites

You must have **approved access** to [MIMIC-IV v2.x](https://physionet.org/content/mimiciv/). Organize files as follows:

```
/path/to/mimic-iv-2.2/
├── hosp/
│   ├── admissions.csv.gz
│   ├── patients.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── procedures_icd.csv.gz
│   └── prescriptions.csv.gz
└── note/
    ├── discharge_detail.csv.gz   # Section-level discharge rows
    └── discharge.csv.gz          # Maps note_id → hadm_id, subject_id
```

### Important Notes

- We only require **discharge notes** for the routed text pipeline
- `discharge_detail.csv.gz` must have `field_name`/`field_value` columns (or equivalent)
- `discharge.csv.gz` is used to map `note_id` → `hadm_id`/`subject_id`

---

## Quick Start

### Build Routed Notes + Cohort + Splits

```bash
python scripts/discharge_only_pipeline.py \
  --mimic_dir /path/to/mimic-iv-2.2 \
  --discharge_detail /path/to/mimic-iv-2.2/note/discharge_detail.csv.gz \
  --processed_root data/processed/mimic4_splits \
  --cohort_out_dir out/cohort_discharge_only \
  --min_visits 3 --require_dx --require_proc --require_treat --require_note \
  --sample_n 10000 --seed 707 --split_seed 17 \
  --encode_with_hf \
  --hf_model emilyalsentzer/Bio_ClinicalBERT \
  --pooling cls --max_length 256 --batch_size 16
```

### Create Code Sequences & Vocabularies

```bash
python scripts/prepare_mimic_iv.py \
  --mimic_dir /path/to/mimic-iv-2.2 \
  --notes_file data/processed/mimic4_splits/train/notes.jsonl \
  --out_dir data/processed/mimic4_splits \
  --use_train_vocab_only
```

### Train Med2Vec+

```bash
python -m src.med2vec_plus.train --config configs/default.yaml
# Checkpoints saved to ckpts/
# TensorBoard logs saved to runs/med2vec_plus/
```

### Evaluate Model

```bash
python -m src.med2vec_plus.evaluate \
  --config configs/default.yaml \
  --ckpt ckpts/best.pt \
  --split test
```

### Calibrate Probabilities

```bash
# Fit calibrator on validation set
python -m src.med2vec_plus.calibrate \
  --config configs/default.yaml \
  --ckpt ckpts/best.pt \
  --split valid \
  --method temperature \
  --save ckpts_calibrated

# Apply calibration at test time
python -m src.med2vec_plus.evaluate \
  --config configs/default.yaml \
  --ckpt ckpts/best.pt \
  --split test \
  --calibration ckpts_calibrated/calibration_temperature_valid.pkl
```

### Using Offline HuggingFace Models

Place your model under `/root/hf_models/Bio_ClinicalBERT` and configure:

```yaml
# In configs/default.yaml
model:
  hf_name_or_path: /root/hf_models/Bio_ClinicalBERT
```

Or set environment variables:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/hf_models
export TRANSFORMERS_CACHE=/root/hf_models
```

---

## Configuration

See `configs/default.yaml` for complete configuration options.

### Key Configuration Sections

**Data**
- Split root directory, file names per split
- Max/min visits per patient
- Maximum note length

**Model**
- Embedding sizes, dropout rates
- Temporal module type (`gru` or `transformer`)
- Use of notes and demographics

**Loss Function**
- `lambda_sup`: Supervised next-visit loss weight
- `lambda_intra`: Intra-visit alignment loss weight
- `lambda_text`: Text-guided attention loss weight

**Optimization**
- AdamW learning rate, weight decay
- Batch size, epochs, gradient clipping
- Warmup steps

**Evaluation**
- Metrics: Recall@k, Precision@k, NDCG@k, AUPRC, log loss

**Logging**
- Directories for TensorBoard runs and checkpoints

---

## Outputs

### Data Artifacts

```
data/processed/mimic4_splits/
  train/valid/test/
    ├── dx.seqs.pkl           # Per-visit diagnosis code IDs
    ├── proc.seqs.pkl         # Per-visit procedure code IDs
    ├── treat.seqs.pkl        # Per-visit treatment code IDs
    ├── demo.npy              # Demographics per patient
    ├── severity.seqs.pkl     # Optional risk labels
    ├── notes.jsonl           # Routed notes (dx/proc/treat/ap)
    └── vocab.pkl             # Code vocabularies (driven by train)
```

### Training Artifacts

```
ckpts/best.pt               # Best checkpoint (lowest validation loss)
runs/med2vec_plus/          # TensorBoard logs
```

### Evaluation Artifacts

```
ckpts/preds/test_metrics.json    # Metrics summary (if save_preds=true)
```

### Calibration Artifacts

```
ckpts_calibrated/calibration_temperature_valid.pkl
```

---

## Evaluation & Calibration

### Prediction Aspects

The model predicts three aspects: **diagnosis**, **procedures**, **treatments** (multilabel next-visit prediction)

### Metrics

- **AUPRC** (micro/macro averaged)
- **Log loss**
- **Recall@k / Precision@k / NDCG@k** for top-k predictions

### Calibration Methods

- **Temperature scaling**: Single parameter to adjust confidence
- **Platt scaling**: Per-class logistic regression
- **Isotonic regression**: Per-class non-parametric calibration

Calibrators are fit on validation predictions and applied to the test set.

---

## MIMIC-IV Variables Used

### Hospital (hosp) Tables

- `admissions.csv.gz`: `subject_id`, `hadm_id`, `admittime`, `dischtime`, `ethnicity` (or `race` fallback)
- `patients.csv.gz`: `subject_id`, `gender`, `anchor_age`
- `diagnoses_icd.csv.gz`: `subject_id`, `hadm_id`, `icd_code`, `icd_version`
- `procedures_icd.csv.gz`: `subject_id`, `hadm_id`, `icd_code`, `icd_version`
- `prescriptions.csv.gz`: `subject_id`, `hadm_id`, `drug`, `ndc`

### Note Tables

- `discharge_detail.csv.gz`: `note_id`, `subject_id`, `field_name`, `field_value`, `field_ordinal`
- `discharge.csv.gz`: `note_id`, `subject_id`, `hadm_id`

---

## License

This code is released under the **MIT License** (see [LICENSE](LICENSE)).

**Important**: MIMIC-IV data are governed by their own usage agreements. Do not redistribute raw data.

---

## Acknowledgments
Thanks to:

- The MIMIC team at MIT for maintaining this invaluable resource
- The open-source community: PyTorch, scikit-learn, HuggingFace Transformers

Authors: Changyue (William) Zhao, Juncheng Yang, Guangxuan Chen, Minzhao Li  
Department of Biostatistics and Bioinformatics, School of Medicine, Duke University

**Questions or Issues?** Open an issue on GitHub or [contact the maintainers](mailto:changyue.zhao@duke.edu).
