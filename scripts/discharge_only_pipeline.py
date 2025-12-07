import re
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Utility: device selection
# ---------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------
# Section routing heuristics
# ---------------------------

ADMIN_TOKENS = {
    "author", "signed", "signature", "dictated", "transcribed",
    "service", "attending", "pager", "phone", "fax",
    "mrn", "csn", "account", "job", "report status", "finalized",
    "cc", "copy to"
}

PLACEHOLDER_RE = re.compile(r'^[\s\W_]+$')  # only whitespace/punct/underscore
MULTI_PUNCT_RE = re.compile(r'^[\s\-_=.]{1,}$')  # runs of separators

def is_admin_section(header: str) -> bool:
    h = (header or "").strip().lower()
    return any(tok in h for tok in ADMIN_TOKENS)

def is_meaningful_text(txt: str, min_chars: int = 12) -> bool:
    t = (txt or "").strip()
    if len(t) < min_chars:
        return False
    if PLACEHOLDER_RE.fullmatch(t) or MULTI_PUNCT_RE.fullmatch(t):
        return False
    # at least 1 letter or digit signal
    return any(ch.isalnum() for ch in t)

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

CAND_SECTION_COLS = ["section", "section_name", "section_header", "category", "label", "title", "field_name"]
CAND_TEXT_COLS = ["text", "section_text", "note_text", "field_value"]

HEADER_RE = re.compile(r'^\s*([A-Z][A-Za-z0-9 /&\-]{2,80}|[A-Za-z][A-Za-z0-9 /&\-]{2,80}:)\s*$')

ADMIN_TOKENS = {
    "author", "signed", "signature", "dictated", "transcribed",
    "service", "attending", "pager", "phone", "fax",
    "mrn", "csn", "account", "job", "report status", "finalized",
    "cc", "copy to"
}
PLACEHOLDER_RE = re.compile(r'^[\s\W_]+$')
MULTI_PUNCT_RE = re.compile(r'^[\s\-_=.]{1,}$')

def is_admin_section(header: str) -> bool:
    h = (header or "").strip().lower()
    return any(tok in h for tok in ADMIN_TOKENS)

def is_meaningful_text(txt: str, min_chars: int = 12) -> bool:
    t = (txt or "").strip()
    if len(t) < min_chars:
        return False
    if PLACEHOLDER_RE.fullmatch(t) or MULTI_PUNCT_RE.fullmatch(t):
        return False
    return any(ch.isalnum() for ch in t)

def rough_sectionize_full_text(text: str) -> Dict[str, str]:
    # Split by lines; detect simple headers (all-caps or endswith ":")
    sections: Dict[str, List[str]] = {}
    current = "ap"
    sections.setdefault(current, [])
    for raw in (text or "").splitlines():
        line = raw.strip("\n\r")
        if HEADER_RE.match(line):
            header = line.rstrip(":").strip()
            if is_admin_section(header):
                current = None
            else:
                slot = route_discharge(header)
                current = slot
                sections.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.strip():
            sections[current].append(line)
    merged = {k: "\n".join(v) for k, v in sections.items()}
    for k in ("dx", "proc", "treat", "ap"):
        merged.setdefault(k, "")
    return merged

def build_fallback_from_discharge(mimic_dir: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    path = mimic_dir / "note" / "discharge.csv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["subject_id", "hadm_id", "text"], low_memory=False)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    grp = df.groupby(["subject_id", "hadm_id"])["text"].apply(lambda s: "\n\n----\n\n".join(s)).reset_index()
    out: Dict[Tuple[int, int], Dict[str, str]] = {}
    for _, row in grp.iterrows():
        sid, hadm, txt = int(row["subject_id"]), int(row["hadm_id"]), str(row["text"])
        slots = rough_sectionize_full_text(txt)
        if any(is_meaningful_text(slots.get(s, "")) for s in ("dx","proc","treat","ap")):
            out[(sid, hadm)] = {
                "dx": slots.get("dx",""),
                "proc": slots.get("proc",""),
                "treat": slots.get("treat",""),
                "ap": slots.get("ap",""),
            }
    return out

def _choose_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _normalize(s: str) -> str:
    return (s or "").strip()


def route_discharge(section_name: str) -> str:
    """Map a discharge section header to dx/proc/treat/ap."""
    name = (section_name or "").strip().lower()
    if any(k in name for k in DISCHARGE_TO_DX):
        return "dx"
    if any(k in name for k in DISCHARGE_TO_PROC):
        return "proc"
    if any(k in name for k in DISCHARGE_TO_TREAT):
        return "treat"
    if any(k in name for k in DISCHARGE_TO_AP):
        return "ap"
    return "ap"


def load_discharge_detail(path: Path, mimic_dir: Path) -> pd.DataFrame:
    """Load discharge_detail.csv(.gz) and return canonical columns:
       subject_id, hadm_id, section, text.
       - Recognize field_name/field_value
       - If hadm_id missing, join via note_id with note/discharge.csv.gz
       - Coalesce suffixed columns (subject_id_x/_y, hadm_id_x/_y) to canonical names
    """
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # Detect section/text columns (includes field_name/field_value)
    sec_col = next((c for c in CAND_SECTION_COLS if c.lower() in df.columns), None)
    txt_col = next((c for c in CAND_TEXT_COLS    if c.lower() in df.columns), None)
    if sec_col is None or txt_col is None:
        raise ValueError(
            f"Cannot detect section/text columns in {path}. "
            f"Got columns: {list(df.columns)[:12]} ..."
        )
    sec_col = sec_col.lower()
    txt_col = txt_col.lower()

    # Join to get hadm_id if needed
    if "hadm_id" not in df.columns:
        if "note_id" not in df.columns:
            raise ValueError(
                f"{path} lacks hadm_id and note_id; cannot map to admissions."
            )
        base = mimic_dir / "note" / "discharge.csv.gz"
        if not base.exists():
            raise FileNotFoundError(
                f"Need {base} to map note_id -> hadm_id, but file was not found."
            )
        base_df = pd.read_csv(base, usecols=["note_id", "subject_id", "hadm_id"])
        base_df.columns = [c.lower() for c in base_df.columns]
        # Add suffixes so we can coalesce explicitly
        df = df.merge(base_df, on="note_id", how="left", validate="many_to_one", suffixes=("_det", "_base"))

    # --- Coalesce subject_id / hadm_id to canonical names
    def coalesce(df_, col: str):
        if col in df_.columns:
            return
        # prefer detail first, then base, then any suffix
        for cand in (f"{col}_det", f"{col}_base", f"{col}_x", f"{col}_y"):
            if cand in df_.columns:
                df_[col] = df_[cand]
                return
        # fallback: any column starting with col_
        for c in df_.columns:
            if c.startswith(col + "_"):
                df_[col] = df_[c]
                return

    coalesce(df, "subject_id")
    coalesce(df, "hadm_id")

    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        raise ValueError("subject_id/hadm_id still missing after join/coalesce.")

    # Clean text, keep order by field_ordinal if present
    df[txt_col] = df[txt_col].astype(str)
    df = df[df[txt_col].str.strip().str.len() > 0]
    if "field_ordinal" in df.columns and "note_id" in df.columns:
        df = df.sort_values(["note_id", "field_ordinal"])

    # Ensure proper dtypes
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["subject_id"] = df["subject_id"].astype("int64")

    # Select and rename canonical columns
    out = df.loc[:, ["subject_id", "hadm_id", sec_col, txt_col]].copy()
    out.rename(columns={sec_col: "section", txt_col: "text"}, inplace=True)
    return out

def aggregate_routed_text(discharge_detail: pd.DataFrame,
                          max_chars_per_field: Optional[int] = None
                          ) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Build routed text for each visit: { (sid, hadm): {"dx": "...", "proc": "...", "treat": "...", "ap": "..."} }
    """
    by_visit: Dict[Tuple[int, int], Dict[str, List[str]]] = defaultdict(lambda: {"dx": [], "proc": [], "treat": [], "ap": []})
    for _, row in discharge_detail.iterrows():
        sid = int(row["subject_id"])
        hadm = int(row["hadm_id"])
        sec  = str(row["section"])
        txt  = str(row["text"])
        if is_admin_section(sec):
            continue
        if not is_meaningful_text(txt):
            continue
        slot = route_discharge(sec)
        chunk = f"[DISCHARGE | {sec}]\n{txt.strip()}"
        by_visit[(sid, hadm)][slot].append(chunk)

    # Collapse lists and drop empty visits
    result: Dict[Tuple[int, int], Dict[str, str]] = {}
    for key, buckets in by_visit.items():
        merged = {}
        for k, lst in buckets.items():
            txt = "\n\n----\n\n".join(lst)
            if max_chars_per_field is not None and len(txt) > max_chars_per_field:
                txt = txt[:max_chars_per_field]
            merged[k] = txt
        result[key] = merged
    return result

# ---------------------------
# HF encoder (optional)
# ---------------------------

@torch.no_grad()
def encode_texts_with_hf(texts: List[str],
                         model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                         pooling: str = "cls",
                         max_length: int = 256,
                         batch_size: int = 16,
                         device: Optional[torch.device] = None) -> np.ndarray:
    """
    Encode a list of texts with a HuggingFace model. Returns float32 embeddings [N, H].
    pooling: 'cls' or 'mean' over tokens (mask-aware).
    """
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Half precision only on CUDA; MPS still requires float32
    use_fp16 = (device.type == "cuda")
    if use_fp16:
        model.half()

    all_vecs: List[np.ndarray] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        hidden = outputs.last_hidden_state  # [B, T, H]

        if pooling == "cls":
            vec = hidden[:, 0, :]  # [CLS]
        else:
            # mean pooling (mask-aware)
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            vec = summed / denom

        vec = vec.float().cpu().numpy()
        all_vecs.append(vec)

    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def build_embeddings_for_visits(visit_texts: Dict[Tuple[int, int], Dict[str, str]],
                                model_name: str,
                                pooling: str,
                                max_length: int,
                                batch_size: int,
                                device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    """
    Build per-slot embeddings (dx/proc/treat/ap). Returns dict with arrays and aligned sid/hadm arrays.
    """
    keys = list(visit_texts.keys())
    slots = ["dx", "proc", "treat", "ap"]
    result = {"subject_id": np.array([k[0] for k in keys], dtype=np.int64),
              "hadm_id":    np.array([k[1] for k in keys], dtype=np.int64)}
    for slot in slots:
        texts = [visit_texts[k].get(slot, "") for k in keys]
        vecs  = encode_texts_with_hf(texts, model_name=model_name, pooling=pooling,
                                     max_length=max_length, batch_size=batch_size, device=device)
        result[f"{slot}_emb"] = vecs.astype(np.float32)
    return result


# ---------------------------
# Presence flags & cohort selection
# ---------------------------

def compute_presence_flags(mimic_dir: Path,
                           visit_texts: Dict[Tuple[int, int], Dict[str, str]],
                           require_dx: bool, require_proc: bool, require_treat: bool, require_note: bool
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hosp = mimic_dir / "hosp"

    admissions = pd.read_csv(hosp / "admissions.csv.gz",
                             usecols=["subject_id", "hadm_id", "admittime"])
    dx = pd.read_csv(hosp / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id"])
    pr = pd.read_csv(hosp / "procedures_icd.csv.gz", usecols=["subject_id", "hadm_id"])
    rx = pd.read_csv(hosp / "prescriptions.csv.gz", usecols=["subject_id", "hadm_id"])

    dx = dx.drop_duplicates(["subject_id", "hadm_id"]).assign(has_dx=True)
    pr = pr.drop_duplicates(["subject_id", "hadm_id"]).assign(has_proc=True)
    rx = rx.drop_duplicates(["subject_id", "hadm_id"]).assign(has_treat=True)

    # Build has_note from cleaned routed texts
    kept = []
    for (sid, hadm), slots in visit_texts.items():
        if any(is_meaningful_text(slots.get(s, ""), min_chars=12) for s in ("dx", "proc", "treat", "ap")):
            kept.append((sid, hadm))
    notes_visits = pd.DataFrame(kept, columns=["subject_id", "hadm_id"])
    if notes_visits.empty:
        notes_visits = pd.DataFrame(columns=["subject_id", "hadm_id", "has_note"])
    else:
        notes_visits["has_note"] = True

    visits = admissions[["subject_id", "hadm_id"]].drop_duplicates()
    visits = visits.merge(dx, on=["subject_id", "hadm_id"], how="left")
    visits = visits.merge(pr, on=["subject_id", "hadm_id"], how="left")
    visits = visits.merge(rx, on=["subject_id", "hadm_id"], how="left")
    visits = visits.merge(notes_visits, on=["subject_id", "hadm_id"], how="left")

    for col in ["has_dx", "has_proc", "has_treat", "has_note"]:
        if col not in visits.columns:
            visits[col] = False
        else:
            visits[col] = visits[col].fillna(False)

    req_cols = []
    if require_dx: req_cols.append("has_dx")
    if require_proc: req_cols.append("has_proc")
    if require_treat: req_cols.append("has_treat")
    if require_note: req_cols.append("has_note")

    grp = visits.groupby("subject_id", as_index=True)
    visit_count = grp["hadm_id"].nunique().rename("visit_count")
    if req_cols:
        all_required = grp[req_cols].min().all(axis=1).rename("all_required_present")
    else:
        all_required = pd.Series(True, index=visit_count.index, name="all_required_present")

    subjects = pd.concat([visit_count, all_required], axis=1).reset_index()
    return visits, subjects


def select_cohort(subjects: pd.DataFrame,
                  visits: pd.DataFrame,
                  min_visits: int,
                  sample_n: int,
                  seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Apply selection rules and sample. Returns (cohort_subjects, cohort_visits, drop_report).
    """
    report = {}

    n_subjects0 = subjects.shape[0]
    report["subjects_total"] = int(n_subjects0)

    # Rule 1: visit_count > 2 (i.e., >= 3)
    cand = subjects[subjects["visit_count"] > (min_visits - 1)]
    report["drop_insufficient_visits"] = int(n_subjects0 - cand.shape[0])

    # Rule 2: no missing across all visits for required aspects
    cand2 = cand[cand["all_required_present"] == True]
    report["drop_missing_required"] = int(cand.shape[0] - cand2.shape[0])

    # Uniform sample
    rng = np.random.default_rng(seed)
    subs = cand2["subject_id"].to_numpy()
    if subs.size > sample_n:
        pick = rng.choice(subs, size=sample_n, replace=False)
    else:
        pick = subs

    cohort_subjects = cand2[cand2["subject_id"].isin(pick)].copy()
    cohort_subjects = cohort_subjects[["subject_id", "visit_count"]].sort_values("subject_id")

    # Cohort visits = all visits of selected subjects
    cohort_visits = visits[visits["subject_id"].isin(cohort_subjects["subject_id"])].copy()
    cohort_visits = cohort_visits.sort_values(["subject_id", "hadm_id"])

    report["subjects_selected"] = int(cohort_subjects.shape[0])
    report["visits_selected"] = int(cohort_visits.shape[0])
    return cohort_subjects, cohort_visits, report


# ---------------------------
# Emit split-ready notes.jsonl
# ---------------------------

def build_subject_splits_from_cohort(mimic_dir: Path,
                                     cohort_subjects_csv: Path,
                                     seed: int = 17,
                                     train_ratio: float = 0.7,
                                     valid_ratio: float = 0.1) -> Dict[str, List[int]]:
    subs = pd.read_csv(cohort_subjects_csv, usecols=["subject_id"])
    subj = subs["subject_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(subj)
    n = len(subj)
    n_tr = int(train_ratio * n)
    n_va = int(valid_ratio * n)
    train_ids = subj[:n_tr]
    valid_ids = subj[n_tr:n_tr+n_va]
    test_ids  = subj[n_tr+n_va:]
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def write_notes_jsonl_for_split(mimic_dir: Path,
                                split_subjects: List[int],
                                visit_texts: Dict[Tuple[int, int], Dict[str, str]],
                                out_jsonl: Path):
    hosp = mimic_dir / "hosp"
    adm = pd.read_csv(hosp / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime"],
                      parse_dates=["admittime"])
    adm = adm[adm["subject_id"].isin(split_subjects)]
    adm = adm.sort_values(["subject_id", "admittime"])

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.rename(out_jsonl.with_suffix(out_jsonl.suffix + ".bak"))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sid, g in adm.groupby("subject_id"):
            g = g.sort_values("admittime")
            for _, row in g.iterrows():
                key = (int(row["subject_id"]), int(row["hadm_id"]))
                slots = visit_texts.get(key, {"dx": "", "proc": "", "treat": "", "ap": ""})
                rec = {
                    "dx": _normalize(slots.get("dx", "")),
                    "proc": _normalize(slots.get("proc", "")),
                    "treat": _normalize(slots.get("treat", "")),
                    "ap": _normalize(slots.get("ap", "")),
                    "subject_id": int(row["subject_id"]),
                    "hadm_id": int(row["hadm_id"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write(json.dumps({"_": -1}) + "\n")


# ---------------------------
# Main CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Discharge-only pipeline: route notes -> HF embeddings -> cohort -> splits")
    ap.add_argument("--mimic_dir", type=str, required=True, help="Root of MIMIC-IV (with hosp/, note/)")
    ap.add_argument("--discharge_detail", type=str, required=True, help="Path to note/discharge_detail.csv.gz")
    ap.add_argument("--processed_root", type=str, default="data/processed/mimic4_splits",
                    help="Where train/valid/test folders live")
    ap.add_argument("--cohort_out_dir", type=str, default="./out/cohort_discharge_only",
                    help="Where to write cohort_subjects.csv, cohort_visits.csv, reports")
    # Cohort rules
    ap.add_argument("--min_visits", type=int, default=3)
    ap.add_argument("--require_dx", action="store_true", default=True)
    ap.add_argument("--no-require_dx", dest="require_dx", action="store_false")
    ap.add_argument("--require_proc", action="store_true", default=True)
    ap.add_argument("--no-require_proc", dest="require_proc", action="store_false")
    ap.add_argument("--require_treat", action="store_true", default=True)
    ap.add_argument("--no-require_treat", dest="require_treat", action="store_false")
    ap.add_argument("--require_note", action="store_true", default=True)
    ap.add_argument("--no-require_note", dest="require_note", action="store_false")
    ap.add_argument("--sample_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--split_seed", type=int, default=17)
    # Note text routing
    ap.add_argument("--max_chars_per_field", type=int, default=8192, help="Optional char cap per field")
    # HF encoder options (optional caching)
    ap.add_argument("--encode_with_hf", action="store_true", help="If set, precompute HF embeddings for each slot")
    ap.add_argument("--hf_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    mimic_dir = Path(args.mimic_dir)
    processed_root = Path(args.processed_root)
    cohort_out = Path(args.cohort_out_dir)
    cohort_out.mkdir(parents=True, exist_ok=True)

    # Step 1: load discharge detail and route into slots
    print("[1/6] Loading discharge_detail and routing sections ...")
    discharge_df = load_discharge_detail(Path(args.discharge_detail), mimic_dir)
    visit_texts = aggregate_routed_text(discharge_df, max_chars_per_field=args.max_chars_per_field)
    print(f"  Routed texts built for {len(visit_texts)} visits.")
    if len(visit_texts) == 0:
        print("  No usable sections found in discharge_detail; falling back to full discharge text ...")
        fb = build_fallback_from_discharge(mimic_dir)
        print(f"  Fallback built for {len(fb)} visits from note/discharge.csv.gz")
        visit_texts = fb
        if len(visit_texts) == 0:
            print("  Fallback also empty. Consider running with --no-require_note to inspect EHR-only cohort.")
            exit(1)

    # Step 2: (optional) precompute HF embeddings per visit
    if args.encode_with_hf:
        print("[2/6] Encoding routed texts with HuggingFace model ...")
        device = pick_device()
        print(f"  Using device: {device}")
        emb = build_embeddings_for_visits(
            visit_texts,
            model_name=args.hf_model,
            pooling=args.pooling,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        # Save one global file (sid/hadm aligned) under cohort_out_dir
        np.savez(cohort_out / "notes_embeddings_discharge_only.npz", **emb)
        print(f"  Saved embeddings: {cohort_out/'notes_embeddings_discharge_only.npz'}")

    # Step 3: merge with EHR -> presence flags
    print("[3/6] Computing presence flags from hosp tables + notes presence ...")
    visits_df, subjects_df = compute_presence_flags(
        mimic_dir=mimic_dir,
        visit_texts=visit_texts,
        require_dx=args.require_dx,
        require_proc=args.require_proc,
        require_treat=args.require_treat,
        require_note=args.require_note,
    )

    # Step 4: cohort selection + sampling
    print("[4/6] Applying cohort selection rules and uniform sampling ...")
    cohort_subjects, cohort_visits, report = select_cohort(
        subjects=subjects_df,
        visits=visits_df,
        min_visits=args.min_visits,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    # Audit outputs
    cohort_subjects.to_csv(cohort_out / "cohort_subjects.csv", index=False)
    cohort_visits.to_csv(cohort_out / "cohort_visits.csv", index=False)
    with open(cohort_out / "drop_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cohort_out / "manifest.json", "w") as f:
        json.dump({
            "mimic_dir": str(mimic_dir),
            "discharge_detail": str(args.discharge_detail),
            "min_visits": args.min_visits,
            "require_dx": args.require_dx,
            "require_proc": args.require_proc,
            "require_treat": args.require_treat,
            "require_note": args.require_note,
            "sample_n": args.sample_n,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "encode_with_hf": bool(args.encode_with_hf),
            "hf_model": args.hf_model if args.encode_with_hf else None,
            "pooling": args.pooling if args.encode_with_hf else None,
            "max_length": args.max_length if args.encode_with_hf else None,
            "batch_size": args.batch_size if args.encode_with_hf else None,
        }, f, indent=2)
    print(f"  Wrote cohort files to: {cohort_out}")

    # Step 5: subject-level splits using selected cohort
    print("[5/6] Building subject splits (70/10/20) ...")
    splits = build_subject_splits_from_cohort(
        mimic_dir=mimic_dir,
        cohort_subjects_csv=cohort_out / "cohort_subjects.csv",
        seed=args.split_seed,
        train_ratio=0.7,
        valid_ratio=0.1,
    )

    # Step 6: write split-ready notes.jsonl
    print("[6/6] Writing routed notes.jsonl under processed_root ...")
    for split in ["train", "valid", "test"]:
        out_jsonl = processed_root / split / "notes.jsonl"
        write_notes_jsonl_for_split(
            mimic_dir=mimic_dir,
            split_subjects=splits[split],
            visit_texts=visit_texts,
            out_jsonl=out_jsonl,
        )
        print(f"  Wrote {out_jsonl}")

    print("Done. You can now run prepare_mimic_iv.py to build code sequences for these splits.")
    print("Example:")
    print("  python scripts/prepare_mimic_iv.py --mimic_dir /data/mimic-iv-2.2 "
          "--notes_file data/processed/mimic4_splits/train/notes.jsonl "
          "--out_dir data/processed/mimic4_splits --use_train_vocab_only")
    print("  (prepare_mimic_iv.py will read hosp tables to create dx/proc/treat sequences.)")


if __name__ == "__main__":
    main()