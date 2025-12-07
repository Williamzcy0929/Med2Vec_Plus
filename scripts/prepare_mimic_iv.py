#!/usr/bin/env python
import argparse, os, json, pickle
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def load_notes_table(note_path: Path) -> pd.DataFrame | None:
    """Load notes from CSV/CSV.GZ or JSONL and return columns: subject_id, hadm_id, text."""
    if note_path is None or not note_path.exists():
        return None
    name = note_path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".json"):
        df = pd.read_json(note_path, lines=True)
        if "_" in df.columns:
            df = df[df["_"].isna()]
        for col in ("subject_id", "hadm_id"):
            if col not in df.columns:
                raise ValueError(f"{note_path} is missing required column: {col}")
        if "text" not in df.columns:
            for c in ("dx", "proc", "treat", "ap"):
                if c not in df.columns:
                    df[c] = ""
            df["text"] = (
                df["dx"].fillna("") + "\n\n" +
                df["proc"].fillna("") + "\n\n" +
                df["treat"].fillna("") + "\n\n" +
                df["ap"].fillna("")
            ).str.strip()
        out = df[["subject_id", "hadm_id", "text"]].copy()
        out["text"] = out["text"].astype(str).str.strip()
        out = out[out["text"].str.len() > 0]
        return out
    df = pd.read_csv(
        note_path,
        usecols=lambda c: str(c).lower() in {"subject_id", "hadm_id", "text"},
        low_memory=False,
    )
    df.columns = [c.lower() for c in df.columns]
    out = df[["subject_id", "hadm_id", "text"]].copy()
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0]
    return out

def load_routed_notes_jsonl(note_path: Path) -> pd.DataFrame | None:
    """Load routed notes.jsonl that has dx/proc/treat/ap; returns subject_id, hadm_id, dx, proc, treat, ap."""
    if note_path is None or not note_path.exists():
        return None
    name = note_path.name.lower()
    if not (name.endswith(".jsonl") or name.endswith(".json")):
        return None
    df = pd.read_json(note_path, lines=True)
    if "_" in df.columns:
        df = df[df["_"].isna()]
    required_ids = {"subject_id", "hadm_id"}
    if not required_ids.issubset(set(df.columns)):
        return None
    routed_cols = {"dx", "proc", "treat", "ap"}
    if not routed_cols.issubset(set(df.columns)):
        return None
    out = df[["subject_id", "hadm_id", "dx", "proc", "treat", "ap"]].copy()
    for c in ["dx", "proc", "treat", "ap"]:
        out[c] = out[c].fillna("").astype(str).str.strip()
    return out

def _group_icd(code: str) -> str:
    c = (str(code) or "").replace(".","").strip()
    return c[:3] if c else "UNK"

def _group_med(row) -> str:
    ndc = str(row.get("ndc", "")).replace("-","").strip()
    if ndc and ndc.lower() != "nan":
        return ndc[:5]
    drug = str(row.get("drug", "")).lower().strip()
    return drug if drug else "unknown"

def _sex_onehot(s):
    s = str(s).strip().upper()
    if s == "M": return [1.0, 0.0, 0.0]
    if s == "F": return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]

def _eth_onehot(e):
    e = str(e).lower()
    cats = ["white","black","asian","hispanic","other"]
    vec = [0.0]*len(cats)
    idx = len(cats)-1
    for i,c in enumerate(cats[:-1]):
        if c in e:
            idx = i; break
    vec[idx]=1.0
    return vec

def main():
    ap = argparse.ArgumentParser(description="Prepare Med2Vec-style sequences from MIMIC-IV")
    ap.add_argument("--mimic_dir", type=str, required=True)
    ap.add_argument("--notes_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="data/processed/mimic4_splits")
    ap.add_argument("--use_train_vocab_only", action="store_true")
    ap.add_argument("--min_visits", type=int, default=2)
    ap.add_argument("--cohort_subjects", type=str, default=None, help="Optional CSV to filter subjects")
    args = ap.parse_args()

    hosp = Path(args.mimic_dir)/"hosp"
    note_path = Path(args.notes_file) if args.notes_file else Path(args.mimic_dir)/"note"/"noteevents.csv.gz"
    out_root = Path(args.out_dir)
    for s in ["train","valid","test"]:
        (out_root/s).mkdir(parents=True, exist_ok=True)

    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        usecols=lambda c: str(c).lower() in {"subject_id", "hadm_id", "admittime", "dischtime", "ethnicity", "race"},
        parse_dates=["admittime", "dischtime"],
        low_memory=False
    )
    admissions.columns = [c.lower() for c in admissions.columns]
    if "ethnicity" not in admissions.columns:
        if "race" in admissions.columns:
            admissions["ethnicity"] = admissions["race"]
        else:
            admissions["ethnicity"] = "UNKNOWN"
    admissions = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "ethnicity"]]
    
    patients = pd.read_csv(hosp/"patients.csv.gz", usecols=["subject_id","gender","anchor_age"])
    diagnoses = pd.read_csv(hosp/"diagnoses_icd.csv.gz", usecols=["subject_id","hadm_id","icd_code","icd_version"])
    procedures = pd.read_csv(hosp/"procedures_icd.csv.gz", usecols=["subject_id","hadm_id","icd_code","icd_version"])
    presc = pd.read_csv(hosp/"prescriptions.csv.gz", usecols=["subject_id","hadm_id","drug","ndc"])

    if note_path.exists():
        routed = load_routed_notes_jsonl(note_path)
        notes = load_notes_table(note_path)
    else:
        routed = None
        notes = pd.DataFrame({"subject_id":[], "hadm_id":[], "text":[]})
    if notes is None and routed is None:
        print("[prepare_mimic_iv] No notes loaded (notes_file missing or empty). Continuing with EHR-only.")
        notes = pd.DataFrame({"subject_id":[], "hadm_id":[], "text":[]})

    if args.cohort_subjects and Path(args.cohort_subjects).exists():
        cohort = pd.read_csv(args.cohort_subjects, usecols=["subject_id"])
        sids = set(cohort["subject_id"].tolist())
        admissions = admissions[admissions["subject_id"].isin(sids)]
        patients = patients[patients["subject_id"].isin(sids)]
        diagnoses = diagnoses[diagnoses["subject_id"].isin(sids)]
        procedures = procedures[procedures["subject_id"].isin(sids)]
        presc = presc[presc["subject_id"].isin(sids)]
        if notes is not None and not notes.empty:
            notes = notes[notes["subject_id"].isin(sids)]
        if routed is not None and not routed.empty:
            routed = routed[routed["subject_id"].isin(sids)]

    pats = patients.drop_duplicates("subject_id")
    adm = admissions.merge(pats, on="subject_id", how="left")
    adm = adm.sort_values(["subject_id","admittime"])

    dx_groups = diagnoses.assign(code_group=diagnoses["icd_code"].apply(_group_icd)).groupby(["subject_id","hadm_id"])["code_group"].apply(list).reset_index()
    pr_groups = procedures.assign(code_group=procedures["icd_code"].apply(_group_icd)).groupby(["subject_id","hadm_id"])["code_group"].apply(list).reset_index()
    med_groups = presc.assign(code_group=presc.apply(_group_med, axis=1)).groupby(["subject_id","hadm_id"])["code_group"].apply(list).reset_index()

    adm = adm.merge(dx_groups, on=["subject_id","hadm_id"], how="left").rename(columns={"code_group":"dx"})
    adm = adm.merge(pr_groups, on=["subject_id","hadm_id"], how="left").rename(columns={"code_group":"proc"})
    adm = adm.merge(med_groups, on=["subject_id","hadm_id"], how="left").rename(columns={"code_group":"treat"})
    for col in ["dx","proc","treat"]:
        adm[col] = adm[col].apply(lambda x: x if isinstance(x, list) else [])

    if notes is not None and not notes.empty:
        note_by_visit = notes.groupby(["subject_id","hadm_id"])["text"].apply(
            lambda x: "\n\n".join([str(t) for t in x.dropna().tolist()[:1]])
        ).reset_index()
        adm = adm.merge(note_by_visit, on=["subject_id","hadm_id"], how="left").rename(columns={"text":"note"})
    else:
        adm["note"] = ""
    adm["note"] = adm["note"].fillna("").astype(str)

    if routed is not None and not routed.empty:
        routed_by_visit = routed.groupby(["subject_id","hadm_id"]).agg({
            "dx": "first", "proc": "first", "treat": "first", "ap": "first"
        }).reset_index()
        adm = adm.merge(routed_by_visit, on=["subject_id","hadm_id"], how="left", suffixes=("", "_routed"))
        for c in ["dx", "proc", "treat", "ap"]:
            col = c if c in adm.columns else f"{c}_routed"
            if col in adm.columns:
                adm[col] = adm[col].fillna("").astype(str)
    else:
        for c in ["dx", "proc", "treat", "ap"]:
            if c not in adm.columns:
                adm[c] = ""

    sujs = adm["subject_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(17)
    rng.shuffle(sujs)
    n = len(sujs)
    train_ids = set(sujs[: int(0.7*n)])
    valid_ids = set(sujs[int(0.7*n): int(0.8*n)])
    test_ids  = set(sujs[int(0.8*n): ])

    splits = {"train": train_ids, "valid": valid_ids, "test": test_ids}
    vocabs = {"dx": Counter(), "proc": Counter(), "treat": Counter()}

    routed_mode = (routed is not None and not routed.empty)

    for split, idset in splits.items():
        out_dir = out_root/split
        dx_seqs, pr_seqs, tr_seqs = [], [], []
        sev = []
        demo_rows = []
        write_notes = not routed_mode  # do not overwrite routed notes.jsonl
        if write_notes:
            notes_rows = []
        for sid, g in tqdm(adm[adm["subject_id"].isin(idset)].groupby("subject_id"), desc=f"{split} patients"):
            g = g.sort_values("admittime")
            for _, row in g.iterrows():
                dx_codes = [str(c) for c in row["dx"]]
                pr_codes = [str(c) for c in row["proc"]]
                tr_codes = [str(c) for c in row["treat"]]
                dx_seqs.append(dx_codes)
                pr_seqs.append(pr_codes)
                tr_seqs.append(tr_codes)
                age = float(row["anchor_age"]) if pd.notnull(row["anchor_age"]) else 60.0
                sex = _sex_onehot(row["gender"])
                eth = _eth_onehot(row["ethnicity"])
                demo_rows.append([age] + sex + eth)
                sev.append([0])
                if write_notes:
                    ap_txt = "" if pd.isna(row.get("note")) else str(row.get("note"))
                    notes_rows.append({"dx":"", "proc":"", "treat":"", "ap": ap_txt})
            dx_seqs.append([-1]); pr_seqs.append([-1]); tr_seqs.append([-1]); sev.append([-1]); demo_rows.append([0.0]*9)
            if write_notes:
                notes_rows.append({"_":-1})
            if split == "train" or not args.use_train_vocab_only:
                for codes in g["dx"].tolist(): vocabs["dx"].update([str(c) for c in codes])
                for codes in g["proc"].tolist(): vocabs["proc"].update([str(c) for c in codes])
                for codes in g["treat"].tolist(): vocabs["treat"].update([str(c) for c in codes])

        with open(out_dir/"dx.seqs.pkl", "wb") as f: pickle.dump(dx_seqs, f)
        with open(out_dir/"proc.seqs.pkl", "wb") as f: pickle.dump(pr_seqs, f)
        with open(out_dir/"treat.seqs.pkl", "wb") as f: pickle.dump(tr_seqs, f)
        with open(out_dir/"severity.seqs.pkl", "wb") as f: pickle.dump(sev, f)
        np.save(out_dir/"demo.npy", np.array(demo_rows, dtype=np.float32))
        if write_notes:
            with open(out_dir/"notes.jsonl", "w") as f:
                for r in notes_rows: f.write(json.dumps(r)+"\n")
        else:
            print(f"[prepare_mimic_iv] Routed notes.jsonl detected; skipping write for split '{split}'.")

    def build_vocab(counter: Counter):
        id2code = sorted(counter.keys())
        code2id = {c:i for i,c in enumerate(id2code)}
        return {"code2id": code2id, "id2code": id2code}

    vmap = {a: build_vocab(vocabs[a]) for a in ["dx","proc","treat"]}
    for split in ["train","valid","test"]:
        with open(out_root/split/"vocab.pkl", "wb") as f:
            pickle.dump(vmap, f)

    for split in ["train","valid","test"]:
        out_dir = out_root/split
        for a in ["dx","proc","treat"]:
            path = out_dir/f"{a}.seqs.pkl"
            with open(path, "rb") as f:
                seqs = pickle.load(f)
            m = vmap[a]["code2id"]
            seqs_id = []
            for visit in seqs:
                if len(visit)==1 and visit[0]==-1:
                    seqs_id.append([-1]); continue
                seqs_id.append([m[c] for c in visit if c in m])
            with open(path, "wb") as f:
                pickle.dump(seqs_id, f)
    print("Done. Outputs in", out_root)

if __name__ == "__main__":
    main()