#!/usr/bin/env python
import argparse, os, sys, json, time, logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path
import numpy as np

try:
    import polars as pl
    HAVE_PL = True
except Exception:
    HAVE_PL = False
    import pandas as pd

@dataclass
class Config:
    mimic_dir: str
    notes_file: str
    min_visits: int
    require_dx: bool
    require_proc: bool
    require_treat: bool
    require_note: bool
    sample_n: int
    seed: int
    engine: str
    pg_dsn: Optional[str]
    out_dir: str
    log_level: str

def setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="MIMIC-IV cohort selection")
    ap.add_argument("--mimic_dir", type=str, required=True)
    ap.add_argument("--notes_file", type=str, default=None)
    ap.add_argument("--min_visits", type=int, default=3)
    ap.add_argument("--require_dx", dest="require_dx", action="store_true", default=True)
    ap.add_argument("--no-require_dx", dest="require_dx", action="store_false")
    ap.add_argument("--require_proc", dest="require_proc", action="store_true", default=True)
    ap.add_argument("--no-require_proc", dest="require_proc", action="store_false")
    ap.add_argument("--require_treat", dest="require_treat", action="store_true", default=True)
    ap.add_argument("--no-require_treat", dest="require_treat", action="store_false")
    ap.add_argument("--require_note", dest="require_note", action="store_true", default=True)
    ap.add_argument("--no-require_note", dest="require_note", action="store_false")
    ap.add_argument("--sample_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--engine", type=str, choices=["csv","postgres"], default="csv")
    ap.add_argument("--pg_dsn", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_level", type=str, default="INFO")
    args = ap.parse_args()
    mimic_dir = Path(args.mimic_dir)
    notes_file = args.notes_file or str(mimic_dir / "note" / "noteevents.csv.gz")
    return Config(str(mimic_dir), notes_file, args.min_visits, args.require_dx, args.require_proc,
                  args.require_treat, args.require_note, args.sample_n, args.seed,
                  args.engine, args.pg_dsn, args.out_dir, args.log_level)

def read_csv_lazy(path: str, columns: list):
    if HAVE_PL:
        return pl.scan_csv(path, has_header=True, infer_schema_length=0, try_parse_dates=True).select([pl.col(c) for c in columns])
    else:
        return pd.read_csv(path, usecols=columns)

def compute_presence_flags_csv(cfg: Config):
    hosp = Path(cfg.mimic_dir) / "hosp"
    admissions = read_csv_lazy(str(hosp/"admissions.csv.gz"), ["subject_id","hadm_id","admittime","dischtime"])
    diag = read_csv_lazy(str(hosp/"diagnoses_icd.csv.gz"), ["subject_id","hadm_id","icd_code","icd_version"])
    proc = read_csv_lazy(str(hosp/"procedures_icd.csv.gz"), ["subject_id","hadm_id","icd_code","icd_version"])
    pres = read_csv_lazy(str(hosp/"prescriptions.csv.gz"), ["subject_id","hadm_id","drug","ndc"])
    notes_path = Path(cfg.notes_file)
    if not notes_path.exists() and cfg.require_note:
        logging.error("Notes file is required but not found: %s", cfg.notes_file); sys.exit(2)
    if notes_path.exists():
        notes = read_csv_lazy(str(notes_path), ["subject_id","hadm_id","text"])
    else:
        if HAVE_PL:
            notes = pl.DataFrame({"subject_id": [], "hadm_id": [], "text": []}).lazy()
        else:
            notes = pd.DataFrame({"subject_id": [], "hadm_id": [], "text": []})
    if HAVE_PL:
        v = admissions.select(["subject_id","hadm_id"]).unique(maintain_order=True)
        dx_flag = diag.select(["subject_id","hadm_id"]).group_by(["subject_id","hadm_id"]).len().with_columns(pl.lit(True).alias("has_dx")).select(["subject_id","hadm_id","has_dx"])
        pr_flag = proc.select(["subject_id","hadm_id"]).group_by(["subject_id","hadm_id"]).len().with_columns(pl.lit(True).alias("has_proc")).select(["subject_id","hadm_id","has_proc"])
        tr_flag = pres.select(["subject_id","hadm_id"]).group_by(["subject_id","hadm_id"]).len().with_columns(pl.lit(True).alias("has_treat")).select(["subject_id","hadm_id","has_treat"])
        if isinstance(notes, pl.LazyFrame):
            nt_flag = notes.select([
                pl.col("subject_id"), pl.col("hadm_id"),
                (pl.col("text").cast(pl.Utf8, strict=False).str.strip_chars().str.len_bytes() > 0).alias("nonempty")
            ]).group_by(["subject_id","hadm_id"]).agg(pl.col("nonempty").any().alias("has_note"))
        else:
            nt_flag = pl.DataFrame({"subject_id": [], "hadm_id": [], "has_note": []}).lazy()
        out = v.lazy()             .join(dx_flag.lazy(), on=["subject_id","hadm_id"], how="left")             .join(pr_flag.lazy(), on=["subject_id","hadm_id"], how="left")             .join(tr_flag.lazy(), on=["subject_id","hadm_id"], how="left")             .join(nt_flag, on=["subject_id","hadm_id"], how="left")             .with_columns([
                pl.col("has_dx").fill_null(False),
                pl.col("has_proc").fill_null(False),
                pl.col("has_treat").fill_null(False),
                pl.col("has_note").fill_null(False),
            ]).collect()
        return out, {"visits": out.height}
    else:
        v = admissions[["subject_id","hadm_id"]].drop_duplicates()
        dx_flag = diag.groupby(["subject_id","hadm_id"], as_index=False).size().assign(has_dx=True)[["subject_id","hadm_id","has_dx"]]
        pr_flag = proc.groupby(["subject_id","hadm_id"], as_index=False).size().assign(has_proc=True)[["subject_id","hadm_id","has_proc"]]
        tr_flag = pres.groupby(["subject_id","hadm_id"], as_index=False).size().assign(has_treat=True)[["subject_id","hadm_id","has_treat"]]
        if not notes_path.exists():
            nt_flag = v.assign(has_note=False)[["subject_id","hadm_id","has_note"]].copy()
        else:
            n = notes[["subject_id","hadm_id","text"]].copy()
            n["has_note"] = n["text"].astype(str).str.strip().str.len() > 0
            nt_flag = n.groupby(["subject_id","hadm_id"], as_index=False)["has_note"].any()
        out = v.merge(dx_flag, on=["subject_id","hadm_id"], how="left")                .merge(pr_flag, on=["subject_id","hadm_id"], how="left")                .merge(tr_flag, on=["subject_id","hadm_id"], how="left")                .merge(nt_flag, on=["subject_id","hadm_id"], how="left")
        for c in ["has_dx","has_proc","has_treat","has_note"]:
            out[c] = out[c].fillna(False)
        return out, {"visits": len(out)}

def compute_presence_flags_pg(cfg: Config):
    import psycopg2, pandas as pd
    q_visits = """
    with v as (
      select a.subject_id, a.hadm_id
      from hosp.admissions a
    ), dx as (
      select subject_id, hadm_id, count(*)>0 as has_dx
      from hosp.diagnoses_icd group by 1,2
    ), pr as (
      select subject_id, hadm_id, count(*)>0 as has_proc
      from hosp.procedures_icd group by 1,2
    ), tr as (
      select subject_id, hadm_id, count(*)>0 as has_treat
      from hosp.prescriptions group by 1,2
    ), nt as (
      select subject_id, hadm_id, bool_or(length(trim(coalesce(text,'')))>0) as has_note
      from note.noteevents group by 1,2
    )
    select v.subject_id, v.hadm_id,
           coalesce(dx.has_dx,false) as has_dx,
           coalesce(pr.has_proc,false) as has_proc,
           coalesce(tr.has_treat,false) as has_treat,
           coalesce(nt.has_note,false) as has_note
    from v
    left join dx using(subject_id, hadm_id)
    left join pr using(subject_id, hadm_id)
    left join tr using(subject_id, hadm_id)
    left join nt using(subject_id, hadm_id);
    """
    with psycopg2.connect(cfg.pg_dsn) as con:
        df = pd.read_sql(q_visits, con)
    return df, {"visits": len(df)}

def aggregate_and_filter(cfg: Config, visits):
    if HAVE_PL and isinstance(visits, pl.DataFrame):
        vc = visits.group_by("subject_id").agg(pl.n_unique("hadm_id").alias("visit_count"))
        flags = visits.group_by("subject_id").agg([
            pl.all_horizontal("has_dx").alias("all_dx"),
            pl.all_horizontal("has_proc").alias("all_proc"),
            pl.all_horizontal("has_treat").alias("all_treat"),
            pl.all_horizontal("has_note").alias("all_note"),
        ])
        subj = vc.join(flags, on="subject_id", how="inner")
        drop_report = {}
        start = subj.height
        kept_min = subj.filter(pl.col("visit_count") > (cfg.min_visits - 1))
        drop_report["drop_insufficient_visits"] = int(start - kept_min.height)
        cond = pl.lit(True)
        if cfg.require_dx: cond = cond & pl.col("all_dx")
        if cfg.require_proc: cond = cond & pl.col("all_proc")
        if cfg.require_treat: cond = cond & pl.col("all_treat")
        if cfg.require_note: cond = cond & pl.col("all_note")
        kept_all = kept_min.filter(cond)
        drop_report["drop_missing_required"] = int(kept_min.height - kept_all.height)
        return kept_all, drop_report
    else:
        import pandas as pd, numpy as np
        vc = visits.groupby("subject_id", as_index=False)["hadm_id"].nunique().rename(columns={"hadm_id":"visit_count"})
        flags = visits.groupby("subject_id", as_index=False).agg({
            "has_dx":"all","has_proc":"all","has_treat":"all","has_note":"all"
        }).rename(columns={"has_dx":"all_dx","has_proc":"all_proc","has_treat":"all_treat","has_note":"all_note"})
        subj = vc.merge(flags, on="subject_id", how="inner")
        drop_report = {}
        start = len(subj)
        kept_min = subj[subj["visit_count"] > (cfg.min_visits - 1)]
        drop_report["drop_insufficient_visits"] = int(start - len(kept_min))
        cond = np.ones(len(kept_min), dtype=bool)
        if cfg.require_dx: cond &= kept_min["all_dx"].values
        if cfg.require_proc: cond &= kept_min["all_proc"].values
        if cfg.require_treat: cond &= kept_min["all_treat"].values
        if cfg.require_note: cond &= kept_min["all_note"].values
        kept_all = kept_min.loc[cond]
        drop_report["drop_missing_required"] = int(len(kept_min) - len(kept_all))
        return kept_all, drop_report

def sample_subjects(cfg: Config, subjects):
    n = subjects.height if HAVE_PL else len(subjects)
    target = min(cfg.sample_n, n)
    if HAVE_PL:
        return subjects.sample(n=target, with_replacement=False, shuffle=True, seed=cfg.seed) if n>0 else subjects.head(0)
    else:
        import pandas as pd, numpy as np
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(np.arange(n), size=target, replace=False) if n>0 else np.array([], dtype=int)
        return subjects.iloc[idx]

def write_outputs(cfg: Config, subjects, visits, drop_report, counts):
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if HAVE_PL and isinstance(subjects, pl.DataFrame):
        subjects.write_csv(out_dir/"cohort_subjects.csv"); visits.write_csv(out_dir/"cohort_visits.csv")
    else:
        subjects.to_csv(out_dir/"cohort_subjects.csv", index=False); visits.to_csv(out_dir/"cohort_visits.csv", index=False)
    manifest = {"config": cfg.__dict__, "visits_total": counts["visits"]}
    (out_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir/"drop_report.json").write_text(json.dumps(drop_report, indent=2), encoding="utf-8")

def validate_outputs(cfg: Config, subjects, visits):
    if HAVE_PL and isinstance(subjects, pl.DataFrame):
        vc = visits.filter(pl.col("subject_id").is_in(subjects["subject_id"])).group_by("subject_id").agg(pl.n_unique("hadm_id").alias("vc"))
        assert (vc["vc"] > (cfg.min_visits - 1)).all()
        reqs = []
        if cfg.require_dx: reqs.append("has_dx")
        if cfg.require_proc: reqs.append("has_proc")
        if cfg.require_treat: reqs.append("has_treat")
        if cfg.require_note: reqs.append("has_note")
        for r in reqs:
            bad = visits.filter((pl.col(r)==False) & (pl.col("subject_id").is_in(subjects["subject_id"]))).height
            assert bad == 0
    else:
        sids = set(subjects["subject_id"].tolist())
        vc = visits[visits["subject_id"].isin(sids)].groupby("subject_id")["hadm_id"].nunique()
        assert (vc > (cfg.min_visits - 1)).all()
        reqs = []
        if cfg.require_dx: reqs.append("has_dx")
        if cfg.require_proc: reqs.append("has_proc")
        if cfg.require_treat: reqs.append("has_treat")
        if cfg.require_note: reqs.append("has_note")
        for r in reqs:
            bad = visits[(visits["subject_id"].isin(sids)) & (~visits[r])]
            assert len(bad) == 0

def main():
    cfg = parse_args(); setup_logging(cfg.log_level)
    if cfg.engine == "csv":
        visits, counts = compute_presence_flags_csv(cfg)
    else:
        visits, counts = compute_presence_flags_pg(cfg)
    subjects_all, drop_report = aggregate_and_filter(cfg, visits)
    subjects = sample_subjects(cfg, subjects_all)
    if HAVE_PL and isinstance(visits, pl.DataFrame):
        sids = subjects["subject_id"]; visits_out = visits.filter(pl.col("subject_id").is_in(sids))
    else:
        sids = set(subjects["subject_id"].tolist()); visits_out = visits[visits["subject_id"].isin(sids)].copy()
    validate_outputs(cfg, subjects, visits_out)
    write_outputs(cfg, subjects, visits_out, drop_report, counts)
    print("Done:", cfg.out_dir)

if __name__ == "__main__":
    main()
