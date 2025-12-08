#!/usr/bin/env python
import argparse, os, json, copy, subprocess, sys, time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------

def load_yaml(p: Path) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def run(cmd, env=None):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def find_ckpt_dir(cfg: dict) -> Path:
    return Path(cfg["logging"]["ckpt_dir"])

def read_eval_json(ckpt_dir: Path, split: str) -> dict:
    p = ckpt_dir / "preds" / f"{split}_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Could not find metrics: {p}")
    with open(p, "r") as f:
        return json.load(f)

def summarize_metrics(eval_dict: dict, aspects=("dx","proc","treat")) -> dict:
    out = {}
    # AUPRC (micro), logloss, recall@k=10 (example) per aspect
    for a in aspects:
        if "aspect" not in eval_dict or a not in eval_dict["aspect"]:
            continue
        ares = eval_dict["aspect"][a]
        out[f"{a}.micro_auprc"] = float(ares.get("micro_auprc", float("nan")))
        out[f"{a}.logloss"] = float(ares.get("logloss", float("nan")))
        tk = ares.get("topk", {}).get(10, {})
        out[f"{a}.recall@10"] = float(tk.get("recall", float("nan")))
        out[f"{a}.precision@10"] = float(tk.get("precision", float("nan")))
        out[f"{a}.ndcg@10"] = float(tk.get("ndcg", float("nan")))
    # Optional risk head
    if "risk" in eval_dict and eval_dict["risk"]:
        for k in ["auroc","auprc","brier"]:
            if k in eval_dict["risk"]:
                out[f"risk.{k}"] = float(eval_dict["risk"][k])
    # Overall convenience metrics
    vals = [out.get(f"{a}.micro_auprc", np.nan) for a in aspects]
    out["mean.micro_auprc"] = float(np.nanmean(vals)) if len(vals) else float("nan")
    return out

# ----------------------------
# Variant factory
# ----------------------------

def variants_from_base(base_cfg: dict) -> dict:
    """Return a dict: {tag: cfg_dict} for ablation."""
    V = {}

    # Baseline
    base = copy.deepcopy(base_cfg)
    base["logging"]["ckpt_dir"] = "ckpts_ablation/base"
    V["base"] = base

    # Remove text modality (multi-modality ablation)
    no_text = copy.deepcopy(base_cfg)
    no_text["model"]["text_enabled"] = False
    no_text["loss"]["lambda_text"] = 0.0
    no_text["logging"]["ckpt_dir"] = "ckpts_ablation/ablate_text"
    V["ablate_text"] = no_text

    # Remove demographics
    no_demo = copy.deepcopy(base_cfg)
    no_demo["model"]["use_demo"] = False
    no_demo["logging"]["ckpt_dir"] = "ckpts_ablation/ablate_demo"
    V["ablate_demo"] = no_demo

    # Remove intra-aspect (tensor) loss
    no_intra = copy.deepcopy(base_cfg)
    no_intra["loss"]["lambda_intra"] = 0.0
    no_intra["logging"]["ckpt_dir"] = "ckpts_ablation/ablate_intra"
    V["ablate_intra"] = no_intra

    # Fusion variant: share code embeddings on/off (flip from base)
    flip_share = copy.deepcopy(base_cfg)
    flip_share["model"]["share_code_embeddings"] = (not bool(base_cfg["model"].get("share_code_embeddings", False)))
    tag = "fusion_share_on" if flip_share["model"]["share_code_embeddings"] else "fusion_share_off"
    flip_share["logging"]["ckpt_dir"] = f"ckpts_ablation/{tag}"
    V[tag] = flip_share

    # Fusion capacity smaller (d_fuse halved)
    small_fuse = copy.deepcopy(base_cfg)
    small_fuse["model"]["d_fuse"] = max(64, int(base_cfg["model"]["d_fuse"] // 2))
    small_fuse["logging"]["ckpt_dir"] = "ckpts_ablation/fusion_small"
    V["fusion_small"] = small_fuse

    # Temporal backbone: GRU
    gru_cfg = copy.deepcopy(base_cfg)
    gru_cfg["model"]["temporal"] = "gru"
    gru_cfg["logging"]["ckpt_dir"] = "ckpts_ablation/temporal_gru"
    V["temporal_gru"] = gru_cfg

    # Temporal backbone: Transformer
    trans_cfg = copy.deepcopy(base_cfg)
    trans_cfg["model"]["temporal"] = "transformer"
    trans_cfg["logging"]["ckpt_dir"] = "ckpts_ablation/temporal_transformer"
    V["temporal_transformer"] = trans_cfg

    # Med2Vec-compat on (if supported)
    mv_on = copy.deepcopy(base_cfg)
    mv_on["model"]["med2vec_compat"] = True
    mv_on["logging"]["ckpt_dir"] = "ckpts_ablation/med2vec_compat_on"
    V["med2vec_compat_on"] = mv_on

    # Codes-only (stronger ablation): no text, no demo, no intra penalty change kept
    codes_only = copy.deepcopy(base_cfg)
    codes_only["model"]["text_enabled"] = False
    codes_only["loss"]["lambda_text"] = 0.0
    codes_only["model"]["use_demo"] = False
    codes_only["logging"]["ckpt_dir"] = "ckpts_ablation/codes_only"
    V["codes_only"] = codes_only

    return V

# ----------------------------
# Plotting
# ----------------------------

def plot_bar_deltas(df: pd.DataFrame, out_png: Path, metric_col: str, base_row="base"):
    # Compute delta vs base for each variant and aspect summary
    if base_row not in df["variant"].values:
        raise ValueError("Base variant not found in results")
    base_val = float(df.loc[df["variant"]==base_row, metric_col].values[0])
    df2 = df.copy()
    df2["delta"] = df2[metric_col] - base_val

    order = df2.sort_values("delta", ascending=False)["variant"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(order)), [df2.loc[df2["variant"]==v, "delta"].values[0] for v in order])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(f"Δ {metric_col} vs base")
    ax.set_title("Component-wise Ablation: improvement over base")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)

def plot_grouped_aspects(df: pd.DataFrame, out_png: Path, aspects=("dx","proc","treat")):
    # Grouped bar: micro_auprc per aspect for each variant
    variants = df["variant"].tolist()
    vals = []
    for a in aspects:
        vals.append(df[f"{a}.micro_auprc"].tolist())
    vals = np.array(vals)  # [A, N]
    A, N = vals.shape
    x = np.arange(N)
    width = 0.8 / max(1, A)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, a in enumerate(aspects):
        ax.bar(x + i*width - (A-1)*width/2, vals[i], width=width, label=a.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Micro AUPRC")
    ax.set_title("Micro AUPRC by aspect across ablations")
    ax.legend(ncol=len(aspects))
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)

# ----------------------------
# Main driver
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Run component-wise ablation and visualize gains.")
    ap.add_argument("--base_config", type=str, default="configs/default.yaml")
    ap.add_argument("--out_dir", type=str, default="out/ablation")
    ap.add_argument("--split", type=str, default="test", choices=["valid","test"])
    ap.add_argument("--max_epochs", type=int, default=None, help="Override epochs for quick ablation")
    ap.add_argument("--reuse", action="store_true", help="Skip training if best.pt exists")
    ap.add_argument("--hf_home", type=str, default=None, help="Optional HF cache dir for offline")
    ap.add_argument("--transformers_offline", action="store_true", help="Set TRANSFORMERS_OFFLINE=1")
    ap.add_argument("--binary_eval", action="store_true", help="Include binary metrics reporting")
    ap.add_argument("--bin_threshold_mode", type=str, default="valid_per_class_opt_f1",
                    choices=["fixed","global_opt_f1","per_class_opt_f1","valid_global_opt_f1","valid_per_class_opt_f1"])
    ap.add_argument("--bin_fixed_threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg0 = load_yaml(Path(args.base_config))
    if args.max_epochs is not None:
        cfg0 = copy.deepcopy(cfg0)
        cfg0["optim"]["max_epochs"] = int(args.max_epochs)

    variants = variants_from_base(cfg0)
    results_rows = []

    # Environment for subprocess (optional offline HF)
    env = os.environ.copy()
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
        env["TRANSFORMERS_CACHE"] = args.hf_home
    if args.transformers_offline:
        env["TRANSFORMERS_OFFLINE"] = "1"

    # Save resolved base config for reference
    dump_yaml(cfg0, out_dir/"_resolved_base.yaml")

    for tag, cfg in variants.items():
        print(f"\n=== Variant: {tag} ===")
        # Ensure unique ckpt dir under out_dir
        ckpt_dir = out_dir / Path(cfg["logging"]["ckpt_dir"]).name
        cfg["logging"]["ckpt_dir"] = str(ckpt_dir)
        cfg_path = out_dir / "configs" / f"{tag}.yaml"
        dump_yaml(cfg, cfg_path)

        # Train if needed
        best_pt = ckpt_dir / "best.pt"
        if args.reuse and best_pt.exists():
            print(f"[SKIP TRAIN] Found {best_pt}")
        else:
            run([sys.executable, "-m", "src.med2vec_plus.train", "--config", str(cfg_path)], env=env)

        # Evaluate
        eval_cmd = [
            sys.executable, "-m", "src.med2vec_plus.evaluate",
            "--config", str(cfg_path),
            "--ckpt", str(best_pt),
            "--split", args.split
        ]
        if args.binary_eval:
            eval_cmd += [
                "--binary_eval",
                "--bin_threshold_mode", args.bin_threshold_mode,
                "--bin_fixed_threshold", str(args.bin_fixed_threshold)
            ]
        run(eval_cmd, env=env)

        # Read metrics
        eval_json = read_eval_json(ckpt_dir, args.split)
        flat = summarize_metrics(eval_json)
        flat["variant"] = tag
        results_rows.append(flat)

    # Collect and save table
    df = pd.DataFrame(results_rows).set_index("variant").reset_index()
    df.to_csv(out_dir/"ablation_results.csv", index=False)

    # Plots
    # 1) Improvement over base on mean.micro_auprc
    if "base" in df["variant"].values:
        plot_bar_deltas(df, out_dir/"ablation_delta_mean_micro_auprc.png", "mean.micro_auprc", base_row="base")
    # 2) Grouped bars by aspect micro_auprc
    for a in ["dx.micro_auprc","proc.micro_auprc","treat.micro_auprc"]:
        if a not in df.columns:
            print(f"[WARN] Missing column in results: {a}")
    plot_grouped_aspects(df, out_dir/"ablation_aspect_micro_auprc.png")

    # Also write a simple JSON lines log for reproducibility
    with open(out_dir/"ablation_results.jsonl", "w") as f:
        for row in results_rows:
            f.write(json.dumps(row) + "\n")

    print("\n[Ablation] Saved artifacts to:", out_dir)
    print(" -", out_dir/"ablation_results.csv")
    print(" -", out_dir/"ablation_delta_mean_micro_auprc.png")
    print(" -", out_dir/"ablation_aspect_micro_auprc.png")

if __name__ == "__main__":
    main()