# src/med2vec_plus/evaluate.py
import os, argparse, yaml, json, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.multiclass import OneVsRestClassifier
from .utils import set_seed, get_device, load_checkpoint
from .data.dataset import PatientSequenceDataset
from .data.collate import pad_visits
from .data.vocab import load_multi_vocab, ASPECTS
from .models.model import Med2VecPlus
# NEW: import calibrator helpers
from .calibrate import load_calibrator, apply_calibration_to_logits, apply_isotonic_to_probs

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def topk_metrics(y_true_multi, scores, ks):
    out = {}
    V = scores.shape[1]
    for k in ks:
        topk_idx = np.argpartition(-scores, kth=min(k-1, V-1), axis=1)[:, :k]
        rel = np.take_along_axis(y_true_multi, topk_idx, axis=1)
        recall_at_k = (rel.sum(axis=1) / np.clip(y_true_multi.sum(axis=1), 1, None)).mean()
        precision_at_k = rel.mean()
        gains = rel / np.log2(np.arange(2, k+2))
        sorted_true = -np.sort(-y_true_multi, axis=1)[:, :k]
        ideal = (sorted_true / np.log2(np.arange(2, k+2))).sum(axis=1)
        ndcg = (gains.sum(axis=1) / np.clip(ideal, 1e-9, None)).mean()
        out[k] = {"recall": float(recall_at_k), "precision": float(precision_at_k), "ndcg": float(ndcg)}
    return out

def micro_macro_auprc(y_true, y_score):
    micro = average_precision_score(y_true.ravel(), y_score.ravel())
    ap_per_class = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        ap_per_class.append(average_precision_score(y_true[:, j], y_score[:, j]))
    macro = float(np.mean(ap_per_class)) if ap_per_class else float("nan")
    return float(micro), macro

def evaluate_split(cfg, ckpt_path: str, split: str = "test", save_preds: bool = True, calibration=None):
    device = get_device(cfg.get("device","auto"))
    split_dir = os.path.join(cfg["data"]["split_dir"], split)
    ds = PatientSequenceDataset(split_dir, dx_file=cfg["data"]["dx_file"], proc_file=cfg["data"]["proc_file"], treat_file=cfg["data"]["treat_file"],
                                demo_file=cfg["data"]["demo_file"], severity_file=cfg["data"]["severity_file"], notes_file=cfg["data"]["notes_file"],
                                max_visits=cfg["data"]["max_visits"], min_visits=cfg["data"]["min_visits"])
    pin_memory = device.type in ("cuda", "mps")
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["optim"]["batch_size_patients"], shuffle=False, num_workers=0, collate_fn=pad_visits, pin_memory=pin_memory)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    vocabs = load_multi_vocab(os.path.join(cfg["data"]["split_dir"], "train", cfg["data"]["vocab_file"]))
    vocab_sizes = {a: vocabs[a].size() for a in ASPECTS}
    model = Med2VecPlus(vocab_sizes=vocab_sizes, d_embed=ckpt["cfg"]["model"]["d_embed"], d_text=ckpt["cfg"]["model"]["d_text"], d_fuse=ckpt["cfg"]["model"]["d_fuse"],
                        n_heads=ckpt["cfg"]["model"]["n_heads"], dropout=ckpt["cfg"]["model"]["dropout"], text_enabled=ckpt["cfg"]["model"]["text_enabled"],
                        temporal=ckpt["cfg"]["model"]["temporal"], med2vec_compat=ckpt["cfg"]["model"]["med2vec_compat"], share_code_embeddings=ckpt["cfg"]["model"]["share_code_embeddings"],
                        d_demo=(loader.dataset[0]["demo"].shape[1] if "demo" in loader.dataset[0] else 0)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    ytrue_by_aspect = {a: [] for a in ASPECTS}
    yscore_by_aspect = {a: [] for a in ASPECTS}
    risk_true, risk_score = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split}]"):
            for k in ["time_mask", "severity"]:
                if batch.get(k, None) is not None and hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device, non_blocking=True)
            out = model(batch)
            mask = batch["time_mask"].to(device)
            for a in ASPECTS:
                logits = out["next_logits"][a]  # B,T,V (torch)
                B, T, V = logits.shape
                target = np.zeros((B, T, V), dtype=np.float32)
                for b in range(B):
                    for t in range(T):
                        idxs = batch["next_codes"][a][b][t]
                        for j in idxs:
                            target[b, t, j] = 1.0
                m = mask.cpu().numpy().astype(bool)

                if calibration is not None and calibration.get("method") in ("temp", "platt"):
                    L = logits.detach().cpu().numpy()
                    Lc = apply_calibration_to_logits(calibration, a, L)
                    probs = 1.0 / (1.0 + np.exp(-Lc))
                elif calibration is not None and calibration.get("method") == "isotonic":
                    L = logits.detach().cpu().numpy()
                    p0 = 1.0 / (1.0 + np.exp(-L))
                    probs = apply_isotonic_to_probs(calibration, a, p0)
                else:
                    probs = torch.sigmoid(logits).detach().cpu().numpy()

                ytrue_by_aspect[a].append(target[m]); yscore_by_aspect[a].append(probs[m])

            if batch.get("severity", None) is not None:
                risk_true.append(batch["severity"][mask].cpu().numpy())
                risk_score.append(torch.sigmoid(out["risk_logit"])[mask].cpu().numpy())

    for a in ASPECTS:
        if len(ytrue_by_aspect[a]) > 0:
            ytrue_by_aspect[a] = np.concatenate(ytrue_by_aspect[a], axis=0)
            yscore_by_aspect[a] = np.concatenate(yscore_by_aspect[a], axis=0)
    risk_true = np.concatenate(risk_true, axis=0) if risk_true else None
    risk_score = np.concatenate(risk_score, axis=0) if risk_score else None

    results = {"aspect": {}, "risk": {}}
    ks = cfg["eval"]["recall_ks"]
    for a in ASPECTS:
        if a not in ytrue_by_aspect or ytrue_by_aspect[a].size == 0:
            continue
        y_t, y_s = ytrue_by_aspect[a], yscore_by_aspect[a]
        topk = topk_metrics(y_t, y_s, ks)
        micro, macro = micro_macro_auprc(y_t, y_s)
        eps = 1e-7
        logloss = float(-(y_t*np.log(y_s+eps) + (1-y_t)*np.log(1-y_s+eps)).mean())
        results["aspect"][a] = {"topk": topk, "micro_auprc": micro, "macro_auprc": macro, "logloss": logloss}
    if risk_true is not None and risk_true.size > 0:
        try:
            auroc = roc_auc_score(risk_true, risk_score)
        except Exception:
            auroc = float("nan")
        auprc = average_precision_score(risk_true, risk_score)
        brier = brier_score_loss(risk_true, risk_score)
        results["risk"] = {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier)}

    if cfg["eval"]["save_preds"]:
        out_dir = os.path.join(cfg["logging"]["ckpt_dir"], "preds"); os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results

def ndcg_at_k_binary(y_true_bin: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, y_score.shape[1])
    idx = np.argpartition(-y_score, kth=k-1, axis=1)[:, :k]
    part_scores = np.take_along_axis(y_score, idx, axis=1)
    order = np.argsort(-part_scores, axis=1)
    topk_idx = np.take_along_axis(idx, order, axis=1)
    gains = np.take_along_axis(y_true_bin, topk_idx, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    dcg = (gains * discounts).sum(axis=1)
    ideal = -np.sort(-y_true_bin, axis=1)[:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    ndcg = np.where(idcg > 0, dcg / np.clip(idcg, 1e-12, None), 0.0).mean()
    return float(ndcg)

def _fit_predict_ovr_with_constant_handling(base_clf, Xtr, Ytr, Xte, min_pos=3):
    n = Ytr.shape[0]
    pos = Ytr.sum(axis=0).astype(int)
    all_pos = (pos == n)
    all_neg = (pos == 0)
    rare    = (pos < min_pos)

    keep_mask = (~all_pos) & (~all_neg) & (~rare)
    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        prevalence = (pos / np.maximum(1, n)).astype(float)
        P_full = np.tile(prevalence, (Xte.shape[0], 1))
        return P_full, keep_mask

    ovr = OneVsRestClassifier(base_clf, n_jobs=-1)
    try:
        ovr.fit(Xtr, Ytr[:, keep_idx])
    except ValueError:
        keep_mask = (~all_pos) & (~all_neg) & (pos >= 2)
        keep_idx = np.where(keep_mask)[0]
        if keep_idx.size == 0:
            prevalence = (pos / np.maximum(1, n)).astype(float)
            P_full = np.tile(prevalence, (Xte.shape[0], 1))
            return P_full, keep_mask
        ovr.fit(Xtr, Ytr[:, keep_idx])

    P_kept = ovr.predict_proba(Xte)
    if isinstance(P_kept, list):
        P_kept = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in P_kept])

    P_full = np.zeros((Xte.shape[0], Ytr.shape[1]), dtype=float)
    P_full[:, keep_idx] = P_kept
    if all_pos.any():
        P_full[:, np.where(all_pos)[0]] = 1.0
    if all_neg.any():
        P_full[:, np.where(all_neg)[0]] = 0.0
    if rare.any():
        P_full[:, np.where(rare)[0]] = (pos[rare] / np.maximum(1, n)).astype(float)
    return P_full, keep_mask

def train_sklearn_baselines(cfg, split_dir, aspects=ASPECTS, top_codes=250):
    import os, pickle as pkl, torch
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import average_precision_score
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise ImportError("xgboost is required for baselines. Install with `pip install xgboost`.") from e

    results = {}
    seed = cfg.get("seed", 707)
    sev_file = cfg["data"].get("severity_file")
    include_risk = bool(sev_file)

    def load_split(s):
        ds = PatientSequenceDataset(
            os.path.join(split_dir, s),
            dx_file=cfg["data"]["dx_file"],
            proc_file=cfg["data"]["proc_file"],
            treat_file=cfg["data"]["treat_file"],
            demo_file=cfg["data"]["demo_file"],
            severity_file=cfg["data"]["severity_file"],
            notes_file=None,
            max_visits=cfg["data"]["max_visits"],
            min_visits=cfg["data"]["min_visits"],
        )
        return ds

    to_eval = list(aspects)
    if include_risk:
        to_eval.append("risk")

    # lengths for concatenated bag-of-codes
    with open(os.path.join(split_dir, "train", cfg["data"]["vocab_file"]), "rb") as f:
        vocabs_all = pkl.load(f)
    dim_dx = len(vocabs_all["dx"]["id2code"])
    dim_pr = len(vocabs_all["proc"]["id2code"])
    dim_tr = len(vocabs_all["treat"]["id2code"])
    dim_total = dim_dx + dim_pr + dim_tr

    def build_xy(ds, aspect):
        X, Y, R = [], [], []
        for i in range(len(ds)):
            item = ds[i]
            T = len(item["dx"])
            for t in range(T - 1):
                v = np.zeros(dim_total, dtype=np.float32)
                off = 0
                for a in ["dx", "proc", "treat"]:
                    for c in item[a][t]:
                        v[off + c] = 1.0
                    off += {"dx": dim_dx, "proc": dim_pr, "treat": dim_tr}[a]
                if aspect in ["dx", "proc", "treat"]:
                    nxt = item[aspect][t + 1]
                    if (len(nxt) == 1 and nxt[0] == -1) or (len(nxt) == 0):
                        continue
                    X.append(v)
                    Y.append(nxt)
                else:
                    if "severity" in item and (t + 1) < len(item["severity"]):
                        X.append(v)
                        R.append(int(item["severity"][t + 1][0]))
        if aspect in ["dx", "proc", "treat"]:
            X = np.stack(X, axis=0) if X else np.zeros((0, dim_total), dtype=np.float32)
            return X, Y, None
        else:
            X = np.stack(X, axis=0) if X else np.zeros((0, dim_total), dtype=np.float32)
            return X, None, np.array(R, dtype=np.int32) if R else np.zeros((0,), dtype=np.int32)

    for aspect in to_eval:
        tr, va, te = (load_split("train"), load_split("valid"), load_split("test"))

        if aspect in ["dx", "proc", "treat"]:
            with open(os.path.join(split_dir, "train", cfg["data"]["vocab_file"]), "rb") as f:
                vocabs = pkl.load(f)
            V = len(vocabs[aspect]["id2code"])

            Xtr, Ytr, _ = build_xy(tr, aspect)
            Xte, Yte, _ = build_xy(te, aspect)

            freq = np.zeros(V, dtype=np.int64)
            for y in Ytr:
                for j in y:
                    freq[j] += 1
            pos_idx = np.where(freq > 0)[0]
            if pos_idx.size == 0 or Xtr.shape[0] == 0 or Xte.shape[0] == 0:
                results[aspect] = {"skipped": "no samples or no positive labels in training"}
                continue

            order = np.argsort(-freq[pos_idx])
            keep = pos_idx[order][:min(top_codes, pos_idx.size)]
            idmap = {j: i for i, j in enumerate(keep)}
            mlb = MultiLabelBinarizer(classes=keep)
            Ytr_b = mlb.fit_transform([[j for j in y if j in idmap] for y in Ytr])
            Yte_b = mlb.transform([[j for j in y if j in idmap] for y in Yte])

            xgb_tree_method = os.environ.get("XGB_TREE_METHOD", "hist")
            models = {
                "rf":  RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed, class_weight="balanced"),
                "mlp": MLPClassifier(hidden_layer_sizes=(256,),
                                     max_iter=200,
                                     random_state=seed,
                                     early_stopping=False,
                                     n_iter_no_change=10,
                                     validation_fraction=0.2),
                "xgb": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     tree_method=xgb_tree_method, n_jobs=-1,
                                     objective="binary:logistic", eval_metric="aucpr",
                                     random_state=seed),
            }

            res_aspect = {}
            for name, model in models.items():
                try:
                    P, _ = _fit_predict_ovr_with_constant_handling(model, Xtr, Ytr_b, Xte, min_pos=3)
                except Exception:
                    if name == "xgb" and xgb_tree_method == "gpu_hist":
                        model.set_params(tree_method="hist")
                        P, _ = _fit_predict_ovr_with_constant_handling(model, Xtr, Ytr_b, Xte, min_pos=3)
                    else:
                        raise
                micro = average_precision_score(Yte_b.ravel(), P.ravel())
                ks = cfg["eval"]["recall_ks"]
                topk = {}
                for k in ks:
                    idx = np.argpartition(-P, kth=min(k - 1, P.shape[1] - 1), axis=1)[:, :k]
                    rel = Yte_b[np.arange(len(Yte_b))[:, None], idx]
                    topk[k] = {
                        "recall": float((rel.sum(axis=1) / np.clip(Yte_b.sum(axis=1), 1, None)).mean()),
                        "precision": float(rel.mean()),
                        "ndcg": ndcg_at_k_binary(Yte_b, P, k),
                    }
                res_aspect[name] = {"micro_auprc": float(micro), "topk_limited": topk}
            results[aspect] = res_aspect

        else:
            Xtr, _, Rtr = build_xy(tr, aspect)
            Xte, _, Rte = build_xy(te, aspect)
            if Rtr.size == 0 or np.unique(Rtr).size < 2 or Xtr.shape[0] == 0 or Xte.shape[0] == 0:
                results[aspect] = {"skipped": "severity labels missing or single-class in training"}
                continue

            xgb_tree_method = os.environ.get("XGB_TREE_METHOD", "hist")
            models = {
                "rf":  RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed, class_weight="balanced"),
                "gb":  GradientBoostingClassifier(random_state=seed),
                "mlp": MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=seed, early_stopping=False),
                "xgb": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     tree_method=xgb_tree_method, n_jobs=-1,
                                     objective="binary:logistic", eval_metric="aucpr",
                                     random_state=seed),
            }
            risk_res = {}
            for name, model in models.items():
                try:
                    model.fit(Xtr, Rtr)
                except Exception:
                    if name == "xgb" and xgb_tree_method == "gpu_hist":
                        model.set_params(tree_method="hist")
                        model.fit(Xtr, Rtr)
                    else:
                        raise
                proba = model.predict_proba(Xte)
                if proba.ndim == 1:
                    P = proba
                elif proba.shape[1] >= 2:
                    P = proba[:, 1]
                else:
                    P = np.full(Xte.shape[0], fill_value=float(Rtr.mean()))
                auprc = average_precision_score(Rte, P)
                try:
                    auroc = roc_auc_score(Rte, P)
                except Exception:
                    auroc = float("nan")
                brier = brier_score_loss(Rte, P)
                risk_res[name] = {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier)}
            results[aspect] = risk_res

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train","valid","test"])
    parser.add_argument("--run_sklearn_baselines", action="store_true")
    # NEW: optional calibration file
    parser.add_argument("--calibration", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 17))

    calib = load_calibrator(args.calibration) if args.calibration else None
    res = evaluate_split(cfg, ckpt_path=args.ckpt, split=args.split, save_preds=cfg["eval"]["save_preds"], calibration=calib)
    if args.run_sklearn_baselines:
        base = train_sklearn_baselines(cfg, cfg["data"]["split_dir"], aspects=ASPECTS, top_codes=cfg["eval"]["baseline_top_codes"])
        print("\n[Sklearn baselines]")
        print(json.dumps(base, indent=2))

if __name__ == "__main__":
    main()