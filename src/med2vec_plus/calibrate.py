# src/med2vec_plus/calibrate.py
import os, argparse, yaml, json, pickle, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from .utils import set_seed, get_device, load_checkpoint
from .data.dataset import PatientSequenceDataset
from .data.collate import pad_visits
from .data.vocab import load_multi_vocab, ASPECTS
from .models.model import Med2VecPlus

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _gather_logits_targets(cfg, ckpt_path: str, split: str = "valid"):
    device = get_device(cfg.get("device","auto"))
    split_dir = os.path.join(cfg["data"]["split_dir"], split)
    ds = PatientSequenceDataset(split_dir,
                                dx_file=cfg["data"]["dx_file"], proc_file=cfg["data"]["proc_file"], treat_file=cfg["data"]["treat_file"],
                                demo_file=cfg["data"]["demo_file"], severity_file=cfg["data"]["severity_file"], notes_file=cfg["data"]["notes_file"],
                                max_visits=cfg["data"]["max_visits"], min_visits=cfg["data"]["min_visits"])
    pin_memory = device.type in ("cuda", "mps")
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["optim"]["batch_size_patients"],
                                         shuffle=False, num_workers=0, collate_fn=pad_visits, pin_memory=pin_memory)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    vocabs = load_multi_vocab(os.path.join(cfg["data"]["split_dir"], "train", cfg["data"]["vocab_file"]))
    vocab_sizes = {a: vocabs[a].size() for a in ASPECTS}
    model = Med2VecPlus(vocab_sizes=vocab_sizes,
                        d_embed=ckpt["cfg"]["model"]["d_embed"], d_text=ckpt["cfg"]["model"]["d_text"], d_fuse=ckpt["cfg"]["model"]["d_fuse"],
                        n_heads=ckpt["cfg"]["model"]["n_heads"], dropout=ckpt["cfg"]["model"]["dropout"],
                        text_enabled=ckpt["cfg"]["model"]["text_enabled"], temporal=ckpt["cfg"]["model"]["temporal"],
                        med2vec_compat=ckpt["cfg"]["model"]["med2vec_compat"], share_code_embeddings=ckpt["cfg"]["model"]["share_code_embeddings"],
                        d_demo=(loader.dataset[0]["demo"].shape[1] if "demo" in loader.dataset[0] else 0)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    out_logits = {a: [] for a in ASPECTS}
    out_tgts   = {a: [] for a in ASPECTS}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collect [{split}]"):
            for k in ["time_mask", "severity"]:
                if batch.get(k, None) is not None and hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device, non_blocking=True)
            outputs = model(batch)
            mask = batch["time_mask"].to(device).cpu().numpy().astype(bool)
            for a in ASPECTS:
                logits = outputs["next_logits"][a].detach().cpu().numpy()  # B,T,V
                B, T, V = logits.shape
                target = np.zeros((B, T, V), dtype=np.float32)
                for b in range(B):
                    for t in range(T):
                        idxs = batch["next_codes"][a][b][t]
                        for j in idxs:
                            target[b, t, j] = 1.0
                out_logits[a].append(logits[mask])
                out_tgts[a].append(target[mask])

    for a in ASPECTS:
        out_logits[a] = np.concatenate(out_logits[a], axis=0) if out_logits[a] else np.zeros((0, 0), dtype=np.float32)
        out_tgts[a]   = np.concatenate(out_tgts[a], axis=0)   if out_tgts[a] else np.zeros((0, 0), dtype=np.float32)
    return out_logits, out_tgts

def _fit_temperature(L: np.ndarray, Y: np.ndarray, iters=500, lr=0.01):
    if L.size == 0:
        return 1.0
    x = torch.from_numpy(L).float()
    y = torch.from_numpy(Y).float()
    logT = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(logT).clamp(min=0.05, max=5.0)
        loss = bce(x / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(logT).clamp(min=0.05, max=5.0).item())
    return T

def _fit_platt_shared(L: np.ndarray, Y: np.ndarray, iters=500, lr=0.01):
    if L.size == 0:
        return 1.0, 0.0
    x = torch.from_numpy(L).float()
    y = torch.from_numpy(Y).float()
    a = torch.nn.Parameter(torch.ones(1))
    b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(a * x + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.item()), float(b.item())

def _fit_isotonic_per_class(P: np.ndarray, Y: np.ndarray, min_pos=10, min_neg=10, max_classes=1000):
    if P.size == 0:
        return {}
    V = P.shape[1]
    pos = Y.sum(axis=0).astype(int)
    neg = (Y.shape[0] - pos).astype(int)
    ok = np.where((pos >= min_pos) & (neg >= min_neg))[0]
    if ok.size == 0:
        return {}
    if ok.size > max_classes:
        order = np.argsort(-pos[ok])
        ok = ok[order[:max_classes]]
    models = {}
    for j in ok:
        ir = IsotonicRegression(out_of_bounds="clip")
        pj = P[:, j]
        yj = Y[:, j]
        try:
            ir.fit(pj, yj)
            # store thresholds to avoid sklearn dependency at inference
            models[int(j)] = {
                "x": ir.X_thresholds_.astype(np.float32),
                "y": ir.y_thresholds_.astype(np.float32),
            }
        except Exception:
            continue
    return models

def save_calibrator(calib, path):
    with open(path, "wb") as f:
        pickle.dump(calib, f)

def load_calibrator(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def apply_calibration_to_logits(calib, aspect: str, logits_np: np.ndarray) -> np.ndarray:
    if calib is None:
        return logits_np
    method = calib.get("method")
    params = calib.get("params", {}).get(aspect, {})
    if method == "temp":
        T = float(params.get("T", 1.0))
        return logits_np / max(T, 1e-6)
    if method == "platt":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        return a * logits_np + b
    return logits_np

def apply_isotonic_to_probs(calib, aspect: str, probs_np: np.ndarray) -> np.ndarray:
    if calib is None:
        return probs_np
    if calib.get("method") != "isotonic":
        return probs_np
    models = calib.get("params", {}).get(aspect, {}).get("iso_models", {})
    if not models:
        return probs_np
    out = probs_np.copy()
    for j_str, xy in models.items():
        j = int(j_str)
        if j >= out.shape[1]:
            continue
        x = np.asarray(xy["x"], dtype=np.float32)
        y = np.asarray(xy["y"], dtype=np.float32)
        out[:, j] = np.interp(out[:, j], x, y, left=y[0], right=y[-1])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="valid", choices=["train","valid","test"])
    ap.add_argument("--method", type=str, required=True, choices=["temp","platt","isotonic"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--min_neg", type=int, default=10)
    ap.add_argument("--max_classes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=707)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    logits_by_a, tgts_by_a = _gather_logits_targets(cfg, args.ckpt, split=args.split)

    calib = {"method": args.method, "params": {}, "meta": {"split": args.split}}
    if args.method == "temp":
        for a in ASPECTS:
            L = logits_by_a[a]
            Y = tgts_by_a[a]
            T = _fit_temperature(L, Y)
            calib["params"][a] = {"T": float(T)}
    elif args.method == "platt":
        for a in ASPECTS:
            L = logits_by_a[a]
            Y = tgts_by_a[a]
            a_hat, b_hat = _fit_platt_shared(L, Y)
            calib["params"][a] = {"a": float(a_hat), "b": float(b_hat)}
    else:
        for a in ASPECTS:
            L = logits_by_a[a]
            Y = tgts_by_a[a]
            P = 1.0 / (1.0 + np.exp(-L))
            models = _fit_isotonic_per_class(P, Y, min_pos=args.min_pos, min_neg=args.min_neg, max_classes=args.max_classes)
            calib["params"][a] = {"iso_models": models}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_calibrator(calib, args.out)
    print(json.dumps({"saved": args.out, "method": args.method, "params": calib["params"]}, indent=2))

if __name__ == "__main__":
    main()