import os, argparse, yaml, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import set_seed, get_device, make_writer, save_checkpoint
from .data.dataset import PatientSequenceDataset
from .data.collate import pad_visits
from .data.vocab import load_multi_vocab, ASPECTS
from .models.model import Med2VecPlus

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataloaders(cfg, pin_memory: bool = False):
    split_dir = cfg["data"]["split_dir"]
    max_visits = cfg["data"]["max_visits"]
    min_visits = cfg["data"]["min_visits"]
    ds_kwargs = dict(dx_file=cfg["data"]["dx_file"], proc_file=cfg["data"]["proc_file"], treat_file=cfg["data"]["treat_file"],
                     demo_file=cfg["data"]["demo_file"], severity_file=cfg["data"]["severity_file"], notes_file=cfg["data"]["notes_file"],
                     max_visits=max_visits, min_visits=min_visits)
    train_ds = PatientSequenceDataset(os.path.join(split_dir, "train"), **ds_kwargs)
    valid_ds = PatientSequenceDataset(os.path.join(split_dir, "valid"), **ds_kwargs)
    collate = pad_visits
    train_loader = DataLoader(train_ds, batch_size=cfg["optim"]["batch_size_patients"], shuffle=True, num_workers=0, collate_fn=collate, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["optim"]["batch_size_patients"], shuffle=False, num_workers=0, collate_fn=collate, pin_memory=pin_memory)
    return train_loader, valid_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 17))
    device = get_device(cfg.get("device", "auto"))
    writer = make_writer(cfg["logging"]["log_dir"])

    pin_memory = device.type in ("cuda", "mps")
    train_loader, valid_loader = build_dataloaders(cfg, pin_memory=pin_memory)
    vocab_path = args.vocab or os.path.join(cfg["data"]["split_dir"], "train", cfg["data"]["vocab_file"])
    vocabs = load_multi_vocab(vocab_path)
    vocab_sizes = {a: vocabs[a].size() for a in ASPECTS}

    mcfg = cfg["model"]
    model = Med2VecPlus(vocab_sizes=vocab_sizes, d_embed=mcfg["d_embed"], d_text=mcfg["d_text"], d_fuse=mcfg["d_fuse"],
                        n_heads=mcfg["n_heads"], dropout=mcfg["dropout"], text_enabled=mcfg["text_enabled"],
                        temporal=mcfg["temporal"], med2vec_compat=mcfg["med2vec_compat"], share_code_embeddings=mcfg["share_code_embeddings"],
                        d_demo=(train_loader.dataset[0]["demo"].shape[1] if "demo" in train_loader.dataset[0] else 0)).to(device)

    ocfg = cfg["optim"]
    optim = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], betas=tuple(ocfg["betas"]), weight_decay=ocfg["weight_decay"])

    best_val = float("inf")
    ckpt_dir = args.save or cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    warmup_epochs = ocfg.get("warmup_epochs", 1)
    for epoch in range(cfg["optim"]["max_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['optim']['max_epochs']} [train]")
        stats = {"loss": 0.0, "next": 0.0, "intra": 0.0, "text": 0.0, "sup": 0.0, "n": 0}
        for batch in pbar:
            text_enabled_original = model.text_enabled
            if epoch < warmup_epochs:
                model.text_enabled = False
            for k in ["time_mask", "severity"]:
                if batch.get(k, None) is not None and hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device, non_blocking=True)
            outputs = model(batch)
            losses = model.compute_losses(batch, outputs, lambda_intra=cfg["loss"]["lambda_intra"], lambda_text=cfg["loss"]["lambda_text"], lambda_sup=cfg["loss"]["lambda_sup"])
            loss = losses["total"]
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
            optim.step()
            model.text_enabled = text_enabled_original
            stats["loss"] += float(loss.item())
            for k in ["next", "intra", "text", "sup"]:
                stats[k] += float(losses[k].item()) if hasattr(losses[k], "item") else float(losses[k])
            stats["n"] += 1
            pbar.set_postfix({k: stats[k]/max(1,stats["n"]) for k in ["loss","next","sup"]})

        for k in ["loss", "next", "intra", "text", "sup"]:
            writer.add_scalar(f"train/{k}", stats[k] / max(1, stats["n"]), epoch)

        model.eval()
        with torch.no_grad():
            vstats = {"loss": 0.0, "n": 0}
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1} [valid]"):
                for k in ["time_mask", "severity"]:
                    if batch.get(k, None) is not None and hasattr(batch[k], "to"):
                        batch[k] = batch[k].to(device, non_blocking=True)
                outputs = model(batch)
                losses = model.compute_losses(batch, outputs, lambda_intra=cfg["loss"]["lambda_intra"], lambda_text=cfg["loss"]["lambda_text"], lambda_sup=cfg["loss"]["lambda_sup"])
                vstats["loss"] += float(losses["total"].item())
                vstats["n"] += 1
            vloss = vstats["loss"] / max(1, vstats["n"])
            writer.add_scalar("valid/loss", vloss, epoch)
            if vloss < best_val:
                best_val = vloss
                save_checkpoint({"cfg": cfg, "state_dict": model.state_dict(), "epoch": epoch, "best_val": best_val, "vocab_sizes": vocab_sizes}, os.path.join(ckpt_dir, "best.pt"))
    print("Training finished. Best valid loss:", best_val)

if __name__ == "__main__":
    main()
