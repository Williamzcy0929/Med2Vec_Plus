import os, random, numpy as np, torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

def set_seed(seed: int = 17):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(pref: str = "auto"):
    if pref in ("cuda","gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_writer(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def masked_mean(t, mask, dim):
    denom = (mask.sum(dim=dim, keepdim=True).clamp(min=1.0))
    return (t * mask.unsqueeze(-1)).sum(dim=dim) / denom

def save_checkpoint(state: Dict[str, Any], ckpt_path: str):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)

def load_checkpoint(ckpt_path: str, map_location=None):
    return torch.load(ckpt_path, map_location=map_location)
