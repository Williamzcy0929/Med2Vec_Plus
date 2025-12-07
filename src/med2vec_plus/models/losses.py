from typing import Dict, List
import torch, torch.nn.functional as F

def bce_next_visit_loss(logits_dict: Dict[str, torch.Tensor], next_codes: Dict[str, List[List[List[int]]]]):
    loss = 0.0
    total = 0
    for a, logits in logits_dict.items():
        B, T, V = logits.shape
        target = torch.zeros_like(logits)
        for b in range(B):
            for t in range(T):
                idxs = next_codes[a][b][t]
                if idxs:
                    target[b, t, torch.tensor(idxs, dtype=torch.long, device=logits.device)] = 1.0
        loss += F.binary_cross_entropy_with_logits(logits, target)
        total += 1
    return loss / max(total, 1)

def severity_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    if labels is None:
        return torch.tensor(0.0, device=logits.device)
    logits = logits[mask]
    y = labels[mask]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, y)

def intra_visit_cooccur_loss(visit_embeddings: Dict[str, torch.Tensor], code_embeddings: Dict[str, torch.Tensor] = None):
    loss = 1e-4 * sum((z ** 2).mean() for z in visit_embeddings.values())
    return loss

def text_align_infonce(z: torch.Tensor, s: torch.Tensor, temperature: float = 0.07):
    if z is None or s is None:
        return torch.tensor(0.0, device=z.device if z is not None else s.device)
    B, T, D = z.shape
    zf = torch.nn.functional.normalize(z.reshape(B * T, D), dim=-1)
    sf = torch.nn.functional.normalize(s.reshape(B * T, D), dim=-1)
    logits = (zf @ sf.t()) / temperature
    target = torch.arange(B * T, device=zf.device)
    return torch.nn.functional.cross_entropy(logits, target)
