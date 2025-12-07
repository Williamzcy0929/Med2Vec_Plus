from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

class VisitEncoder(nn.Module):
    def __init__(self, d_embed: int):
        super().__init__()
        self.proj = nn.Linear(d_embed, d_embed)
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, visit_code_indices: List[List[int]], embed_module) -> torch.Tensor:
        outs = []
        device = next(self.parameters()).device
        for codes in visit_code_indices:
            if len(codes) == 0:
                outs.append(torch.zeros(embed_module.weight.shape[1], device=device))
                continue
            idx = torch.tensor(codes, dtype=torch.long, device=device)
            E = embed_module(idx)
            pooled = E.mean(0)
            z = self.proj(F.relu(pooled))
            z = self.norm(z)
            outs.append(z)
        return torch.stack(outs, dim=0)
