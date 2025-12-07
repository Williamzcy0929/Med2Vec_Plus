import torch, torch.nn as nn

class CrossAttentionGate(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, z: torch.Tensor, s: torch.Tensor):
        B, T, d = z.shape
        Q = self.q_proj(z).reshape(B * T, 1, d)
        K = self.k_proj(s).reshape(B * T, 1, d)
        V = self.v_proj(s).reshape(B * T, 1, d)
        out, _ = self.attn(Q, K, V, need_weights=False)
        out = out.reshape(B, T, d)
        alpha = torch.sigmoid(self.gate(torch.cat([z, s], dim=-1)))
        return self.norm(z + alpha * out)
