# src/med2vec_plus/models/temporal.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, added to inputs."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class GRUTemporalEncoder(nn.Module):
    """Time encoder with GRU, batch_first, length-masked by time_mask."""
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        hidden = d_model // 2 if bidirectional else d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.out_proj = None
        if bidirectional:
            # bidirectional output dim = hidden*2 = d_model; keep as-is
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], time_mask: [B, T] with 1/True for valid, 0/False for pad
        if time_mask.dtype != torch.bool:
            mask_bool = time_mask > 0
        else:
            mask_bool = time_mask
        lengths = mask_bool.sum(dim=1).clamp(min=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        if self.out_proj is not None:
            out = self.out_proj(out)
        return out  # [B, T, D]


class TransformerTemporalEncoder(nn.Module):
    """Time encoder with TransformerEncoder, masking with src_key_padding_mask."""
    def __init__(self, d_model: int, n_heads: int = 4, dim_ff: int | None = None,
                 num_layers: int = 1, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        dim_ff = dim_ff or (4 * d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], time_mask: [B, T] with 1/True for valid
        if time_mask.dtype != torch.bool:
            valid = time_mask > 0
        else:
            valid = time_mask
        # src_key_padding_mask: True means "to be ignored"
        pad_mask = ~valid
        h = self.pos(x)
        out = self.encoder(h, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return out  # [B, T, D]


class TemporalEncoder(nn.Module):
    """Factory wrapper to unify GRU and Transformer under one interface."""
    def __init__(self, d_model: int, mode: str = "gru",
                 n_layers: int = 1, n_heads: int = 4, dim_ff: int | None = None,
                 dropout: float = 0.1, bidirectional: bool = False, max_len: int = 1024):
        super().__init__()
        mode = (mode or "gru").lower()
        if mode == "transformer":
            self.impl = TransformerTemporalEncoder(
                d_model=d_model, n_heads=n_heads, dim_ff=dim_ff,
                num_layers=n_layers, dropout=dropout, max_len=max_len
            )
        elif mode == "gru":
            self.impl = GRUTemporalEncoder(
                d_model=d_model, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown temporal mode: {mode}")

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        return self.impl(x, time_mask)