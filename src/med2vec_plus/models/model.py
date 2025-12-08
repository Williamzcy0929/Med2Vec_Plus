from typing import Dict, List
import torch, torch.nn as nn
from .code_embed import NonNegEmbedding
from .visit_encoder import VisitEncoder
from .text_encoder import HFSectionTextEncoder, GRUTextEncoder
from .cross_attention import CrossAttentionGate
from .temporal import TemporalEncoder
from .fusion import AspectFusion
from .heads import MultiLabelHead, RiskHead
from .losses import bce_next_visit_loss, severity_loss, intra_visit_cooccur_loss, text_align_infonce

ASPECTS = ["dx", "proc", "treat"]

class Med2VecPlus(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], d_embed: int = 128, d_text: int = 128, d_fuse: int = 256,
                 n_heads: int = 4, dropout: float = 0.1, text_enabled: bool = True, temporal: str = "gru",
                 med2vec_compat: bool = False, share_code_embeddings: bool = False, d_demo: int = 0):
        super().__init__()
        self.med2vec_compat = med2vec_compat
        self.text_enabled = text_enabled and (not med2vec_compat)

        if share_code_embeddings:
            total_V = sum(vocab_sizes.values())
            self.shared_emb = NonNegEmbedding(total_V, d_embed)
            offsets, cur = {}, 0
            for a in ASPECTS:
                offsets[a] = (cur, cur + vocab_sizes[a])
                cur += vocab_sizes[a]
            self._offsets = offsets
            self.embeds = None
        else:
            self.shared_emb = None
            self.embeds = nn.ModuleDict({a: NonNegEmbedding(vocab_sizes[a], d_embed) for a in ASPECTS})

        self.visit_enc = nn.ModuleDict({a: VisitEncoder(d_embed) for a in ASPECTS})

        if self.text_enabled:
            try:
                self.text_enc = HFSectionTextEncoder(d_out=d_text)
            except Exception:
                self.text_enc = GRUTextEncoder(d_out=d_text)
            self.cross_attn = nn.ModuleDict({a: CrossAttentionGate(d_embed, n_heads=n_heads, dropout=dropout) for a in ASPECTS})
            self.align_proj = nn.ModuleDict({a: nn.Linear(d_text, d_embed) for a in ASPECTS})
        else:
            self.text_enc = None
            self.cross_attn = None
            self.align_proj = None

        if not med2vec_compat:
            self.temporal = nn.ModuleDict(
                {
                    a: TemporalEncoder(
                        d_model=d_embed,
                        mode=temporal,
                        n_layers=1,
                        n_heads=n_heads,
                        dropout=dropout,
                        bidirectional=False
                    )
                    for a in ASPECTS
                }
            )
        else:
            self.temporal = None

        self.fusion = AspectFusion(d=d_embed, d_demo=d_demo, d_fuse=d_fuse)
        self.next_heads = nn.ModuleDict({a: MultiLabelHead(d_fuse, vocab_sizes[a]) for a in ASPECTS})
        self.risk_head = RiskHead(d_fuse)

    def _embed_visit_list(self, aspect: str, visit_codes: List[List[int]]):
        if self.shared_emb is not None:
            off0, off1 = self._offsets[aspect]
            z_list = []
            for codes in visit_codes:
                if codes:
                    idx = torch.tensor([c + off0 for c in codes], dtype=torch.long, device=next(self.parameters()).device)
                    E = self.shared_emb(idx)
                    z_list.append(E.mean(0))
                else:
                    z_list.append(torch.zeros(self.shared_emb.weight.shape[1], device=next(self.parameters()).device))
            return torch.stack(z_list, dim=0)
        else:
            return self.visit_enc[aspect](visit_codes, self.embeds[aspect])

    def forward(self, batch: Dict[str, any]):
        codes = batch["codes"]
        time_mask = batch["time_mask"].to(next(self.parameters()).device)
        demo = batch["demo"].to(next(self.parameters()).device) if batch.get("demo", None) is not None else None
        notes = batch.get("notes", None)
        B, T = time_mask.shape

        z = {}
        for a in ASPECTS:
            z_list = []
            for b in range(B):
                z_b = self._embed_visit_list(a, codes[a][b])
                z_list.append(z_b)
            z[a] = torch.stack(z_list, dim=0)
        visit_embeds_for_aux = z.copy()

        s_proj = {}
        if self.text_enabled and notes is not None:
            sections = {k: [] for k in ["dx", "proc", "treat", "ap"]}
            for a in sections.keys():
                flat = []
                for b in range(B):
                    flat.extend(notes[a][b][:T])
                sections[a] = flat
            enc = self.text_enc({k: sections[k] for k in sections})
            for a in ["dx", "proc", "treat"]:
                vec = enc[a]
                vec = vec.reshape(B, T, -1)
                vec = self.align_proj[a](vec) if self.align_proj is not None else vec
                s_proj[a] = vec
                z[a] = self.cross_attn[a](z[a], vec)

        if self.temporal is not None:
            for a in ASPECTS:
                z[a] = self.temporal[a](z[a], time_mask)

        h = self.fusion(z["dx"], z["proc"], z["treat"], demo=demo)
        next_logits = {a: self.next_heads[a](h) for a in ASPECTS}
        risk_logit = self.risk_head(h)

        return {"next_logits": next_logits, "risk_logit": risk_logit, "visit_embeddings": visit_embeds_for_aux,
                "text_proj": s_proj if self.text_enabled else None}

    def compute_losses(self, batch, outputs, lambda_intra=0.2, lambda_text=0.1, lambda_sup=1.0):
        L_next = bce_next_visit_loss(outputs["next_logits"], batch["next_codes"])
        L_sup = severity_loss(outputs["risk_logit"], batch.get("severity", None).to(outputs["risk_logit"].device) if batch.get("severity", None) is not None else None, batch["time_mask"].to(outputs["risk_logit"].device))
        L_intra = intra_visit_cooccur_loss(outputs["visit_embeddings"])
        L_text = torch.tensor(0.0, device=outputs["risk_logit"].device)
        if outputs.get("text_proj", None) is not None and "dx" in outputs["text_proj"]:
            L_text = text_align_infonce(outputs["visit_embeddings"]["dx"], outputs["text_proj"]["dx"])
        loss = L_next + lambda_intra * L_intra + lambda_text * L_text + lambda_sup * L_sup
        return {"total": loss, "next": L_next, "intra": L_intra, "text": L_text, "sup": L_sup}
