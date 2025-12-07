from typing import List, Dict
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HFSectionTextEncoder(nn.Module):
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", d_out: int = 128, max_length: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hid = self.encoder.config.hidden_size
        self.proj = nn.Linear(hid, d_out)
        self.max_length = max_length
        self.layernorm = nn.LayerNorm(d_out)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        toks = {k: v.to(next(self.parameters()).device) for k, v in toks.items()}
        out = self.encoder(**toks)
        cls = out.last_hidden_state[:, 0]
        return cls

    def forward(self, sections: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        outs = {}
        for k, texts in sections.items():
            if len(texts) == 0:
                outs[k] = torch.zeros(0, self.proj.out_features, device=next(self.parameters()).device)
                continue
            h = self._encode_texts(texts)
            z = self.layernorm(self.proj(h))
            outs[k] = z
        return outs

class GRUTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 50000, d_out: int = 128, max_length: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_out)
        self.gru = nn.GRU(d_out, d_out, batch_first=True)
        self.layernorm = nn.LayerNorm(d_out)
        self.max_length = max_length

    def _simple_tokenize(self, s: str):
        toks = s.strip().split()
        ids = [hash(t) % self.vocab_size for t in toks[:self.max_length]]
        if not ids: ids = [0]
        return ids

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        seqs = [torch.tensor(self._simple_tokenize(t), dtype=torch.long) for t in texts]
        L = max(x.numel() for x in seqs) if seqs else 1
        B = len(seqs)
        pad = torch.zeros(B, L, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :s.numel()] = s
        pad = pad.to(device)
        H = self.emb(pad)
        _, h = self.gru(H)
        return h.squeeze(0)

    def forward(self, sections: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        outs = {}
        for k, texts in sections.items():
            if len(texts) == 0:
                outs[k] = torch.zeros(0, self.emb.embedding_dim, device=next(self.parameters()).device)
                continue
            h = self._encode_texts(texts)
            z = self.layernorm(h)
            outs[k] = z
        return outs
