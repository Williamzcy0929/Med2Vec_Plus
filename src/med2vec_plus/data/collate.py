from typing import Dict, Any, List
import torch
from .vocab import ASPECTS

def pad_visits(batch: List[Dict[str, Any]]):
    B = len(batch)
    T = max(len(b["dx"]) for b in batch)
    time_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        time_mask[i, :len(b["dx"])] = True

    codes = {a: [] for a in ASPECTS}
    next_codes = {a: [] for a in ASPECTS}
    for a in ASPECTS:
        for b in batch:
            seq = b[a]
            padded = seq + ([[]] * (T - len(seq)))
            codes[a].append(padded)
            targets = seq[1:] + [[]]
            targets = targets + ([[]] * (T - len(seq)))
            next_codes[a].append(targets)

    demo = None
    if "demo" in batch[0]:
        p = batch[0]["demo"].shape[1]
        demo = torch.zeros(B, T, p, dtype=torch.float32)
        for i, b in enumerate(batch):
            dv = torch.tensor(b["demo"], dtype=torch.float32)
            demo[i, :dv.shape[0]] = dv

    severity = None
    if "severity" in batch[0]:
        severity = torch.zeros(B, T, dtype=torch.float32)
        for i, b in enumerate(batch):
            sv = [x[0] if isinstance(x, list) else int(x) for x in b["severity"]]
            sv_shift = sv[1:] + [0]
            severity[i, :len(sv_shift)] = torch.tensor(sv_shift, dtype=torch.float32)

    notes = None
    if "notes" in batch[0]:
        notes = {a: [[""] * T for _ in range(B)] for a in ["dx", "proc", "treat", "ap"]}
        for i, b in enumerate(batch):
            vlist = b["notes"]
            for t in range(min(len(vlist), T)):
                r = vlist[t]
                notes["dx"][i][t] = r.get("dx", r.get("dx_text", ""))
                notes["proc"][i][t] = r.get("proc", r.get("proc_text", ""))
                notes["treat"][i][t] = r.get("treat", r.get("treat_text", ""))
                notes["ap"][i][t] = r.get("ap", r.get("ap_text", ""))

    return {"codes": codes, "next_codes": next_codes, "time_mask": time_mask,
            "demo": demo, "severity": severity, "notes": notes}
