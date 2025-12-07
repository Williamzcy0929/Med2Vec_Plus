from typing import List, Dict, Any, Optional
import pickle, json, numpy as np, os
from torch.utils.data import Dataset
from .vocab import ASPECTS

def _load_seq_pickle(path: str) -> List[List[int]]:
    with open(path, "rb") as f:
        return pickle.load(f)

def _split_by_patient(seqs: List[List[int]]) -> List[List[List[int]]]:
    patients, cur = [], []
    for v in seqs:
        if len(v) == 1 and v[0] == -1:
            if cur:
                patients.append(cur); cur = []
        else:
            cur.append(v)
    if cur: patients.append(cur)
    return patients

def _load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

class PatientSequenceDataset(Dataset):
    def __init__(self, split_dir: str,
                 dx_file: str = "dx.seqs.pkl",
                 proc_file: str = "proc.seqs.pkl",
                 treat_file: str = "treat.seqs.pkl",
                 demo_file: Optional[str] = "demo.npy",
                 severity_file: Optional[str] = "severity.seqs.pkl",
                 notes_file: Optional[str] = "notes.jsonl",
                 max_visits: Optional[int] = None,
                 min_visits: int = 2):
        self.split_dir = split_dir
        self.paths = {
            "dx": os.path.join(split_dir, dx_file),
            "proc": os.path.join(split_dir, proc_file),
            "treat": os.path.join(split_dir, treat_file),
        }
        self.notes_path = None if notes_file is None else os.path.join(split_dir, notes_file)
        self.severity_path = None if severity_file is None else os.path.join(split_dir, severity_file)
        self.demo_path = None if demo_file is None else os.path.join(split_dir, demo_file)

        self.patients = {a: _split_by_patient(_load_seq_pickle(self.paths[a])) for a in ASPECTS}
        n = len(self.patients["dx"])
        assert all(len(self.patients[a]) == n for a in ASPECTS)

        self.notes = None
        if self.notes_path and os.path.exists(self.notes_path):
            self.notes = _load_jsonl(self.notes_path)

        self.severity = None
        if self.severity_path and os.path.exists(self.severity_path):
            self.severity = _split_by_patient(_load_seq_pickle(self.severity_path))

        self.demo = None
        if self.demo_path and os.path.exists(self.demo_path):
            self.demo = np.load(self.demo_path)

        self.idxs = []
        if self.demo is not None:
            demo_offset = 0
        for pid in range(n):
            T = len(self.patients["dx"][pid])
            if T < min_visits:
                if self.demo is not None:
                    demo_offset += T + 1
                continue
            if max_visits is not None and T > max_visits:
                for a in ASPECTS:
                    self.patients[a][pid] = self.patients[a][pid][-max_visits:]
                if self.severity is not None:
                    self.severity[pid] = self.severity[pid][-max_visits:]
                T = len(self.patients["dx"][pid])
            self.idxs.append(pid)
            if self.demo is not None:
                demo_offset += T + 1

        self._demo_slices = None
        if self.demo is not None:
            self._demo_slices = []
            cursor = 0
            for pid in range(n):
                T = len(self.patients["dx"][pid])
                if T < min_visits:
                    cursor += T + 1
                    continue
                self._demo_slices.append((cursor, cursor + T))
                cursor += T + 1

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        pid = self.idxs[i]
        item = {"pid": pid}
        for a in ASPECTS:
            item[a] = self.patients[a][pid]
        if self.severity is not None:
            item["severity"] = self.severity[pid]
        if self._demo_slices is not None:
            s, e = self._demo_slices[i]
            item["demo"] = self.demo[s:e]
        if self.notes is not None:
            if not hasattr(self, "_notes_patients"):
                records = self.notes
                patients, cur = [], []
                for r in records:
                    if isinstance(r, dict) and r.get("_") == -1:
                        if cur:
                            patients.append(cur); cur = []
                    else:
                        cur.append(r)
                if cur: patients.append(cur)
                self._notes_patients = patients
            item["notes"] = self._notes_patients[pid]
        return item
