from dataclasses import dataclass
from typing import Dict, List, Iterable
import pickle, collections

ASPECTS = ["dx", "proc", "treat"]

@dataclass
class Vocab:
    code2id: Dict[str, int]
    id2code: List[str]

    @classmethod
    def from_codes(cls, codes: Iterable[str]):
        counter = collections.Counter(codes)
        id2code = sorted(counter.keys())
        code2id = {c: i for i, c in enumerate(id2code)}
        return cls(code2id=code2id, id2code=id2code)

    @classmethod
    def from_pickle(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "code2id" in obj:
            return cls(code2id=obj["code2id"], id2code=obj["id2code"])
        raise ValueError("Invalid vocab pickle format.")

    def size(self) -> int:
        return len(self.id2code)

def load_multi_vocab(path: str) -> Dict[str, Vocab]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return {a: Vocab(code2id=obj[a]["code2id"], id2code=obj[a]["id2code"]) for a in ASPECTS}
