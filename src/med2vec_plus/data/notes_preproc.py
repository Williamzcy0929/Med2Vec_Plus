import re
from typing import Dict

SECTION_PATTERNS = {
    "dx": re.compile(r"(?im)^\s*(diagnosis|diagnoses)\s*:\s*$"),
    "proc": re.compile(r"(?im)^\s*(procedures?)\s*:\s*$"),
    "treat": re.compile(r"(?im)^\s*(medications?|meds?)\s*:\s*$"),
    "ap": re.compile(r"(?im)^\s*(assessment\s*&\s*plan|assessment and plan|plan)\s*:\s*$"),
}

def rough_sectionize(text: str) -> Dict[str, str]:
    if not text or not text.strip():
        return {"dx": "", "proc": "", "treat": "", "ap": ""}
    lines = text.splitlines()
    current = "ap"
    out = {"dx": "", "proc": "", "treat": "", "ap": ""}
    for ln in lines:
        l = ln.strip()
        switched = False
        for k, pat in SECTION_PATTERNS.items():
            if pat.match(l):
                current = k
                switched = True
                break
        if switched:
            continue
        out[current] += ln + "\n"
    return out
