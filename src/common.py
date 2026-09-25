"""
common.py
---------
Shared helpers used by every stage of the pipeline: loading TSVs and
normalizing messy business names/addresses before we compare them.

WHY NORMALIZE FIRST?
Two records like "ABC Pvt Ltd" and "ABC Private Limited" are the SAME
business, but a plain string-similarity score treats "Pvt" and
"Private" as totally different words. If we expand abbreviations to a
common form BEFORE comparing, the similarity scores in features.py
become much more accurate — this is a free accuracy win that costs
almost nothing computationally.
"""

import re
import pandas as pd

# Legal-suffix / business-abbreviation normalization.
# Left side = pattern to find (word-boundary matched), right side = what to replace it with.
# Add more pairs here as you spot new abbreviations in the data during EDA.
LEGAL_SUFFIX_MAP = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\b&\b": "and",
}

# Common address abbreviations. Same idea — expand so "Rd" and "Road" match.
ADDRESS_ABBR_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}


# Precompiled module-level fast regexes for single-pass alternation
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

_LEGAL_MAP = {k.replace(r"\b", ""): v for k, v in LEGAL_SUFFIX_MAP.items()}
_LEGAL_PAT = re.compile(r"\b(" + "|".join(re.escape(k.replace(r"\b", "")) for k in LEGAL_SUFFIX_MAP) + r")\b")

_ADDR_MAP = {k.replace(r"\b", ""): v for k, v in ADDRESS_ABBR_MAP.items()}
_ADDR_PAT = re.compile(r"\b(" + "|".join(re.escape(k.replace(r"\b", "")) for k in ADDRESS_ABBR_MAP) + r")\b")


def load_source(path: str) -> pd.DataFrame:
    """Load one source TSV file. Always use sep='\\t' — see problem statement warning
    about commas inside address/ID-list fields silently breaking a comma-separated read."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")  # missing address/country components become empty strings, not NaN
    return df


def normalize_text(text: str, extra_map: dict | None = None) -> str:
    """Lowercase + expand abbreviations + strip extra punctuation/whitespace.
    Applied to business_name and business_address independently.
    Uses precompiled single-pass regex alternation for maximum throughput."""
    if not isinstance(text, str) or text == "":
        return ""
    t = _PUNCT_RE.sub(" ", text.lower())
    if extra_map is LEGAL_SUFFIX_MAP or extra_map == LEGAL_SUFFIX_MAP:
        t = _LEGAL_PAT.sub(lambda m: _LEGAL_MAP[m.group(0)], t)
    elif extra_map is ADDRESS_ABBR_MAP or extra_map == ADDRESS_ABBR_MAP:
        t = _ADDR_PAT.sub(lambda m: _ADDR_MAP[m.group(0)], t)
    elif extra_map:
        for pattern, replacement in extra_map.items():
            t = re.sub(pattern, replacement, t)
    return _WS_RE.sub(" ", t).strip()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds name_norm and addr_norm columns used downstream."""
    df = df.copy()
    names = df["business_name"].fillna("").astype(str).values
    addrs = df["business_address"].fillna("").astype(str).values
    df["name_norm"] = [normalize_text(x, LEGAL_SUFFIX_MAP) for x in names]
    df["addr_norm"] = [normalize_text(x, ADDRESS_ABBR_MAP) for x in addrs]
    return df
