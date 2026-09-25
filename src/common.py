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


def load_source(path: str) -> pd.DataFrame:
    """Load one source TSV file. Always use sep='\\t' — see problem statement warning
    about commas inside address/ID-list fields silently breaking a comma-separated read."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")  # missing address/country components become empty strings, not NaN
    return df


def normalize_text(text: str, extra_map: dict | None = None) -> str:
    """Lowercase + expand abbreviations + strip extra punctuation/whitespace.
    Applied to business_name and business_address independently."""
    if not isinstance(text, str) or text == "":
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)  # drop punctuation (keeps word chars + spaces)
    if extra_map:
        for pattern, replacement in extra_map.items():
            t = re.sub(pattern, replacement, t)
    t = re.sub(r"\s+", " ", t).strip()  # collapse repeated whitespace
    return t


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds name_norm, addr_norm, blocking_text columns used downstream."""
    df = df.copy()
    df["name_norm"] = df["business_name"].apply(lambda x: normalize_text(x, LEGAL_SUFFIX_MAP))
    df["addr_norm"] = df["business_address"].apply(lambda x: normalize_text(x, ADDRESS_ABBR_MAP))
    df["blocking_text"] = (df["name_norm"] + " " + df["addr_norm"]).str.strip()
    return df
