"""
blocking.py (Frozen v4 — Country Partitioned, IDF-Weighted Multi-Perspective Inverted Index)
---------------------------------------------------------------------------------------------
Architecture:
  1. Dynamic Country Partitioning:
     Processes records per country (US, India, France, etc.) independently.
     Guarantees 0 cross-country candidate pollution and keeps memory footprint minimal.
  2. Memory-Safe Inverted Indexing:
     Uses lightweight integer array posting lists for pool records.
     No explosive DataFrame `explode()` or Cartesian joins.
  3. IDF Evidence Weighting:
     Tokens are weighted by Inverse Document Frequency:
       idf(token) = ln((N_pool + 1) / (df + 1)) + 1
     Distinctive tokens dominate scoring; generic stopwords are suppressed.
  4. Three High-Recall Signals:
     - Name tokens (min length 2)
     - Address tokens (min length 2, alphanumeric / house numbers preserved)
     - Name prefixes (4-character prefixes of words >= 4 chars for typo tolerance)
  5. Multi-Perspective Candidate Pooling:
     Pools top candidates from composite score, address-only score, and name-only score
     to ensure cross-script/transliteration matches (e.g. Indian languages) and
     sparse-address matches are not crowded out.
  6. Frozen K=50:
     Yields ~94.84% recall ceiling with tight candidate volume.
"""

import gc
import math
from collections import defaultdict
import pandas as pd

from common import add_normalized_columns


def _tokenize(text: str, min_len: int = 2) -> list:
    if not text:
        return []
    return [w for w in text.split() if len(w) >= min_len]


def _prefixes(text: str, min_len: int = 4, pref_len: int = 4) -> list:
    if not text:
        return []
    return [w[:pref_len] for w in text.split() if len(w) >= min_len]


def build_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = 50,
    max_block_size: int = 50000,
    name_weight: float = 1.5,
    address_weight: float = 1.5,
    prefix_weight: float = 1.0,
    multi_perspective: bool = True,
    min_token_len: int = 2,
    return_metadata: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate top-K candidate pairs for each Source 1 entity partitioned by country.

    Returns DataFrame with:
      source1_entity_id, candidate_entity_id,
      (plus name_idf_score, address_idf_score, prefix_idf_score, total_block_score, candidate_rank
       if return_metadata=True)
    """
    if verbose:
        print(f"  [blocking] Starting candidate generation (frozen K={k}, max_block_size={max_block_size})...")

    # Normalize if not already normalized
    if "name_norm" not in s1_df.columns:
        s1_df = add_normalized_columns(s1_df)
    if "name_norm" not in s2_df.columns:
        s2_df = add_normalized_columns(s2_df)
    if "name_norm" not in s3_df.columns:
        s3_df = add_normalized_columns(s3_df)

    all_rows = []
    countries = s1_df["country"].unique()
    if verbose:
        print(f"  [blocking] Dynamic country partitions: {list(countries)}")

    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]
        pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
        N_pool = len(pool_c)

        if len(s1_c) == 0:
            continue
        if N_pool == 0:
            if verbose:
                print(f"  [blocking] Country {country}: Pool is empty! No candidates generated for {len(s1_c)} S1 entities.")
            continue

        if verbose:
            print(f"  [blocking] Country {country}: S1={len(s1_c):,}, Pool={N_pool:,}...")

        s1_names = s1_c["name_norm"].values
        s1_addrs = s1_c["addr_norm"].values
        s1_ids = s1_c["entity_id"].values

        s1_name_toks = [set(_tokenize(t, min_token_len)) for t in s1_names]
        s1_addr_toks = [set(_tokenize(t, min_token_len)) for t in s1_addrs]
        s1_pref_toks = [set(_prefixes(t, min_len=4, pref_len=4)) for t in s1_names]

        needed_name = set.union(*s1_name_toks) if s1_name_toks else set()
        needed_addr = set.union(*s1_addr_toks) if s1_addr_toks else set()
        needed_pref = set.union(*s1_pref_toks) if s1_pref_toks else set()

        pool_ids = pool_c["entity_id"].values
        pool_names = pool_c["name_norm"].values
        pool_addrs = pool_c["addr_norm"].values

        name_index = defaultdict(list)
        addr_index = defaultdict(list)
        pref_index = defaultdict(list)

        for idx in range(N_pool):
            p_name = pool_names[idx]
            p_addr = pool_addrs[idx]

            for tok in set(_tokenize(p_name, min_token_len)):
                if tok in needed_name:
                    name_index[tok].append(idx)

            for tok in set(_tokenize(p_addr, min_token_len)):
                if tok in needed_addr:
                    addr_index[tok].append(idx)

            for pref in set(_prefixes(p_name, min_len=4, pref_len=4)):
                if pref in needed_pref:
                    pref_index[pref].append(idx)

        # Token IDF mapping: ln((N_pool + 1) / (df + 1)) + 1
        def _calc_idf(inv_idx):
            idf_map = {}
            for tok, post in inv_idx.items():
                df = len(post)
                if df > max_block_size:
                    continue  # suppress high-frequency posting lists
                idf_map[tok] = math.log((N_pool + 1.0) / (df + 1.0)) + 1.0
            return idf_map

        name_idf = _calc_idf(name_index)
        addr_idf = _calc_idf(addr_index)
        pref_idf = _calc_idf(pref_index)

        # Score candidates for each S1 query
        country_rows = []
        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            q_name_toks = s1_name_toks[i]
            q_addr_toks = s1_addr_toks[i]
            q_pref_toks = s1_pref_toks[i]

            name_scores = defaultdict(float)
            addr_scores = defaultdict(float)
            pref_scores = defaultdict(float)
            all_candidate_indices = set()

            for tok in q_name_toks:
                if tok in name_idf:
                    w = name_idf[tok]
                    for c_idx in name_index[tok]:
                        name_scores[c_idx] += w
                        all_candidate_indices.add(c_idx)

            for pref in q_pref_toks:
                if pref in pref_idf:
                    w = pref_idf[pref]
                    for c_idx in pref_index[pref]:
                        pref_scores[c_idx] += w
                        all_candidate_indices.add(c_idx)

            for tok in q_addr_toks:
                if tok in addr_idf:
                    w = addr_idf[tok]
                    for c_idx in addr_index[tok]:
                        addr_scores[c_idx] += w
                        all_candidate_indices.add(c_idx)

            if not all_candidate_indices:
                continue

            comb_scores = {}
            for c_idx in all_candidate_indices:
                comb_scores[c_idx] = (
                    name_weight * name_scores[c_idx]
                    + address_weight * addr_scores[c_idx]
                    + prefix_weight * pref_scores[c_idx]
                )

            if multi_perspective:
                k_comb = max(1, int(k * 0.70))
                k_addr = max(1, int(k * 0.30))
                k_name = max(1, int(k * 0.20))

                top_comb = sorted(comb_scores.keys(), key=lambda idx: comb_scores[idx], reverse=True)[:k_comb]
                top_addr = sorted(addr_scores.keys(), key=lambda idx: addr_scores[idx], reverse=True)[:k_addr]
                top_name = sorted(name_scores.keys(), key=lambda idx: name_scores[idx], reverse=True)[:k_name]

                selected = []
                seen_idx = set()
                for c_idx in top_comb + top_addr + top_name:
                    if c_idx not in seen_idx:
                        seen_idx.add(c_idx)
                        selected.append(c_idx)
                        if len(selected) >= k:
                            break
            else:
                selected = sorted(comb_scores.keys(), key=lambda idx: comb_scores[idx], reverse=True)[:k]

            selected.sort(key=lambda idx: comb_scores[idx], reverse=True)

            for rank, c_idx in enumerate(selected):
                cand_id = pool_ids[c_idx]
                if return_metadata:
                    country_rows.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "name_idf_score": round(name_scores[c_idx], 4),
                        "address_idf_score": round(addr_scores[c_idx], 4),
                        "prefix_idf_score": round(pref_scores[c_idx], 4),
                        "total_block_score": round(comb_scores[c_idx], 4),
                        "candidate_rank": rank,
                    })
                else:
                    country_rows.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                    })

        all_rows.extend(country_rows)

        del name_index, addr_index, pref_index, name_idf, addr_idf, pref_idf
        del pool_c, s1_c, s2_c, s3_c, country_rows
        gc.collect()

    candidates_df = pd.DataFrame(all_rows)
    if verbose:
        s1_with_cands = candidates_df["source1_entity_id"].nunique() if len(candidates_df) > 0 else 0
        print(f"  [blocking] Finished — {len(candidates_df):,} candidate pairs for {s1_with_cands:,} / {len(s1_df):,} S1 entities.")
    return candidates_df


def candidates_to_tsv_format(candidates_df: pd.DataFrame, all_s1_ids: list) -> pd.DataFrame:
    """
    Format candidates into the official submission candidate_pairs.tsv format:
    source1_entity_id \\t candidate_entity_ids (comma-separated, empty if singleton)
    """
    if len(candidates_df) > 0:
        grouped = (
            candidates_df.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(dict.fromkeys(ids)))
            .reset_index()
        )
        grouped = grouped.rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    else:
        grouped = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])

    full = pd.DataFrame({"source1_entity_id": all_s1_ids})
    full = full.merge(grouped, on="source1_entity_id", how="left")
    full["candidate_entity_ids"] = full["candidate_entity_ids"].fillna("")
    return full
