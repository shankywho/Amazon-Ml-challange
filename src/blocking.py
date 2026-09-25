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


import os
import multiprocessing as mp


def _score_s1_batch(batch_args) -> list:
    """Worker function for parallel scoring of a slice of S1 queries."""
    (
        s1_ids_slice,
        s1_name_toks_slice,
        s1_addr_toks_slice,
        s1_pref_toks_slice,
        name_index,
        addr_index,
        pref_index,
        name_idf,
        addr_idf,
        pref_idf,
        pool_ids,
        k,
        name_weight,
        address_weight,
        prefix_weight,
        multi_perspective,
        return_metadata,
    ) = batch_args

    k_comb = max(1, int(k * 0.70))
    k_addr = max(1, int(k * 0.30))
    k_name = max(1, int(k * 0.20))

    country_rows = []
    for i in range(len(s1_ids_slice)):
        s1_id = s1_ids_slice[i]
        q_name_toks = s1_name_toks_slice[i]
        q_addr_toks = s1_addr_toks_slice[i]
        q_pref_toks = s1_pref_toks_slice[i]

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
            top_comb = sorted(comb_scores.keys(), key=comb_scores.get, reverse=True)[:k_comb]
            top_addr = sorted(addr_scores.keys(), key=addr_scores.get, reverse=True)[:k_addr]
            top_name = sorted(name_scores.keys(), key=name_scores.get, reverse=True)[:k_name]

            selected = []
            seen_idx = set()
            for c_idx in top_comb + top_addr + top_name:
                if c_idx not in seen_idx:
                    seen_idx.add(c_idx)
                    selected.append(c_idx)
                    if len(selected) >= k:
                        break
        else:
            selected = sorted(comb_scores.keys(), key=comb_scores.get, reverse=True)[:k]

        selected.sort(key=comb_scores.get, reverse=True)

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

    return country_rows


def _tokenize(text: str, min_len: int = 2) -> list:
    if not text:
        return []
    return [w for w in text.split() if len(w) >= min_len]


def _prefixes(text: str, min_len: int = 4, pref_len: int = 4) -> list:
    if not text:
        return []
    return [w[:pref_len] for w in text.split() if len(w) >= min_len]


def build_candidates_for_country(
    s1_c: pd.DataFrame,
    s2_c: pd.DataFrame,
    s3_c: pd.DataFrame,
    country: str,
    k: int = 50,
    max_block_size: int = 50000,
    name_weight: float = 1.5,
    address_weight: float = 1.5,
    prefix_weight: float = 1.0,
    multi_perspective: bool = True,
    min_token_len: int = 2,
    return_metadata: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate top-K candidate pairs for a single country partition.
    Guarantees zero cross-country pollution and releases memory after completion.
    """
    if "name_norm" not in s1_c.columns:
        s1_c = add_normalized_columns(s1_c)
    if "name_norm" not in s2_c.columns:
        s2_c = add_normalized_columns(s2_c)
    if "name_norm" not in s3_c.columns:
        s3_c = add_normalized_columns(s3_c)

    pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
    N_pool = len(pool_c)

    if len(s1_c) == 0 or N_pool == 0:
        del pool_c
        return pd.DataFrame()

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

    # Score candidates for each S1 query (parallelized across CPU cores)
    n_workers = min(os.cpu_count() or 4, 8)
    if len(s1_ids) <= 100 or n_workers <= 1:
        country_rows = _score_s1_batch((
            s1_ids,
            s1_name_toks,
            s1_addr_toks,
            s1_pref_toks,
            name_index,
            addr_index,
            pref_index,
            name_idf,
            addr_idf,
            pref_idf,
            pool_ids,
            k,
            name_weight,
            address_weight,
            prefix_weight,
            multi_perspective,
            return_metadata,
        ))
    else:
        chunk_sz = math.ceil(len(s1_ids) / n_workers)
        tasks = []
        for w in range(n_workers):
            start = w * chunk_sz
            end = min(start + chunk_sz, len(s1_ids))
            if start >= end:
                continue
            tasks.append((
                s1_ids[start:end],
                s1_name_toks[start:end],
                s1_addr_toks[start:end],
                s1_pref_toks[start:end],
                name_index,
                addr_index,
                pref_index,
                name_idf,
                addr_idf,
                pref_idf,
                pool_ids,
                k,
                name_weight,
                address_weight,
                prefix_weight,
                multi_perspective,
                return_metadata,
            ))
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_score_s1_batch, tasks)
        country_rows = [row for batch in results for row in batch]

    del name_index, addr_index, pref_index, name_idf, addr_idf, pref_idf
    del pool_c
    gc.collect()

    return pd.DataFrame(country_rows)


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
    output_parquet_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Generate top-K candidate pairs for each Source 1 entity partitioned by country.
    If output_parquet_path is specified, writes each country directly to Parquet on disk
    without accumulating candidate DataFrames in memory.
    """
    if verbose:
        print(f"  [blocking] Starting candidate generation (frozen K={k}, max_block_size={max_block_size})...")

    countries = s1_df["country"].unique()
    if verbose:
        print(f"  [blocking] Dynamic country partitions: {list(countries)}")

    parquet_writer = None
    country_dfs = []

    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]

        cands_c = build_candidates_for_country(
            s1_c, s2_c, s3_c, country,
            k=k,
            max_block_size=max_block_size,
            name_weight=name_weight,
            address_weight=address_weight,
            prefix_weight=prefix_weight,
            multi_perspective=multi_perspective,
            min_token_len=min_token_len,
            return_metadata=return_metadata,
            verbose=verbose,
        )

        if len(cands_c) == 0:
            continue

        if output_parquet_path:
            import os
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(cands_c)
            if parquet_writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
                parquet_writer = pq.ParquetWriter(output_parquet_path, table.schema, compression="zstd")
            parquet_writer.write_table(table)
            del table, cands_c
            gc.collect()
        else:
            country_dfs.append(cands_c)

    if output_parquet_path and parquet_writer:
        parquet_writer.close()
        if verbose:
            print(f"  [blocking] Streamed candidates directly to {output_parquet_path}")
        return None

    if country_dfs:
        candidates_df = pd.concat(country_dfs, ignore_index=True)
    else:
        candidates_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    if verbose:
        s1_with_cands = candidates_df["source1_entity_id"].nunique() if len(candidates_df) > 0 else 0
        print(f"  [blocking] Finished — {len(candidates_df):,} candidate pairs for {s1_with_cands:,} / {len(s1_df):,} S1 entities.")
    return candidates_df


def candidates_to_tsv_format(candidates_df: pd.DataFrame, all_s1_ids: list) -> pd.DataFrame:
    """
    Format candidates into the official submission candidate_pairs.tsv format:
    source1_entity_id \t candidate_entity_ids (comma-separated, empty if singleton)
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


def append_candidates_to_tsv(
    candidates_c_df: pd.DataFrame,
    s1_c_ids: list,
    file_handle,
):
    """
    Incrementally appends candidate pairs for a country partition directly to an open TSV file handle.
    Guarantees every S1 entity appears (singletons get empty string).
    """
    cands_by_s1 = defaultdict(list)
    if len(candidates_c_df) > 0:
        for r in candidates_c_df[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
            cands_by_s1[r.source1_entity_id].append(r.candidate_entity_id)

    for s1_id in s1_c_ids:
        c_list = ",".join(dict.fromkeys(cands_by_s1.get(s1_id, [])))
        file_handle.write(f"{s1_id}\t{c_list}\n")
