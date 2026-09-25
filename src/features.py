"""
features.py (Enhanced v2 — 36 Features: String, Entity-Relative & Address Unit Features)
---------------------------------------------------------------------------------------
Feature categories:
1. NAME SIMILARITY (10 features):
   - ratio, token_sort_ratio, token_set_ratio, WRatio
   - Jaro-Winkler, Levenshtein normalized similarity
   - length difference, normalized length difference, token count difference
   - exact equality indicator
2. ADDRESS SIMILARITY (10 features):
   - ratio, token_sort_ratio, token_set_ratio, WRatio
   - Jaro-Winkler, Levenshtein normalized similarity
   - length difference, normalized length difference, token count difference
   - exact equality indicator
3. BLOCKING SCORES & METADATA (5 features):
   - name_idf_score, address_idf_score, prefix_idf_score, total_block_score
   - candidate_rank
4. ENTITY-RELATIVE CONTEXTUAL FEATURES (4 features):
   - block_score_diff (max S1 block score - candidate block score)
   - block_score_ratio (candidate block score / max S1 block score)
   - is_best_addr (1 if candidate has highest address IDF in S1 pool)
   - is_best_name (1 if candidate has highest name IDF in S1 pool)
5. GENERIC ADDRESS UNIT & NUMBER FEATURES (6 features):
   - unit_match (both have unit/suite/gala/plot/villa and they match)
   - unit_mismatch (both have unit/suite/gala/plot/villa and they differ)
   - first_num_match (leading street/door number matches)
   - first_num_mismatch (leading street/door number differs)
   - num_overlap_count (number of shared numeric tokens)
   - num_jaccard (Jaccard similarity of numeric tokens)
6. OTHER (1 feature):
   - country_match
"""

import gc
import os
import re
import math
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from common import normalize_text, LEGAL_SUFFIX_MAP, ADDRESS_ABBR_MAP

# Generic regex patterns for address unit and house number extraction
UNIT_PATTERN = re.compile(
    r'\b(?:unit|suite|ste|gala|plot|villa|flat|block|door|shop|room|floor|apt|apartment|no)\b\s*[:.#/-]?\s*([a-z0-9/-]+)',
    re.IGNORECASE
)
NUMBER_PATTERN = re.compile(r'\b\d+[a-z]?(?:[/-]\d+[a-z]?)?\b', re.IGNORECASE)


def extract_units_and_numbers(text: str) -> tuple[set, set, str | None]:
    """Extract unit identifiers, standalone numbers, and first numeric token from address."""
    if not text or not isinstance(text, str):
        return set(), set(), None
    t = text.lower().strip()
    raw_units = set(UNIT_PATTERN.findall(t))
    cleaned_units = set()
    for u in raw_units:
        u_clean = re.sub(r'^(?:no|#|[:.-])+', '', u).strip()
        if u_clean:
            cleaned_units.add(u_clean)

    numbers = set(NUMBER_PATTERN.findall(t))
    first_m = NUMBER_PATTERN.search(t)
    first_num = first_m.group(0) if first_m else None
    return cleaned_units, numbers, first_num


def _string_features(a: str, b: str, prefix: str) -> dict:
    """Compute comprehensive pairwise similarity features between two strings."""
    a, b = a or "", b or ""
    len_a, len_b = len(a), len(b)
    max_len = max(len_a, len_b, 1)
    tokens_a = a.split()
    tokens_b = b.split()

    return {
        f"{prefix}_ratio": np.float32(fuzz.ratio(a, b)),
        f"{prefix}_token_sort": np.float32(fuzz.token_sort_ratio(a, b)),
        f"{prefix}_token_set": np.float32(fuzz.token_set_ratio(a, b)),
        f"{prefix}_w_ratio": np.float32(fuzz.WRatio(a, b)),
        f"{prefix}_jaro_winkler": np.float32(JaroWinkler.normalized_similarity(a, b) * 100.0),
        f"{prefix}_levenshtein": np.float32(Levenshtein.normalized_similarity(a, b) * 100.0),
        f"{prefix}_len_diff": np.float32(abs(len_a - len_b)),
        f"{prefix}_norm_len_diff": np.float32(abs(len_a - len_b) / max_len),
        f"{prefix}_token_count_diff": np.float32(abs(len(tokens_a) - len(tokens_b))),
        f"{prefix}_exact_match": np.float32(1.0 if a == b and len_a > 0 else 0.0),
    }


def compute_features_chunk(
    chunk_df: pd.DataFrame,
    s1_lookup: dict,
    other_lookup: dict,
) -> pd.DataFrame:
    """Compute features for a single chunk of candidate pairs."""
    has_name_idf = "name_idf_score" in chunk_df.columns
    has_addr_idf = "address_idf_score" in chunk_df.columns
    has_pref_idf = "prefix_idf_score" in chunk_df.columns
    has_total_block = "total_block_score" in chunk_df.columns
    has_rank = "candidate_rank" in chunk_df.columns

    rows = []
    for row in chunk_df.itertuples(index=False):
        s1 = s1_lookup.get(row.source1_entity_id)
        s2 = other_lookup.get(row.candidate_entity_id)
        if s1 is None or s2 is None:
            continue

        raw_name_a = s1.get("business_name", "")
        raw_name_b = s2.get("business_name", "")
        raw_addr_a = s1.get("business_address", "")
        raw_addr_b = s2.get("business_address", "")

        name_a = s1.get("name_norm") or normalize_text(raw_name_a, LEGAL_SUFFIX_MAP)
        name_b = s2.get("name_norm") or normalize_text(raw_name_b, LEGAL_SUFFIX_MAP)
        addr_a = s1.get("addr_norm") or normalize_text(raw_addr_a, ADDRESS_ABBR_MAP)
        addr_b = s2.get("addr_norm") or normalize_text(raw_addr_b, ADDRESS_ABBR_MAP)

        feat = {
            "source1_entity_id": row.source1_entity_id,
            "candidate_entity_id": row.candidate_entity_id,
        }

        # 1. Name & address string metrics
        feat.update(_string_features(name_a, name_b, "name"))
        feat.update(_string_features(addr_a, addr_b, "addr"))

        # 2. Blocking features
        feat["name_idf_score"] = np.float32(getattr(row, "name_idf_score", 0.0) if has_name_idf else 0.0)
        feat["address_idf_score"] = np.float32(getattr(row, "address_idf_score", 0.0) if has_addr_idf else 0.0)
        feat["prefix_idf_score"] = np.float32(getattr(row, "prefix_idf_score", 0.0) if has_pref_idf else 0.0)
        feat["total_block_score"] = np.float32(getattr(row, "total_block_score", 0.0) if has_total_block else 0.0)
        feat["candidate_rank"] = np.float32(getattr(row, "candidate_rank", 0.0) if has_rank else 0.0)

        # 3. Generic address unit & number features
        u_a, num_a, f_a = extract_units_and_numbers(raw_addr_a)
        u_b, num_b, f_b = extract_units_and_numbers(raw_addr_b)

        both_u = bool(u_a and u_b)
        feat["unit_match"] = np.float32(1.0 if both_u and bool(u_a & u_b) else 0.0)
        feat["unit_mismatch"] = np.float32(1.0 if both_u and not bool(u_a & u_b) else 0.0)

        both_f = bool(f_a and f_b)
        feat["first_num_match"] = np.float32(1.0 if both_f and f_a == f_b else 0.0)
        feat["first_num_mismatch"] = np.float32(1.0 if both_f and f_a != f_b else 0.0)

        both_num = bool(num_a or num_b)
        feat["num_overlap_count"] = np.float32(len(num_a & num_b))
        feat["num_jaccard"] = np.float32(len(num_a & num_b) / len(num_a | num_b) if both_num else 0.0)

        # 4. Country match
        c1 = (s1.get("country", "") or "").strip().lower()
        c2 = (s2.get("country", "") or "").strip().lower()
        feat["country_match"] = np.float32(1.0 if c1 == c2 and c1 != "" else 0.0)

        # Preserve label if present
        if hasattr(row, "label"):
            feat["label"] = int(row.label)

        rows.append(feat)

    chunk_result = pd.DataFrame(rows)

    # 5. Entity-Relative Features computed per S1 group within this chunk
    if len(chunk_result) > 0:
        grouped = chunk_result.groupby("source1_entity_id")
        max_block = grouped["total_block_score"].transform("max")
        chunk_result["block_score_diff"] = (max_block - chunk_result["total_block_score"]).astype(np.float32)
        chunk_result["block_score_ratio"] = (chunk_result["total_block_score"] / np.maximum(max_block, 1e-4)).astype(np.float32)

        max_addr = grouped["address_idf_score"].transform("max")
        chunk_result["is_best_addr"] = ((chunk_result["address_idf_score"] == max_addr) & (max_addr > 0)).astype(np.float32)

        max_name = grouped["name_idf_score"].transform("max")
        chunk_result["is_best_name"] = ((chunk_result["name_idf_score"] == max_name) & (max_name > 0)).astype(np.float32)

    return chunk_result


def compute_features(
    pairs_df: pd.DataFrame,
    s1_lookup: dict,
    other_lookup: dict,
    chunk_size: int = 250000,
    output_parquet_path: str | None = None,
) -> pd.DataFrame:
    """
    Memory-safe chunked feature extraction.
    If output_parquet_path is specified, writes chunks to Parquet.
    Otherwise returns a single consolidated DataFrame.
    """
    total_pairs = len(pairs_df)
    if total_pairs == 0:
        empty = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)
        return empty

    num_chunks = max(1, math.ceil(total_pairs / chunk_size))
    print(f"  [features] Computing features for {total_pairs:,} pairs in {num_chunks} chunk(s) (chunk_size={chunk_size:,})...")

    parquet_writer = None
    chunks_dfs = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_pairs)
        chunk_slice = pairs_df.iloc[start_idx:end_idx]

        chunk_feat = compute_features_chunk(chunk_slice, s1_lookup, other_lookup)

        if output_parquet_path:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(chunk_feat)
            if parquet_writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
                parquet_writer = pq.ParquetWriter(output_parquet_path, table.schema, compression="zstd")
            parquet_writer.write_table(table)
            del table, chunk_feat
            gc.collect()
        else:
            chunks_dfs.append(chunk_feat)

    if output_parquet_path and parquet_writer:
        parquet_writer.close()
        print(f"  [features] Wrote {total_pairs:,} feature rows to {output_parquet_path}")
        return pd.read_parquet(output_parquet_path)

    consolidated = pd.concat(chunks_dfs, ignore_index=True)
    return consolidated


def compute_features_country_partitioned(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_parquet_path: str,
    chunk_size: int = 100000,
    verbose: bool = True,
):
    """
    Memory-safe country-partitioned feature extraction.
    Builds country-scoped lookups filtered strictly to the candidate IDs present
    in each country's candidate pairs, streaming feature chunks directly to Parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_pairs = len(pairs_df)
    if total_pairs == 0:
        empty = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)
        table = pa.Table.from_pandas(empty)
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
        pq.write_table(table, output_parquet_path, compression="zstd")
        return

    countries = s1_df["country"].unique()
    if verbose:
        print(f"  [features] Streaming features across {len(countries)} country partition(s) for {total_pairs:,} pairs...")

    parquet_writer = None
    processed_count = 0

    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        s1_ids_c = set(s1_c["entity_id"])
        pairs_c = pairs_df[pairs_df["source1_entity_id"].isin(s1_ids_c)]

        if len(pairs_c) == 0:
            continue

        needed_cands = set(pairs_c["candidate_entity_id"])
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]
        pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
        pool_filtered = pool_c[pool_c["entity_id"].isin(needed_cands)].drop_duplicates(subset="entity_id")

        s1_lookup_c = s1_c.drop_duplicates(subset="entity_id").set_index("entity_id").to_dict("index")
        other_lookup_c = pool_filtered.set_index("entity_id").to_dict("index")
        del pool_c, pool_filtered

        n_chunks = math.ceil(len(pairs_c) / chunk_size)
        for ch in range(n_chunks):
            start = ch * chunk_size
            end = min(start + chunk_size, len(pairs_c))
            chunk_slice = pairs_c.iloc[start:end]

            chunk_feat = compute_features_chunk(chunk_slice, s1_lookup_c, other_lookup_c)
            table = pa.Table.from_pandas(chunk_feat)
            if parquet_writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
                parquet_writer = pq.ParquetWriter(output_parquet_path, table.schema, compression="zstd")
            parquet_writer.write_table(table)
            processed_count += len(chunk_feat)
            del table, chunk_feat
            gc.collect()

        del s1_lookup_c, other_lookup_c
        gc.collect()

    if parquet_writer:
        parquet_writer.close()
    if verbose:
        print(f"  [features] Wrote {processed_count:,} feature rows to {output_parquet_path}")


FEATURE_COLUMNS = [
    # 1. Name features (10)
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_w_ratio",
    "name_jaro_winkler",
    "name_levenshtein",
    "name_len_diff",
    "name_norm_len_diff",
    "name_token_count_diff",
    "name_exact_match",
    # 2. Address features (10)
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_w_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein",
    "addr_len_diff",
    "addr_norm_len_diff",
    "addr_token_count_diff",
    "addr_exact_match",
    # 3. Blocking features (5)
    "name_idf_score",
    "address_idf_score",
    "prefix_idf_score",
    "total_block_score",
    "candidate_rank",
    # 4. Entity-Relative Contextual features (4)
    "block_score_diff",
    "block_score_ratio",
    "is_best_addr",
    "is_best_name",
    # 5. Generic Address Unit & Number features (6)
    "unit_match",
    "unit_mismatch",
    "first_num_match",
    "first_num_mismatch",
    "num_overlap_count",
    "num_jaccard",
    # 6. Country match (1)
    "country_match",
]
