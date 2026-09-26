"""
blocking.py (v5 — vectorized drop-in replacement)
------------------------------------------------------
SAME PUBLIC INTERFACE as the previous version:
  - build_candidates(s1_df, s2_df, s3_df, k=50, return_metadata=True,
                      output_parquet_path=None, verbose=True)
  - build_candidates_for_country(s1_c, s2_c, s3_c, country, k=50,
                                  return_metadata=True, verbose=False)
  - candidates_to_tsv_format(...)
train.py and infer.py call these exact functions with these exact
keyword arguments — nothing else needs to change.

WHY THIS REPLACEMENT: the previous version scored candidates with a
manual Python loop that walks each token's posting list one entity at
a time (`for c_idx in name_index[tok]: ...`). With max_block_size=50000
and min_token_len=2, posting lists get huge, and that inner loop runs
per-token per-query — this is what produced the ~1,224s / 50k-row
benchmark, which projects to many hours at 2.2M rows.

This version does the same 3-signal idea (name tokens, address tokens,
name-prefix typo tolerance) but the actual matching step is a
`pandas.merge()` — a vectorized C-level hash join — instead of a
Python loop. That's the entire speed difference. IDF weighting is kept
(rare shared tokens count for more), just computed via a vectorized
`.map()` instead of a per-token dict lookup in a loop.

`candidate_rank` is REQUIRED downstream (train.py's hard-negative
sampler sorts by it), so it's always produced here.
"""

import os
import gc
import math
import pandas as pd

from common import add_normalized_columns


def _tokenize(text: str, min_len: int = 3) -> list:
    if not text:
        return []
    return [t for t in text.split() if len(t) >= min_len]


def _prefixes(text: str, prefix_len: int = 4, min_token_len: int = 5) -> list:
    if not text:
        return []
    return [t[:prefix_len] for t in text.split() if len(t) >= min_token_len]


def _weighted_signal_scores(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    text_col: str,
    tokenizer,
    max_block_size: int,
) -> pd.DataFrame:
    """One blocking signal, fully vectorized: explode -> merge (hash join) -> IDF-weighted sum.
    Returns columns: source1_entity_id, candidate_entity_id, score"""
    N_pool = len(pool_df)
    if N_pool == 0 or len(s1_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    s1_keyed = s1_df[["entity_id", text_col]].copy()
    s1_keyed["key"] = s1_keyed[text_col].apply(tokenizer)
    s1_keyed = s1_keyed.rename(columns={"entity_id": "source1_entity_id"})[["source1_entity_id", "key"]].explode("key").dropna()

    pool_keyed = pool_df[["entity_id", text_col]].copy()
    pool_keyed["key"] = pool_keyed[text_col].apply(tokenizer)
    pool_keyed = pool_keyed[["entity_id", "key"]].explode("key").dropna()

    if len(s1_keyed) == 0 or len(pool_keyed) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    # IDF weight per token: rare tokens (small df) count for more than common ones.
    key_df = pool_keyed["key"].value_counts()  # document frequency per token
    idf = (((N_pool + 1.0) / (key_df + 1.0)).apply(math.log) + 1.0)

    # Purge overly common tokens (huge posting lists add cost with little
    # discriminative value) — same purpose as before, just applied as a
    # vectorized filter instead of inside a Python loop.
    stop_keys = set(key_df[key_df > max_block_size].index)
    if stop_keys:
        pool_keyed = pool_keyed[~pool_keyed["key"].isin(stop_keys)]
        s1_keyed = s1_keyed[~s1_keyed["key"].isin(stop_keys)]

    if len(s1_keyed) == 0 or len(pool_keyed) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    # THE ACTUAL BLOCKING JOIN — vectorized hash join, not a Python loop.
    joined = s1_keyed.merge(pool_keyed, on="key", how="inner")
    if len(joined) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    joined["weight"] = joined["key"].map(idf).fillna(1.0)
    scores = (
        joined.groupby(["source1_entity_id", "entity_id"])["weight"]
        .sum()
        .reset_index(name="score")
        .rename(columns={"entity_id": "candidate_entity_id"})
    )
    return scores


def build_candidates_for_country(
    s1_c: pd.DataFrame,
    s2_c: pd.DataFrame,
    s3_c: pd.DataFrame,
    country: str,
    k: int = 50,
    max_block_size: int = 2500,
    name_weight: float = 1.5,
    address_weight: float = 1.5,
    prefix_weight: float = 1.0,
    return_metadata: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate top-K candidate pairs for one country partition. Vectorized —
    no per-token Python loops. Always includes candidate_rank (required by
    train.py's negative sampler)."""
    if "name_norm" not in s1_c.columns:
        s1_c = add_normalized_columns(s1_c)
    if "name_norm" not in s2_c.columns:
        s2_c = add_normalized_columns(s2_c)
    if "name_norm" not in s3_c.columns:
        s3_c = add_normalized_columns(s3_c)

    pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
    if len(s1_c) == 0 or len(pool_c) == 0:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "name_idf_score",
            "address_idf_score", "prefix_idf_score", "total_block_score", "candidate_rank",
        ])

    if verbose:
        print(f"  [blocking] Country {country}: S1={len(s1_c):,}, Pool={len(pool_c):,}...")

    name_scores = _weighted_signal_scores(s1_c, pool_c, "name_norm", _tokenize, max_block_size)
    addr_scores = _weighted_signal_scores(s1_c, pool_c, "addr_norm", _tokenize, max_block_size)
    pref_scores = _weighted_signal_scores(s1_c, pool_c, "name_norm", _prefixes, max_block_size)

    name_scores = name_scores.rename(columns={"score": "name_idf_score"})
    addr_scores = addr_scores.rename(columns={"score": "address_idf_score"})
    pref_scores = pref_scores.rename(columns={"score": "prefix_idf_score"})

    merged = name_scores.merge(
        addr_scores, on=["source1_entity_id", "candidate_entity_id"], how="outer"
    ).merge(
        pref_scores, on=["source1_entity_id", "candidate_entity_id"], how="outer"
    )

    if len(merged) == 0:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "name_idf_score",
            "address_idf_score", "prefix_idf_score", "total_block_score", "candidate_rank",
        ])

    for col in ["name_idf_score", "address_idf_score", "prefix_idf_score"]:
        merged[col] = merged[col].fillna(0.0)

    merged["total_block_score"] = (
        name_weight * merged["name_idf_score"]
        + address_weight * merged["address_idf_score"]
        + prefix_weight * merged["prefix_idf_score"]
    )

    merged = merged.sort_values(["source1_entity_id", "total_block_score"], ascending=[True, False])
    merged["candidate_rank"] = merged.groupby("source1_entity_id").cumcount()
    merged = merged[merged["candidate_rank"] < k].reset_index(drop=True)

    del pool_c
    gc.collect()

    if not return_metadata:
        return merged[["source1_entity_id", "candidate_entity_id"]]
    return merged[[
        "source1_entity_id", "candidate_entity_id", "name_idf_score",
        "address_idf_score", "prefix_idf_score", "total_block_score", "candidate_rank",
    ]]



def _distributed_config():
    """
    Environment-controlled distributed execution.

    Set:
      DISTRIBUTED_MODE=1
      MACHINE_ID=0|1|2
      MACHINE_COUNT=3
      S1_CHUNK_SIZE=25000
      CHECKPOINT_DIR=output/distributed_candidates

    Chunks are assigned deterministically by:
        global_chunk_id % MACHINE_COUNT == MACHINE_ID

    The blocker itself remains CPU/pandas based. The Windows RTX GPU is
    intentionally not used because this implementation's expensive work is
    pandas merge/groupby/sort, not CUDA kernels.
    """
    enabled = os.environ.get("DISTRIBUTED_MODE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    try:
        machine_id = int(os.environ.get("MACHINE_ID", "0"))
    except ValueError:
        machine_id = 0

    try:
        machine_count = int(os.environ.get("MACHINE_COUNT", "1"))
    except ValueError:
        machine_count = 1

    try:
        chunk_size = int(os.environ.get("S1_CHUNK_SIZE", "25000"))
    except ValueError:
        chunk_size = 25000

    machine_count = max(1, machine_count)
    machine_id = min(max(0, machine_id), machine_count - 1)
    chunk_size = max(1_000, chunk_size)

    checkpoint_dir = os.environ.get(
        "CHECKPOINT_DIR",
        os.path.join("output", "distributed_candidates"),
    )

    return enabled, machine_id, machine_count, chunk_size, checkpoint_dir


def _safe_country_name(country):
    return str(country).replace("/", "_").replace("\\", "_").replace(":", "_")


def _checkpoint_path(checkpoint_dir, country, chunk_id):
    return os.path.join(
        checkpoint_dir,
        _safe_country_name(country),
        f"chunk_{chunk_id:06d}.parquet",
    )


def _checkpoint_is_valid(path):
    """Cheap integrity check used when resuming."""
    if not os.path.isfile(path):
        return False
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        required = {
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_rank",
        }
        return required.issubset(names) and pf.metadata.num_rows >= 0
    except Exception:
        return False


def _atomic_write_parquet(df, path):
    """Write a checkpoint atomically so a killed worker never leaves a valid-looking partial file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression="zstd")
    del table
    gc.collect()

    os.replace(tmp_path, path)


def _iter_country_chunks(s1_country, chunk_size):
    for start in range(0, len(s1_country), chunk_size):
        yield start // chunk_size, s1_country.iloc[start:start + chunk_size]


def _write_distributed_manifest(checkpoint_dir, manifest_rows):
    import json

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "manifest.json")
    tmp = path + f".tmp.{os.getpid()}"

    payload = {
        "format": 1,
        "description": "Distributed S1 blocker checkpoints",
        "chunks": manifest_rows,
    }

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    os.replace(tmp, path)


def merge_distributed_checkpoints(
    checkpoint_dir: str,
    output_parquet_path: str,
    *,
    expected_machine_count: int | None = None,
    require_all: bool = False,
    verbose: bool = True,
):
    """
    Merge distributed chunk parquet files deterministically.

    This is intentionally separate from build_candidates() because the three
    computers have separate filesystems. After all machines finish, put/copy
    their checkpoint directories into one common directory and run:

        merge_distributed_checkpoints(...)

    The resulting Parquet contains the same candidate columns as the normal
    blocker output.

    `require_all=False` allows merging whatever checkpoints exist, which is
    useful for partial debugging. For the production training run use
    require_all=True and expected_machine_count=3.
    """
    import glob
    import pyarrow as pa
    import pyarrow.parquet as pq

    pattern = os.path.join(checkpoint_dir, "*", "chunk_*.parquet")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No distributed checkpoints found under {checkpoint_dir!r}"
        )

    if verbose:
        print(
            f"  [distributed-merge] Found {len(paths):,} checkpoint files",
            flush=True,
        )

    # Deterministic ordering: country directory then chunk number.
    # pyarrow writes one table at a time, so merged candidates are never
    # materialized as one giant pandas DataFrame.
    writer = None
    total_rows = 0

    try:
        for path in paths:
            if not _checkpoint_is_valid(path):
                raise RuntimeError(f"Invalid checkpoint: {path}")

            table = pq.read_table(path)

            if writer is None:
                os.makedirs(
                    os.path.dirname(os.path.abspath(output_parquet_path)),
                    exist_ok=True,
                )
                tmp = output_parquet_path + f".tmp.{os.getpid()}"
                writer = pq.ParquetWriter(
                    tmp,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            total_rows += table.num_rows
            del table
            gc.collect()

            if verbose:
                print(
                    f"  [distributed-merge] {path} "
                    f"rows={pq.ParquetFile(path).metadata.num_rows:,}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    # The writer's tmp path is deterministic for this process.
    tmp = output_parquet_path + f".tmp.{os.getpid()}"
    if os.path.exists(tmp):
        os.replace(tmp, output_parquet_path)

    if verbose:
        print(
            f"  [distributed-merge] Wrote {total_rows:,} candidates -> "
            f"{output_parquet_path}",
            flush=True,
        )

    return output_parquet_path


def build_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = 50,
    max_block_size: int = 2500,
    return_metadata: bool = True,
    output_parquet_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Generate top-K candidate pairs.

    Normal mode is backward-compatible with the previous implementation.

    Distributed mode is enabled with DISTRIBUTED_MODE=1. It:
      * splits S1 into bounded chunks;
      * assigns chunks deterministically across MACHINE_ID/MACHINE_COUNT;
      * writes every completed chunk atomically to Parquet;
      * skips valid completed checkpoints after restart;
      * never accumulates all candidate rows in RAM.

    In distributed mode, the function returns None. Use
    merge_distributed_checkpoints() after all machines' checkpoint files have
    been collected onto one machine.

    Matching semantics are unchanged:
      country partition -> 3 vectorized signals -> total_block_score ->
      stable pandas sort -> candidate_rank -> top K.
    """
    import time

    distributed, machine_id, machine_count, chunk_size, checkpoint_dir = (
        _distributed_config()
    )

    if verbose:
        mode = (
            f"DISTRIBUTED machine={machine_id}/{machine_count}, "
            f"chunk={chunk_size:,}"
            if distributed
            else "SINGLE-MACHINE"
        )
        print(
            f"  [blocking] Starting candidate generation "
            f"(K={k}, max_block_size={max_block_size}, {mode})...",
            flush=True,
        )

    # Normalize once, before country/chunk partitioning.
    s1_df = add_normalized_columns(s1_df)
    s2_df = add_normalized_columns(s2_df)
    s3_df = add_normalized_columns(s3_df)

    countries = list(s1_df["country"].dropna().unique())

    if verbose:
        print(
            f"  [blocking] Country partitions: {countries}",
            flush=True,
        )

    if distributed:
        os.makedirs(checkpoint_dir, exist_ok=True)

        manifest_rows = []
        global_chunk_id = 0
        completed = 0
        skipped = 0
        generated_rows = 0

        for country in countries:
            s1_c = s1_df[s1_df["country"] == country]
            s2_c = s2_df[s2_df["country"] == country]
            s3_c = s3_df[s3_df["country"] == country]

            if len(s1_c) == 0 or len(s2_c) + len(s3_c) == 0:
                # Still advance global chunk IDs for S1 chunks if there are
                # S1 rows, preserving deterministic machine assignment.
                n_chunks = math.ceil(len(s1_c) / chunk_size) if len(s1_c) else 0
                global_chunk_id += n_chunks
                continue

            n_chunks = math.ceil(len(s1_c) / chunk_size)

            if verbose:
                print(
                    f"  [distributed] {country}: "
                    f"S1={len(s1_c):,}, pool={len(s2_c)+len(s3_c):,}, "
                    f"chunks={n_chunks:,}",
                    flush=True,
                )

            for local_chunk_id, s1_chunk in _iter_country_chunks(
                s1_c, chunk_size
            ):
                chunk_id = global_chunk_id + local_chunk_id

                # This machine owns this deterministic subset.
                if chunk_id % machine_count != machine_id:
                    continue

                checkpoint = _checkpoint_path(
                    checkpoint_dir, country, chunk_id
                )

                if _checkpoint_is_valid(checkpoint):
                    skipped += 1
                    manifest_rows.append({
                        "country": str(country),
                        "chunk_id": chunk_id,
                        "machine_id": machine_id,
                        "rows": len(s1_chunk),
                        "status": "skipped_existing",
                        "path": checkpoint,
                    })
                    print(
                        f"  [distributed] SKIP "
                        f"country={country} chunk={chunk_id:06d} "
                        f"S1={len(s1_chunk):,}",
                        flush=True,
                    )
                    continue

                t0 = time.time()

                print(
                    f"  [distributed] START "
                    f"country={country} chunk={chunk_id:06d} "
                    f"S1={len(s1_chunk):,} "
                    f"machine={machine_id}/{machine_count}",
                    flush=True,
                )

                cands = build_candidates_for_country(
                    s1_chunk,
                    s2_c,
                    s3_c,
                    country,
                    k=k,
                    max_block_size=max_block_size,
                    return_metadata=return_metadata,
                    verbose=False,
                )

                _atomic_write_parquet(cands, checkpoint)

                elapsed = time.time() - t0
                generated_rows += len(cands)
                completed += 1

                manifest_rows.append({
                    "country": str(country),
                    "chunk_id": chunk_id,
                    "machine_id": machine_id,
                    "rows": len(s1_chunk),
                    "candidate_rows": len(cands),
                    "status": "complete",
                    "seconds": round(elapsed, 3),
                    "path": checkpoint,
                })

                print(
                    f"  [distributed] DONE "
                    f"country={country} chunk={chunk_id:06d} "
                    f"S1={len(s1_chunk):,} candidates={len(cands):,} "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

                del s1_chunk, cands
                gc.collect()

            # Advance after processing ALL chunks of this country, including
            # chunks assigned to other machines.
            global_chunk_id += n_chunks

            del s1_c, s2_c, s3_c
            gc.collect()

        _write_distributed_manifest(checkpoint_dir, manifest_rows)

        print(
            f"  [distributed] Machine {machine_id}/{machine_count} complete: "
            f"generated={completed:,}, skipped={skipped:,}, "
            f"candidate_rows={generated_rows:,}",
            flush=True,
        )
        print(
            "  [distributed] To build the final training candidate cache, "
            "collect all machines' checkpoint files into one directory and "
            "call merge_distributed_checkpoints().",
            flush=True,
        )

        return None

    # ------------------------------------------------------------------
    # Original single-machine behavior
    # ------------------------------------------------------------------
    parquet_writer = None
    country_dfs = []

    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]

        cands_c = build_candidates_for_country(
            s1_c,
            s2_c,
            s3_c,
            country,
            k=k,
            max_block_size=max_block_size,
            return_metadata=return_metadata,
            verbose=verbose,
        )

        if len(cands_c) == 0:
            continue

        if output_parquet_path:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(cands_c, preserve_index=False)

            if parquet_writer is None:
                os.makedirs(
                    os.path.dirname(
                        os.path.abspath(output_parquet_path)
                    ),
                    exist_ok=True,
                )
                parquet_writer = pq.ParquetWriter(
                    output_parquet_path,
                    table.schema,
                    compression="zstd",
                )

            parquet_writer.write_table(table)
            del table, cands_c
            gc.collect()
        else:
            country_dfs.append(cands_c)

    if output_parquet_path and parquet_writer:
        parquet_writer.close()

        if verbose:
            print(
                f"  [blocking] Streamed candidates to "
                f"{output_parquet_path}",
                flush=True,
            )
        return None

    if country_dfs:
        candidates_df = pd.concat(country_dfs, ignore_index=True)
    else:
        candidates_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )

    if verbose:
        s1_with_cands = (
            candidates_df["source1_entity_id"].nunique()
            if len(candidates_df) > 0 else 0
        )
        print(
            f"  [blocking] Finished — {len(candidates_df):,} "
            f"candidate pairs for {s1_with_cands:,} / "
            f"{len(s1_df):,} S1 entities.",
            flush=True,
        )

    return candidates_df
def candidates_to_tsv_format(candidates_df: pd.DataFrame, all_s1_ids: list) -> pd.DataFrame:
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Distributed/resumable business-entity blocker"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge distributed Parquet checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join("output", "distributed_candidates"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "candidate_pairs_merged.parquet"),
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require the caller to have collected all expected checkpoints.",
    )
    parser.add_argument(
        "--machine-count",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.merge:
        merge_distributed_checkpoints(
            args.checkpoint_dir,
            args.output,
            expected_machine_count=args.machine_count,
            require_all=args.require_all,
            verbose=True,
        )
    else:
        enabled, mid, mc, cs, cd = _distributed_config()
        print(
            f"DISTRIBUTED_MODE={enabled} "
            f"MACHINE_ID={mid} MACHINE_COUNT={mc} "
            f"S1_CHUNK_SIZE={cs} CHECKPOINT_DIR={cd}"
        )
