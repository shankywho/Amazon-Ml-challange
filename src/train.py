"""
train.py
--------
Training & Evaluation Pipeline for Business Entity Resolution.

Phases implemented:
1. Training Pair Construction (using frozen K=50 blocker)
2. Hard Negative Sampling (all positives + top-ranked hard negatives + sampled negatives)
3. 26-feature Engineering (RapidFuzz name/address metrics + blocking IDF scores & rank)
4. Memory-safe chunked feature generation
5. Leak-free GroupShuffleSplit grouped by source1_entity_id
6. LightGBM binary classifier training with early stopping
7. Exact macro-averaged F_0.5 evaluation over all S1 entities (including singletons)
8. Comprehensive entity-level decision analysis & blocking vs model miss attribution
"""

import os
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_source
from blocking import build_candidates
from features import compute_features, FEATURE_COLUMNS


def parse_ground_truth(gt_df: pd.DataFrame) -> tuple[set, dict]:
    """Parse ground truth into a pair-set and a per-entity dict of sets."""
    true_pairs = set()
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matches = row.matched_entity_ids
        match_set = set()
        if matches and str(matches).strip() != "" and str(matches).lower() != "nan":
            for m in str(matches).split(","):
                m = m.strip()
                if m:
                    true_pairs.add((s1_id, m))
                    match_set.add(m)
        gt_dict[s1_id] = match_set
    return true_pairs, gt_dict


def sample_hard_negatives(
    candidates_df: pd.DataFrame,
    true_pairs: set,
    neg_per_pos: float = 4.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Construct high-quality training pairs:
    1. ALL positive pairs from blocker
    2. Hard negatives from top-ranked candidates
    3. Additional sampled negatives from lower ranks
    4. Guard against singletons by keeping top 2 hard negatives for zero-match entities
    """
    rng = np.random.default_rng(random_state)
    candidates_df = candidates_df.copy()

    # Determine ground truth label
    pair_tuples = list(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    candidates_df["label"] = np.array([int(p in true_pairs) for p in pair_tuples], dtype=np.int32)

    positives = candidates_df[candidates_df["label"] == 1]
    negatives = candidates_df[candidates_df["label"] == 0]

    num_pos = len(positives)
    target_negs = int(num_pos * neg_per_pos)
    print(f"  [sampling] Total candidate pairs: {len(candidates_df):,} | Positives: {num_pos:,} | Available negatives: {len(negatives):,}")

    selected_neg_indices = set()

    # Strategy: group negatives by S1 entity
    # Prioritize top ranks (hardest negatives)
    for s1_id, group in negatives.groupby("source1_entity_id"):
        # Take top 2 hardest negatives for each entity
        sorted_indices = group.sort_values("candidate_rank").index.tolist()
        top_hard = sorted_indices[:2]
        selected_neg_indices.update(top_hard)

    # If we need more negatives to reach target_negs, sample uniformly from remaining
    remaining_indices = list(set(negatives.index) - selected_neg_indices)
    needed = target_negs - len(selected_neg_indices)
    if needed > 0 and len(remaining_indices) > 0:
        sample_size = min(needed, len(remaining_indices))
        sampled = rng.choice(remaining_indices, size=sample_size, replace=False)
        selected_neg_indices.update(sampled)

    final_negatives = negatives.loc[list(selected_neg_indices)]
    training_pairs = pd.concat([positives, final_negatives], ignore_index=True)
    training_pairs = training_pairs.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"  [sampling] Sampled {len(final_negatives):,} negatives ({len(final_negatives)/max(num_pos,1):.2f}x pos) | Total training rows: {len(training_pairs):,}")
    return training_pairs


def evaluate_macro_f05(
    val_scored_df: pd.DataFrame,
    val_gt_dict: dict,
    all_val_s1_ids: list,
    threshold: float,
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """
    Computes exact entity-level macro-averaged F_0.5 across all validation S1 entities.
    Singleton entities with predicted empty match receive 1.0; with false matches receive 0.0.
    """
    # Filter predictions above threshold
    preds_above = val_scored_df[val_scored_df["proba"] >= threshold]
    preds_by_s1 = preds_above.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    scores = []
    precisions = []
    recalls = []
    beta2 = beta ** 2

    for s1_id in all_val_s1_ids:
        pred_set = preds_by_s1.get(s1_id, set())
        true_set = val_gt_dict.get(s1_id, set())

        len_p = len(pred_set)
        len_t = len(true_set)

        if len_p == 0 and len_t == 0:
            # Singleton correctly predicted as empty
            scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        if len_p > 0 and len_t == 0:
            # False merge on singleton
            scores.append(0.0)
            precisions.append(0.0)
            recalls.append(1.0)
            continue
        if len_p == 0 and len_t > 0:
            # Missed all matches
            scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(pred_set & true_set)
        fp = len_p - tp
        fn = len_t - tp

        p = tp / len_p if len_p > 0 else 0.0
        r = tp / len_t if len_t > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

        if p == 0.0 or r == 0.0:
            scores.append(0.0)
        else:
            f = (1.0 + beta2) * p * r / (beta2 * p + r)
            scores.append(f)

    return float(np.mean(scores)), float(np.mean(precisions)), float(np.mean(recalls))


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution LightGBM Model Trainer")
    parser.add_argument("--data-dir", default="dataset/train_sample", help="Path to training data directory")
    parser.add_argument("--k", type=int, default=50, help="Candidate top-K for frozen blocker")
    parser.add_argument("--neg-ratio", type=float, default=4.0, help="Hard negatives per positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-out", default="model.joblib", help="Output path for trained model & metadata")
    parser.add_argument("--distributed-blocking", action="store_true", help="Run distributed candidate generation only and exit")
    parser.add_argument("--machine-id", type=int, default=0, help="0-based ID of this machine in the cluster")
    parser.add_argument("--machine-count", type=int, default=1, help="Total number of participating machines")
    parser.add_argument("--chunk-size", type=int, default=25000, help="S1 chunk size for distributed candidate generation")
    parser.add_argument("--checkpoint-dir", default="output/distributed_candidates", help="Directory to write chunk checkpoints")
    args = parser.parse_args()

    if args.distributed_blocking:
        if args.machine_count < 1:
            parser.error("--machine-count must be >= 1")
        if args.machine_id < 0 or args.machine_id >= args.machine_count:
            parser.error(f"--machine-id must be between 0 and {args.machine_count - 1}")
        if args.chunk_size <= 0:
            parser.error("--chunk-size must be > 0")

    t_start = time.time()
    print("=" * 70)
    print("PHASE 1: TRAINING DATA & CANDIDATE GENERATION (FROZEN K=50)")
    print("=" * 70)
    s1 = load_source(f"{args.data_dir}/train_source1.tsv")
    s2 = load_source(f"{args.data_dir}/train_source2.tsv")
    s3 = load_source(f"{args.data_dir}/train_source3.tsv")
    print(f"Loaded {len(s1):,} S1 entities | {len(s2):,} S2 | {len(s3):,} S3")

    if args.distributed_blocking:
        os.environ["DISTRIBUTED_MODE"] = "1"
        os.environ["MACHINE_ID"] = str(args.machine_id)
        os.environ["MACHINE_COUNT"] = str(args.machine_count)
        os.environ["S1_CHUNK_SIZE"] = str(args.chunk_size)
        os.environ["CHECKPOINT_DIR"] = str(args.checkpoint_dir)

        print("\n" + "=" * 70)
        print("[DISTRIBUTED BLOCKING]")
        print(f"Machine: {args.machine_id} / {args.machine_count}")
        print(f"Chunk size: {args.chunk_size:,}")
        print(f"Checkpoint directory: {args.checkpoint_dir}")
        print("=" * 70)

        t_block = time.time()
        build_candidates(
            s1,
            s2,
            s3,
            k=args.k,
            return_metadata=True,
            output_parquet_path=None,
            verbose=True,
        )
        print(f"\n[DISTRIBUTED BLOCKING] Machine {args.machine_id}/{args.machine_count} complete in {time.time() - t_block:.2f}s.")
        sys.exit(0)

    gt_df = pd.read_csv(f"{args.data_dir}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    all_s1_ids = s1["entity_id"].tolist()
    true_pairs, gt_dict = parse_ground_truth(gt_df)
    singleton_count = sum(1 for s in all_s1_ids if len(gt_dict.get(s, set())) == 0)

    print(f"Ground truth match pairs: {len(true_pairs):,} | Singletons: {singleton_count:,} ({singleton_count/len(s1):.2%})")

    # Candidate generation with parquet caching
    cache_cand_path = f"{args.data_dir}/candidates_k{args.k}.parquet"
    if os.path.exists(cache_cand_path):
        print(f"  [blocking] Loading cached candidates from {cache_cand_path}...")
        candidates = pd.read_parquet(cache_cand_path)
        block_runtime = 0.0
    else:
        t_block = time.time()
        build_candidates(s1, s2, s3, k=args.k, return_metadata=True, output_parquet_path=cache_cand_path, verbose=True)
        block_runtime = time.time() - t_block
        candidates = pd.read_parquet(cache_cand_path)
        print(f"  [blocking] Saved candidate cache to {cache_cand_path}")

    # Blocking recall check
    cand_pair_set = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    found_true_pairs = cand_pair_set & true_pairs
    block_recall = len(found_true_pairs) / max(len(true_pairs), 1)
    block_misses = len(true_pairs) - len(found_true_pairs)
    print(f"\n[BLOCKING SUMMARY K={args.k}]")
    print(f"  Recall: {block_recall:.2%} ({len(found_true_pairs):,} / {len(true_pairs):,} true matches)")
    print(f"  Blocking Misses: {block_misses:,} ({block_misses/len(true_pairs):.2%})")
    print(f"  Total Candidate Pairs: {len(candidates):,}")
    print(f"  Avg candidates per S1: {len(candidates)/len(s1):.2f}")
    if block_runtime > 0:
        print(f"  Blocking Runtime: {block_runtime:.2f}s")

    print("\n" + "=" * 70)
    print("PHASE 2: HARD NEGATIVE SAMPLING")
    print("=" * 70)
    sampled_pairs = sample_hard_negatives(candidates, true_pairs, neg_per_pos=args.neg_ratio, random_state=args.seed)

    # Free raw candidate pairs from memory immediately
    del candidates, cand_pair_set
    import gc
    gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 3 & 4: FEATURE ENGINEERING")
    print("=" * 70)
    cache_feat_path = f"{args.data_dir}/features_k{args.k}_neg{int(args.neg_ratio)}_{len(FEATURE_COLUMNS)}feat.parquet"
    use_cached_features = False
    if os.path.exists(cache_feat_path):
        print(f"  [features] Checking cached features from {cache_feat_path}...")
        cached_df = pd.read_parquet(cache_feat_path)
        actual_rows = len(cached_df)
        expected_rows = len(sampled_pairs)
        cached_cols = set(cached_df.columns)
        schema_valid = all(col in cached_cols for col in FEATURE_COLUMNS) and ("label" in cached_cols)
        if actual_rows == expected_rows and schema_valid:
            feature_df = cached_df
            feat_runtime = 0.0
            use_cached_features = True
            print(f"  [features] Cache VALID: {actual_rows:,} rows with complete schema ({len(FEATURE_COLUMNS)} features)")
        else:
            reason = []
            if actual_rows != expected_rows:
                reason.append(f"rows mismatch ({actual_rows:,} != {expected_rows:,})")
            if not schema_valid:
                missing = [c for c in FEATURE_COLUMNS if c not in cached_cols]
                reason.append(f"schema mismatch (missing {len(missing)} features)")
            print(f"  [features] STALE CACHE DETECTED: {', '.join(reason)}. Regenerating...")
            del cached_df
            try:
                os.remove(cache_feat_path)
            except OSError:
                pass

    if not use_cached_features:
        t_feat = time.time()
        from features import compute_features_country_partitioned
        compute_features_country_partitioned(
            sampled_pairs, s1, s2, s3,
            output_parquet_path=cache_feat_path,
            chunk_size=100000,
            verbose=True,
        )
        feature_df = pd.read_parquet(cache_feat_path)
        feat_runtime = time.time() - t_feat

    print(f"  [features] {len(FEATURE_COLUMNS)} features for {len(feature_df):,} rows (runtime: {feat_runtime:.2f}s)")

    # Free raw source DataFrames and sampled pairs now that features are computed
    del s1, s2, s3, sampled_pairs
    gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 5: LEAK-FREE GROUPED VALIDATION SPLIT")
    print("=" * 70)
    # Split by source1_entity_id so all candidates for an entity stay strictly in train or val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, val_idx = next(gss.split(feature_df, groups=feature_df["source1_entity_id"]))

    train_df = feature_df.iloc[train_idx].copy()
    val_df = feature_df.iloc[val_idx].copy()
    del feature_df
    gc.collect()

    val_s1_entities = list(val_df["source1_entity_id"].unique())
    print(f"  Train: {len(train_df):,} pairs ({train_df.label.sum():,} pos, {(train_df.label==0).sum():,} neg) across {train_df.source1_entity_id.nunique():,} S1 entities")
    print(f"  Val:   {len(val_df):,} pairs ({val_df.label.sum():,} pos, {(val_df.label==0).sum():,} neg) across {len(val_s1_entities):,} S1 entities")

    print("\n" + "=" * 70)
    print("PHASE 6: LIGHTGBM TRAINING (EARLY STOPPING)")
    print("=" * 70)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=31,
        max_bin=63,
        n_jobs=-1,
        random_state=args.seed,
    )

    t_train = time.time()
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=callbacks,
    )
    train_runtime = time.time() - t_train
    best_iteration = model.best_iteration_ if hasattr(model, "best_iteration_") else model.n_estimators

    val_df["proba"] = model.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, val_df["proba"])
    val_pr_auc = average_precision_score(y_val, val_df["proba"])

    print(f"  Model trained in {train_runtime:.2f}s (best iteration: {best_iteration})")
    print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
    print(f"  Validation PR-AUC:  {val_pr_auc:.4f}")

    print("\n" + "=" * 70)
    print("PHASE 7: EXACT MACRO F_0.5 THRESHOLD SWEEP")
    print("=" * 70)
    # Complete threshold sweep: 0.10 -> 0.99, step 0.01
    thresholds = [round(t, 2) for t in np.arange(0.10, 1.00, 0.01)]
    threshold_results = []

    best_thresh = 0.50
    best_f05 = -1.0
    best_p = 0.0
    best_r = 0.0

    for t in thresholds:
        f05, prec, rec = evaluate_macro_f05(val_df, gt_dict, val_s1_entities, threshold=t)
        threshold_results.append({"threshold": t, "macro_f05": f05, "precision": prec, "recall": rec})
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            best_p = prec
            best_r = rec

    top10_results = sorted(threshold_results, key=lambda x: x["macro_f05"], reverse=True)[:10]
    print(f"Top 10 Thresholds by Macro F_0.5:")
    for r in top10_results:
        marker = " <--- BEST" if r["threshold"] == best_thresh else ""
        print(f"  Threshold={r['threshold']:.2f} | Macro F_0.5={r['macro_f05']:.4f} | Precision={r['precision']:.4f} | Recall={r['recall']:.4f}{marker}")

    print(f"\nOPTIMAL DECISION THRESHOLD: {best_thresh:.2f} (Macro F_0.5 = {best_f05:.4f})")

    print("\n" + "=" * 70)
    print("PHASE 8 & 9: ENTITY-LEVEL DECISION & FAILURE ATTRIBUTION ANALYSIS")
    print("=" * 70)
    # Val set true matches
    val_true_pairs = [(s, m) for s in val_s1_entities for m in gt_dict.get(s, set())]
    total_val_true = len(val_true_pairs)
    val_cand_pairs = set(zip(val_df["source1_entity_id"], val_df["candidate_entity_id"]))
    val_cand_matches = [p for p in val_true_pairs if p in val_cand_pairs]
    val_blocking_misses = total_val_true - len(val_cand_matches)
    val_blocking_recall = len(val_cand_matches) / max(total_val_true, 1)

    # Model predictions at optimal threshold
    val_preds = val_df[val_df["proba"] >= best_thresh]
    pred_val_pairs = set(zip(val_preds["source1_entity_id"], val_preds["candidate_entity_id"]))
    val_true_pair_set = set(val_true_pairs)

    val_correct_matches = pred_val_pairs & val_true_pair_set
    val_false_positives = pred_val_pairs - val_true_pair_set

    # Model misses: True pairs present in candidate set, but rejected by model (proba < threshold)
    model_misses = [p for p in val_cand_matches if p not in pred_val_pairs]
    cond_model_recall = len(val_correct_matches) / max(len(val_cand_matches), 1)

    # Singleton false alarms
    val_singletons = [s for s in val_s1_entities if len(gt_dict.get(s, set())) == 0]
    preds_by_s1 = val_preds.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
    singleton_false_positives = sum(1 for s in val_singletons if s in preds_by_s1)

    print(f"\n[CRITICAL FAILURE ATTRIBUTION]")
    print(f"  Total True Matches in Validation Split: {total_val_true:,}")
    print(f"  A. Blocking Misses (never reached model): {val_blocking_misses:,} ({val_blocking_misses/max(total_val_true,1):.2%})")
    print(f"  B. Model Misses (present in candidates, but rejected): {len(model_misses):,} ({len(model_misses)/max(total_val_true,1):.2%})")
    print(f"  C. Correctly Matched Pairs: {len(val_correct_matches):,} ({len(val_correct_matches)/max(total_val_true,1):.2%})")
    print(f"\n[RECALL DECOMPOSITION]")
    print(f"  Blocking Recall: {val_blocking_recall:.2%}")
    print(f"  Model Recall (conditional on candidate presence): {cond_model_recall:.2%}")
    print(f"  End-to-End True Match Recall: {len(val_correct_matches)/max(total_val_true,1):.2%}")
    print(f"\n[ERROR ANALYSIS]")
    print(f"  Total False Positive Pairs: {len(val_false_positives):,}")
    print(f"  Validation Singletons: {len(val_singletons):,}")
    print(f"  Singleton False Positives (predicted match for singleton): {singleton_false_positives:,} ({singleton_false_positives/max(len(val_singletons),1):.2%})")

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(f"\nTop 10 Most Important Features:")
    for feat_name, imp in importances.head(10).items():
        print(f"  {feat_name:25s}: {imp:6d}")

    # Save model + optimal threshold + feature names
    joblib.dump({
        "model": model,
        "threshold": best_thresh,
        "features": FEATURE_COLUMNS,
        "val_macro_f05": best_f05,
        "k": args.k,
    }, args.model_out)
    print(f"\nSaved model & calibration to {args.model_out}")
    print(f"Total modeling pipeline completed in {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
