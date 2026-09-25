"""
infer.py
--------
Runs the full pipeline on the TEST set using the model trained by
train.py, and writes the two files the competition requires:

  output/candidate_pairs.tsv   -- the blocking shortlist (audit file)
  output/matching_results.tsv  -- your actual scored predictions

"Inference" just means: use the already-trained model to make
predictions on new data it has never seen (the test set), as opposed
to "training" where the model was learning from labeled examples.
"""

import sys
import argparse
import pandas as pd
import joblib

sys.path.insert(0, "src")
from common import load_source
from blocking import build_candidates, candidates_to_tsv_format
from features import compute_features, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading test sources from {args.data_dir}...")
    s1 = load_source(f"{args.data_dir}/test_source1.tsv")
    s2 = load_source(f"{args.data_dir}/test_source2.tsv")
    s3 = load_source(f"{args.data_dir}/test_source3.tsv")

    print("Loading trained model...")
    saved = joblib.load("model.joblib")
    model, threshold = saved["model"], saved["threshold"]
    print(f"Using threshold={threshold}")

    print("Building candidate pairs on test data...")
    candidates = build_candidates(s1, s2, s3, k=args.k)

    # Write the audit file: candidate_pairs.tsv
    all_s1_ids = s1["entity_id"].tolist()
    cand_tsv = candidates_to_tsv_format(candidates, all_s1_ids)
    cand_path = f"{args.output_dir}/candidate_pairs.tsv"
    cand_tsv.to_csv(cand_path, sep="\t", index=False)
    print(f"Wrote {cand_path} ({len(cand_tsv)} S1 entities)")

    print("Computing features for test candidate pairs...")
    s1_lookup = s1.drop_duplicates(subset="entity_id").set_index("entity_id").to_dict("index")
    other = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset="entity_id")
    other_lookup = other.set_index("entity_id").to_dict("index")
    feat_df = compute_features(candidates, s1_lookup, other_lookup)

    print("Scoring pairs with the trained model...")
    feat_df["proba"] = model.predict_proba(feat_df[FEATURE_COLUMNS])[:, 1]

    # Keep only pairs the model is confident enough about (>= tuned threshold).
    # Everything below the threshold is dropped -> that S1 entity becomes a
    # singleton (empty match list) unless another candidate clears the bar.
    matches = feat_df[feat_df["proba"] >= threshold]

    grouped = (
        matches.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    full = pd.DataFrame({"source1_entity_id": all_s1_ids})
    full = full.merge(grouped, on="source1_entity_id", how="left")
    full["matched_entity_ids"] = full["matched_entity_ids"].fillna("")

    match_path = f"{args.output_dir}/matching_results.tsv"
    full.to_csv(match_path, sep="\t", index=False)
    print(f"Wrote {match_path} ({len(full)} S1 entities, "
          f"{(full.matched_entity_ids != '').sum()} with at least one match)")

    print("\nNext step: run the organizer's validator before submitting:")
    print("  python3 utils/validate_submission.py --matching output/matching_results.tsv "
          "--candidate output/candidate_pairs.tsv --test-dir dataset/test")


if __name__ == "__main__":
    main()
