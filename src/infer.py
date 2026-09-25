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

import os
import gc
import sys
import math
import argparse
from collections import defaultdict
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_source
from blocking import build_candidates_for_country
from features import compute_features_chunk, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Memory-Safe Streaming Test Inference")
    parser.add_argument("--data-dir", default="dataset/test", help="Path to test data directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--k", type=int, default=50, help="Candidate top-K for frozen blocker")
    parser.add_argument("--threshold", type=float, default=0.76, help="Decision threshold (frozen at 0.76)")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Feature extraction chunk size")
    args = parser.parse_args()

    print(f"Loading test sources from {args.data_dir}...")
    s1 = load_source(f"{args.data_dir}/test_source1.tsv")
    s2 = load_source(f"{args.data_dir}/test_source2.tsv")
    s3 = load_source(f"{args.data_dir}/test_source3.tsv")

    print("Loading trained model and optimal threshold...")
    saved = joblib.load("model.joblib")
    model = saved["model"]
    threshold = args.threshold if args.threshold is not None else saved.get("threshold", 0.76)
    print(f"Loaded LightGBM model (decision threshold = {threshold:.2f}, frozen K={args.k})")

    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = f"{args.output_dir}/candidate_pairs.tsv"
    match_path = f"{args.output_dir}/matching_results.tsv"

    countries = s1["country"].unique()
    print(f"Streaming inference across {len(countries)} country partition(s)...")

    total_s1 = 0
    total_cands = 0
    total_matched_s1 = 0

    with open(cand_path, "w", encoding="utf-8") as f_cand, open(match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for country in countries:
            s1_c = s1[s1["country"] == country]
            s2_c = s2[s2["country"] == country]
            s3_c = s3[s3["country"] == country]
            s1_c_ids = s1_c["entity_id"].tolist()

            if len(s1_c) == 0:
                continue

            print(f"  [country: {country}] S1={len(s1_c):,}, Pool={len(s2_c)+len(s3_c):,}...")

            # 1. Generate top-K candidates for this country partition
            cands_c = build_candidates_for_country(
                s1_c, s2_c, s3_c, country, k=args.k, return_metadata=True, verbose=False
            )
            total_cands += len(cands_c)
            total_s1 += len(s1_c)

            # 2. Write ALL generated candidates to candidate_pairs.tsv incrementally
            cands_by_s1 = defaultdict(list)
            if len(cands_c) > 0:
                for r in cands_c[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
                    cands_by_s1[r.source1_entity_id].append(r.candidate_entity_id)

            for s1_id in s1_c_ids:
                c_list = ",".join(dict.fromkeys(cands_by_s1.get(s1_id, [])))
                f_cand.write(f"{s1_id}\t{c_list}\n")

            # 3. Country-scoped feature extraction & immediate model scoring
            country_matches = defaultdict(list)
            if len(cands_c) > 0:
                needed_cands = set(cands_c["candidate_entity_id"])
                pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
                pool_filtered = pool_c[pool_c["entity_id"].isin(needed_cands)].drop_duplicates(subset="entity_id")

                s1_lookup_c = s1_c.drop_duplicates(subset="entity_id").set_index("entity_id").to_dict("index")
                other_lookup_c = pool_filtered.set_index("entity_id").to_dict("index")
                del pool_c, pool_filtered

                num_chunks = math.ceil(len(cands_c) / args.chunk_size)
                for ch in range(num_chunks):
                    start = ch * args.chunk_size
                    end = min(start + args.chunk_size, len(cands_c))
                    chunk_slice = cands_c.iloc[start:end]

                    chunk_feat = compute_features_chunk(chunk_slice, s1_lookup_c, other_lookup_c)
                    probas = model.predict_proba(chunk_feat[FEATURE_COLUMNS])[:, 1]
                    pos_mask = probas >= threshold

                    if pos_mask.any():
                        pos_df = chunk_feat[pos_mask]
                        for r in pos_df[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
                            country_matches[r.source1_entity_id].append(r.candidate_entity_id)

                    del chunk_feat, probas, pos_mask
                    gc.collect()

                del s1_lookup_c, other_lookup_c
                gc.collect()

            # 4. Write matching results to matching_results.tsv (preserving singletons)
            for s1_id in s1_c_ids:
                m_list = ",".join(dict.fromkeys(country_matches.get(s1_id, [])))
                f_match.write(f"{s1_id}\t{m_list}\n")
                if m_list:
                    total_matched_s1 += 1

            # 5. Release country partition memory
            del s1_c, s2_c, s3_c, cands_c, cands_by_s1, country_matches
            gc.collect()

    print(f"\nInference complete:")
    print(f"  Total S1 entities processed: {total_s1:,}")
    print(f"  Total candidate pairs generated: {total_cands:,}")
    print(f"  S1 entities with matched pairs: {total_matched_s1:,} ({total_matched_s1/max(total_s1,1):.2%})")
    print(f"  Singleton S1 entities: {total_s1 - total_matched_s1:,} ({(total_s1 - total_matched_s1)/max(total_s1,1):.2%})")
    print(f"  Wrote {cand_path}")
    print(f"  Wrote {match_path}")

    print("\nNext step: run the organizer's validator before submitting:")
    print(f"  python3 utils/validate_submission.py --matching {match_path} "
          f"--candidate {cand_path} --test-dir {args.data_dir}")


if __name__ == "__main__":
    main()

