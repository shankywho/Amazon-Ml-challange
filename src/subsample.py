"""
subsample.py
-------------
Creates a SMALL slice of the training data so you can test the whole
pipeline (blocking -> features -> train -> threshold tuning) in minutes
instead of committing to a run against all 2.2M+ S1 records.

WHY THIS MATTERS AT YOUR DATASET SIZE:
You have 2.2M S1 records, 5M S2 records, 5.3M S3 records. Even with the
scalable blocking rewrite, you do NOT want your first-ever run of this
pipeline to be against the full dataset — if there's a bug, you find out
after waiting a long time instead of after a minute. Always validate on
a subsample first, then scale up once you've confirmed everything works
and looked at the blocking recall check.

USAGE:
  python3 src/subsample.py --n-s1 5000

This reads dataset/train/*.tsv and writes a smaller copy to
dataset/train_sample/*.tsv. Point train.py at it with --data-dir:

  python3 src/train.py --data-dir dataset/train_sample
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-s1", type=int, default=5000,
                         help="Number of Source 1 entities to sample")
    parser.add_argument("--input-dir", default="dataset/train")
    parser.add_argument("--output-dir", default="dataset/train_sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading full training data (this part is unavoidably slow the first time)...")
    s1 = pd.read_csv(f"{args.input_dir}/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{args.input_dir}/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{args.input_dir}/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{args.input_dir}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    # Sample S1 entities — mix of ones WITH matches and WITHOUT (singletons),
    # so your validation set actually has positive examples to learn/test threshold on.
    gt["has_match"] = gt["matched_entity_ids"].str.strip() != ""
    with_matches = gt[gt.has_match]["source1_entity_id"]
    without_matches = gt[~gt.has_match]["source1_entity_id"]

    n_with = min(int(args.n_s1 * 0.6), len(with_matches))
    n_without = min(args.n_s1 - n_with, len(without_matches))

    sampled_ids = pd.concat([
        with_matches.sample(n=n_with, random_state=args.seed),
        without_matches.sample(n=n_without, random_state=args.seed),
    ])
    print(f"Sampled {len(sampled_ids)} S1 entities ({n_with} with matches, {n_without} singletons)")

    s1_sample = s1[s1.entity_id.isin(sampled_ids)]
    gt_sample = gt[gt.source1_entity_id.isin(sampled_ids)].drop(columns=["has_match"])

    # Keep the FULL S2/S3 pool — blocking needs to search the real pool to be
    # a meaningful test. Only S1 (the query side) is subsampled.
    s1_sample.to_csv(f"{args.output_dir}/train_source1.tsv", sep="\t", index=False)
    s2.to_csv(f"{args.output_dir}/train_source2.tsv", sep="\t", index=False)
    s3.to_csv(f"{args.output_dir}/train_source3.tsv", sep="\t", index=False)
    gt_sample.to_csv(f"{args.output_dir}/train_ground_truth.tsv", sep="\t", index=False)

    print(f"Wrote subsample to {args.output_dir}/")
    print(f"Now run: python3 src/train.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
