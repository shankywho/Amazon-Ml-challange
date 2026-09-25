# Business Entity Resolution — Starter Pipeline

This is a working scaffold for the Amazon ML Challenge 2026 problem. It's built
so you can run it end-to-end even if you've never trained an ML model before —
every file has comments explaining WHAT it's doing and WHY, not just code.

## Quick concept primer (read this first if ML is new to you)

- **Blocking**: cheaply narrowing millions of possible pairs down to a short
  candidate list per entity, using fast text similarity (not the real model).
- **Feature engineering**: turning text pairs into numbers (similarity scores)
  a model can actually learn from.
- **Training**: showing the model examples of (features → correct answer) so
  it learns a pattern. Here it's a LightGBM classifier — decision trees, not
  a neural network. It trains in minutes on your CPU, no GPU needed.
- **Validation set**: data held back from training so you can honestly check
  if the model generalizes, or just memorized.
- **Threshold**: the model outputs a confidence score (0 to 1) per pair, not
  a yes/no. We pick the cutoff score above which we call it a "match" — we
  don't just use 0.5, we test several cutoffs and pick whichever actually
  scores best on your validation data.

## Folder structure expected

```
business_entity_resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── src/
│   ├── common.py       # text normalization + loading helpers
│   ├── blocking.py      # candidate generation (TF-IDF + nearest neighbors)
│   ├── features.py      # pairwise similarity feature engineering
│   ├── train.py         # model training + threshold tuning
│   └── infer.py          # runs everything on test data, writes outputs
├── output/               # matching_results.tsv + candidate_pairs.tsv land here
└── requirements.txt
```

**You need to drop the actual competition data into `dataset/train/` and
`dataset/test/` yourself** (matching the exact filenames above) — this
scaffold doesn't include the real data.

## How to run it

**IMPORTANT — if your dataset is large (hundreds of thousands+ rows),
always test on a subsample first, never your first run against the full
data.** A bug is much cheaper to discover after 30 seconds than after an
hour.

## Memory-Safe Execution Architecture (Google Colab & Production)

The pipeline is engineered to process massive datasets (14.7M train rows, 11.7M test rows) within standard **12.7 GB RAM** Colab runtimes by replacing global materialization with streaming country-partitioned execution.

### Key Architectural Safeguards:
1. **Dynamic Country Partitioning**:
   - The blocker evaluates records per country independently (`India`, `US`, etc.).
   - Inverted indexes (`name_index`, `addr_index`, `pref_index`) are scoped strictly to the current country and released via explicit `gc.collect()` before moving to the next country.
   - Zero global candidate dictionaries or 100M-row Python lists in RAM.
2. **Filtered Country-Scoped Lookups**:
   - Instead of building a global 10.3M-entity Python dictionary (~6 GB RAM), lookups are built strictly per country and pre-filtered to the unique candidate IDs retrieved by the blocker (~10–20 MB RAM).
3. **On-The-Fly Inference Scoring**:
   - `src/infer.py` streams candidates directly to `output/candidate_pairs.tsv`.
   - Feature engineering runs in 100,000-pair chunks that are scored immediately by the trained LightGBM model and discarded.
   - Pairs clearing the frozen threshold ($\ge 0.76$) are written incrementally to `output/matching_results.tsv`.
   - All S1 entities (including singletons with 0 matches) are guaranteed to appear.
4. **RAM & Disk Footprint**:
   - **Peak System RAM**: **< 5.0 GB** on full test inference (safely inside Colab's 12.7 GB limit).
   - **Disk Space Required**: ~10–15 GB free disk (well within Colab's 84 GB free disk).

---

## Google Colab & Local Execution Guide

### 1. Environment Setup
```bash
cd business_entity_resolution
pip install -r requirements.txt
```

### 2. Training Pipeline (Train $\to$ Grouped Validation $\to$ Calibration)
```bash
python3 src/train.py --data-dir dataset/train --k 50 --neg-ratio 4.0
```
This executes:
- Country-partitioned IDF inverted index blocking ($K=50$, streaming candidates to disk)
- Hard negative sampling ($4\times$ negative-to-positive ratio)
- 36-feature extraction (RapidFuzz string similarity, blocking IDF scores, entity-relative features, address unit matching)
- Leak-free `GroupShuffleSplit` by `source1_entity_id`
- LightGBM training with early stopping
- Exact macro $F_{0.5}$ threshold sweep ($0.10 \to 0.99$)
- Saves calibrated model + metadata to `model.joblib`

### 3. Memory-Safe Streaming Test Inference
```bash
python3 src/infer.py --data-dir dataset/test --k 50 --threshold 0.76 --chunk-size 100000
```
Generates the two official competition artifacts incrementally with < 5 GB peak RAM:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### 4. Competition Formatting Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Blocking is now token-based (word-overlap), not nearest-neighbor
search** — this is what lets it handle millions of rows in minutes
instead of hours. See the comments at the top of `src/blocking.py` for
how it works and why it changed.

## What to check first, before trusting any of it

1. **Run `train.py` and read the "blocking recall check" line it prints.**
   This tells you what % of true matches your blocking stage is even
   capable of finding. If it's low (below ~90%), the model can never
   recover those pairs no matter how good the classifier is — fix that
   first by increasing `k` in `build_candidates()` (more candidates per
   entity) before touching anything else.

2. **Look at the threshold sweep output.** It prints validation F_0.5 at
   several thresholds — sanity check that the trend makes sense (usually
   rises then falls) rather than blindly trusting the picked number.

3. **Open `output/matching_results.tsv` and skim it manually** — do a few
   of the matched pairs actually look right when you read the names/
   addresses side by side? This "eyeball check" catches obvious bugs
   faster than staring at metrics.

## Places you'll likely want to improve once the baseline runs

- **Increase `k` in blocking** if recall check is low (trade-off: slower,
  more candidate pairs to score).
- **Add more entries to `LEGAL_SUFFIX_MAP` / `ADDRESS_ABBR_MAP`** in
  `common.py` as you spot new abbreviation patterns during EDA.
- **Add a semantic embedding feature** (optional) — e.g. a small
  multilingual sentence-transformers model's cosine similarity as one
  more column in `FEATURE_COLUMNS`, if pure string similarity plateaus.
- **Tune LightGBM hyperparameters** in `train.py` (`n_estimators`,
  `num_leaves`, etc.) once the pipeline works end-to-end — don't do this
  before the basic pipeline is correct, it's a distraction early on.

## Common first-run errors

- `FileNotFoundError` on a `.tsv` path → you haven't placed the dataset
  files yet, or a filename doesn't match exactly (case-sensitive).
- Training takes forever / hangs → your dataset is much larger than
  expected; reduce `k` in `build_candidates()` temporarily to test the
  pipeline on a subsample first, then scale up.
- All predictions come back empty → your threshold is set too high, or
  the blocking recall check was near 0 (candidates never contained real
  matches to begin with) — check step 1 above.
