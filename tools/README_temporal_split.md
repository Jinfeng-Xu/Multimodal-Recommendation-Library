# Temporal leave-one-out split tool for MRLib datasets

## Why

An audit of the bundled splits shows the `x_label` column is effectively random:

- only **37.3%** of users have their test interaction later than all their training interactions;
- only **14%** of intra-user rows are time-ordered.

Any method that assumes "predict the future" semantics (sequential recommenders,
memory-based modules, temporal evaluation protocols) is silently evaluated on a
shuffled protocol.

## What this tool provides

`tools/make_time_split.py` builds a **per-user temporal leave-one-out** split
(TIGER / SASRec convention) that drops into MRLib **without code changes**:

- sorts each user's interactions by timestamp (stable mergesort, ties keep row order);
- assigns `x_label = 2` (test) to the last interaction, `1` (valid) to the
  second-to-last, `0` (train) to the rest;
- keeps the original `x_label` column untouched for users with < 3 interactions;
- works through the existing `inter_splitting_label` config knob;
- ships with a built-in verifier (reports the fraction of users whose test
  interaction is not earlier than their last training interaction; expected ~100%).

## Usage

```bash
python tools/make_time_split.py --dataset baby --data-dir ./data
# -> ./data/baby_time/baby_time.inter (+ copies of *feat*.npy)
# then run MRLib with data_path pointing to baby_time
```

## Scope & limitations

- This is a per-user split: it still interleaves across users in time. For a
  strictly leakage-free protocol, a global timepoint split (split-by-time) is a
  natural follow-up and is planned as a separate tool.
- Cold-start filtering in validation/test remains MRLib's existing behavior.

## Verification (example, Baby 5-core)

```
[verify] test-after-train users: ~100% (n=<all users with >=3 interactions>)
```
