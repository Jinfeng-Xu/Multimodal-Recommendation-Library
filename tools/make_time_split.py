# -*- coding: utf-8 -*-
"""
make_time_split.py — Temporal leave-one-out split for MRLib .inter datasets

Motivation (from an audit of the bundled split):
  In the released splits, the x_label column is randomly shuffled: only 37.3% of users
  have their test interaction later than all training interactions, and only 14% of
  intra-user rows are time-ordered. Temporal signals are effectively destroyed, which
  breaks any method that assumes "predict the future" semantics (sequential recommenders,
  memory-based methods, temporal evaluation protocols such as TIGER / SASRec-style LOO).

What this tool does (standard leave-one-out by timestamp, TIGER/SASRec convention):
  For every user, sort interactions by timestamp (stable mergesort: ties keep original
  row order), then assign
      last interaction      -> x_label = 2 (test)
      second-to-last        -> x_label = 1 (valid)
      all remaining         -> x_label = 0 (train)
  Users with fewer than 3 interactions keep their original x_label values (the MRLib
  loader filters cold-start users in validation/test as usual).

Compatibility:
  Output keeps the original column layout plus the x_label column, so it works with
  MRLib's existing config knob `inter_splitting_label` (default 'x_label') without any
  code change — just point `data_path` to the new directory.

Usage:
  python make_time_split.py --dataset baby --data-dir ./data
  # produces ./data/{dataset}_time/{dataset}_time.inter (+ copies of feature files)

Built-in verification:
  After writing, the script reports the fraction of users whose test interaction is
  later than or equal to their last training interaction (should be ~100%).

Note on residual leakage:
  This per-user split still interleaves across users in time (a T2-style protocol).
  For a strictly leakage-free global timepoint split (T3), a separate tool is planned.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def make_time_split(dataset: str, data_dir: Path) -> Path:
    src_dir = data_dir / dataset
    src = src_dir / f"{dataset}.inter"
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = data_dir / f"{dataset}_time"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, sep="\t")
    n_rows, n_users, n_items = len(df), df.userID.nunique(), df.itemID.nunique()
    print(f"[load] rows={n_rows} users={n_users} items={n_items}")

    # 稳定排序：mergesort 保证并列时间保持原行序（可复现）
    df = df.sort_values(["userID", "timestamp"], kind="mergesort").reset_index(drop=True)
    size = df.groupby("userID")["userID"].transform("size")
    pos_from_end = df.groupby("userID").cumcount(ascending=False)

    df["x_label"] = df["x_label"].where(size < 3, 0)          # <3 条交互保持原样
    df.loc[(size >= 3) & (pos_from_end == 0), "x_label"] = 2  # test
    df.loc[(size >= 3) & (pos_from_end == 1), "x_label"] = 1  # valid

    out_file = out_dir / f"{dataset}_time.inter"
    df.to_csv(out_file, sep="\t", index=False)

    # 特征文件原样复制（与 inter 同目录约定）
    for f in src_dir.glob("*feat*.npy"):
        shutil.copy2(f, out_dir / f.name)

    # 内置验证：test 时间 >= train 末次时间的用户占比（应接近 100%）
    chk = df.groupby("userID").apply(
        lambda g: (g.loc[g.x_label == 2, "timestamp"].min() >= g.loc[g.x_label == 0, "timestamp"].max())
        if (g.x_label == 2).any() else None,
        include_groups=False,
    ).dropna()
    ratio = float(chk.mean())
    print(f"[verify] test-after-train users: {ratio * 100:.1f}% (n={len(chk)})")
    print(f"[write] {out_file}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Temporal leave-one-out split for MRLib .inter datasets")
    ap.add_argument("--dataset", required=True, help="dataset directory name, e.g. baby")
    ap.add_argument("--data-dir", default="./data", help="data root (contains <dataset>/<dataset>.inter)")
    args = ap.parse_args()
    make_time_split(args.dataset, Path(args.data_dir))


if __name__ == "__main__":
    main()
