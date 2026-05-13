#!/usr/bin/env python
"""Dump key statistics of N safetensors files for comparison.

For each file: total keys, key prefixes (e.g. "ema_model.", "vlm.", "action_expert."),
key dtype distribution. Helps spot ckpt structure mismatches.
"""
import argparse
import os
from collections import Counter, defaultdict

from safetensors import safe_open


def inspect(path):
    print(f"\n=== {path} ===")
    print(f"size: {os.path.getsize(path) / 1024**3:.2f} GB")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print(f"total keys: {len(keys)}")

        # Top-level prefixes (split on first '.')
        prefixes = Counter(k.split(".")[0] for k in keys)
        print(f"top-level prefixes: {dict(prefixes.most_common())}")

        # Dtype distribution
        dtypes = Counter()
        for k in keys[:200]:  # sample first 200
            t = f.get_tensor(k)
            dtypes[str(t.dtype)] += 1
        print(f"dtype dist (first 200 keys sampled): {dict(dtypes)}")

        # Look for EMA / model duplication
        bare_keys = set(k for k in keys if not k.startswith("ema"))
        ema_keys = set(k for k in keys if k.startswith("ema"))
        print(f"non-ema keys: {len(bare_keys)}, ema-prefixed keys: {len(ema_keys)}")

        # Check 3 sample keys
        for k in keys[:3]:
            t = f.get_tensor(k)
            print(f"  {k!r}: shape={list(t.shape)}, dtype={t.dtype}, mean={t.float().mean().item():.5f}, std={t.float().std().item():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Paths to safetensors files")
    args = ap.parse_args()
    for p in args.paths:
        try:
            inspect(p)
        except Exception as e:
            print(f"\n=== {p} ===\nERROR: {e}")


if __name__ == "__main__":
    main()
