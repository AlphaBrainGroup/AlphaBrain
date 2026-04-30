"""Compute ASR / BWT / F from a T×T continual-learning evaluation matrix.

The CL eval pipeline (`run_cl_eval.sh --matrix`) produces a directory layout

    <matrix_dir>/
        task_0_id0_steps_*/eval.log
        task_1_id1_steps_*/eval.log
        ...
        task_{T-1}_id{T-1}_steps_*/eval.log

where each subdirectory contains the evaluation of the checkpoint *after
having trained the (k+1)-th task* against **all T tasks of the benchmark**
in the suite-defined evaluation order.  This script parses the per-task
SRs out of each `eval.log`, recovers the train→eval permutation, and
prints the three standard CL metrics:

* **ASR** — average final-checkpoint success rate, `mean_j a_{T-1, j}`
* **BWT** — Backward Transfer (Lopez-Paz & Ranzato 2017),
            `1/(T-1) · Σ_{i<T-1} (a_{T-1, π(i)} − a_{i, π(i)})` in
            percentage points.  `0` = no forgetting; `<0` = forgetting.
* **F**   — Forgetting (Chaudhry et al. 2018),
            `1/(T-1) · Σ_{i<T-1} (max_{l∈[i,T-1)} a_{l, π(i)} − a_{T-1, π(i)})`
            in percentage points.  `0` = peak retained; `>0` = drop from peak.

Usage:
    python scripts/run_continual_learning_scripts/compute_cl_matrix_metrics.py \
        results/eval_cl/<run_id>_matrix
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_per_task_sr(eval_log: Path) -> list[float]:
    text = eval_log.read_text(errors="ignore")
    clean = re.sub(r"\x1b\[[^m]*m", "", text)
    rates = re.findall(r"Current task success rate:\s*([0-9.]+)", clean)
    return [float(x) for x in rates]


def load_matrix(matrix_dir: Path) -> list[list[float]]:
    ckpt_dirs = sorted(
        (p for p in matrix_dir.iterdir() if p.is_dir() and p.name.startswith("task_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not ckpt_dirs:
        raise SystemExit(f"No `task_*` subdirs found under {matrix_dir}")

    mat: list[list[float]] = []
    for d in ckpt_dirs:
        log = d / "eval.log"
        if not log.exists():
            raise SystemExit(f"Missing {log}")
        row = parse_per_task_sr(log)
        if not row:
            raise SystemExit(f"Could not parse per-task SRs from {log}")
        mat.append(row)

    T = len(mat)
    for k, row in enumerate(mat):
        if len(row) < T:
            raise SystemExit(
                f"Row {k} has only {len(row)} eval cells, expected {T} "
                f"(matrix may be incomplete)"
            )
        mat[k] = row[:T]
    return mat


def infer_train_eval_permutation(mat: list[list[float]]) -> list[int]:
    T = len(mat)
    perm = [max(range(T), key=lambda j: mat[0][j])]
    for k in range(1, T):
        prev, cur = mat[k - 1], mat[k]
        perm.append(max(range(T), key=lambda j: cur[j] - prev[j]))
    return perm


def compute_metrics(mat: list[list[float]], perm: list[int]) -> dict:
    T = len(mat)
    asr = sum(mat[T - 1]) / T

    bwt_terms = [mat[T - 1][perm[i]] - mat[i][perm[i]] for i in range(T - 1)]
    bwt = sum(bwt_terms) / (T - 1)

    f_terms = []
    for i in range(T - 1):
        peak = max(mat[k][perm[i]] for k in range(i, T - 1))
        f_terms.append(peak - mat[T - 1][perm[i]])
    F = sum(f_terms) / (T - 1)

    return {
        "asr": asr,
        "bwt": bwt,
        "F": F,
        "bwt_terms": bwt_terms,
        "f_terms": f_terms,
        "perm": perm,
        "matrix": mat,
    }


def fmt_matrix(mat: list[list[float]]) -> str:
    T = len(mat)
    head = "        " + "  ".join(f"T{j}".rjust(5) for j in range(T))
    rows = [head]
    for i, row in enumerate(mat):
        rows.append(f"  T{i:<2}  " + "  ".join(f"{x:.2f}".rjust(5) for x in row))
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "matrix_dir",
        type=Path,
        help="Path to the T×T matrix eval root (contains task_0_*/, task_1_*/ ...)",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the full T×T SR matrix in addition to the headline metrics.",
    )
    args = parser.parse_args()

    if not args.matrix_dir.exists():
        print(f"matrix_dir does not exist: {args.matrix_dir}", file=sys.stderr)
        return 1

    mat = load_matrix(args.matrix_dir)
    perm = infer_train_eval_permutation(mat)
    m = compute_metrics(mat, perm)

    if args.show_matrix:
        print(fmt_matrix(mat))
        print()

    print(f"Run: {args.matrix_dir.name}")
    print(f"T = {len(mat)} tasks")
    print(f"Train→eval permutation: {perm}")
    print()
    print(f"ASR  = {m['asr'] * 100:6.2f} %")
    print(f"BWT  = {m['bwt'] * 100:+6.2f} pp   (Lopez-Paz; 0 = no forgetting)")
    print(f"F    = {m['F'] * 100:6.2f} pp   (Chaudhry; 0 = peak retained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
