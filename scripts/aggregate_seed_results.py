"""Aggregate per-seed evaluation JSONs into across-seed mean and std.

The paper reports every metric as mean and standard deviation over five
sampling seeds. Each seed is one gen_and_evaluate.py run (or one
eval_liveband_protocol.py run) writing one results JSON, and this script
collapses those files into the reported numbers.

Usage:
    python scripts/aggregate_seed_results.py \
        logs/eval_results/<model>_eval_s42.json \
        logs/eval_results/<model>_eval_s43.json \
        ... (one file per seed)

Every numeric leaf that appears in all input files is aggregated. For the
headline protocol the relevant leaves are beat_alignment/pred_f_measure/mean,
cocola/*/pred_scores/mean, and fad/score, with the gt_* leaves giving the
ground-truth reference row.
"""
import argparse
import json

import numpy as np


def leaves(obj, prefix=""):
    """Yield (path, value) for every numeric leaf in a nested dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}" if prefix else k)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", nargs="+", help="one results JSON per seed")
    args = parser.parse_args()

    per_file = []
    for path in args.results_json:
        with open(path) as f:
            per_file.append(dict(leaves(json.load(f))))

    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)

    print(f"{len(per_file)} seed files, {len(common)} shared numeric leaves\n")
    width = max(len(k) for k in common)
    for key in sorted(common):
        vals = np.array([d[key] for d in per_file])
        # Per-seed "std" leaves are within-run stds; averaging them across
        # seeds is not meaningful, so only "mean" and score leaves are shown.
        if key.endswith("/std"):
            continue
        print(f"{key:<{width}}  {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()
