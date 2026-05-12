import argparse
import json
import os
from collections import Counter

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def decision_regions(scores, accept_threshold, reject_threshold):
    decisions = np.full(scores.shape, "boundary", dtype=object)
    decisions[scores <= accept_threshold] = "accept"
    decisions[scores >= reject_threshold] = "reject"
    return decisions


def summarize_region(decisions, correct, scores, name):
    mask = decisions == name
    return {
        "count": int(mask.sum()),
        "rate": float(mask.mean()),
        "accuracy": float(correct[mask].mean()) if mask.any() else None,
        "mean_uncertainty": float(scores[mask].mean()) if mask.any() else None,
        "error_count": int((~correct[mask]).sum()) if mask.any() else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--accept_quantile", type=float, default=0.1)
    parser.add_argument("--reject_quantile", type=float, default=0.9)
    parser.add_argument("--output", default="calibrated_three_way.json")
    args = parser.parse_args()

    if args.accept_quantile >= args.reject_quantile:
        raise ValueError("--accept_quantile must be lower than --reject_quantile")

    cases_path = os.path.join(args.run_dir, "cases.json")
    cases = load_json(cases_path)
    scores = np.array([case["uncertainty"]["uncertainty_score"] for case in cases])
    correct = np.array([case["prediction"] == case["label"] for case in cases])
    labels = np.array([case["label"] for case in cases])

    accept_threshold = float(np.quantile(scores, args.accept_quantile))
    reject_threshold = float(np.quantile(scores, args.reject_quantile))
    decisions = decision_regions(scores, accept_threshold, reject_threshold)

    label_region_counts = {}
    for label in sorted(set(labels.tolist())):
        label_mask = labels == label
        label_region_counts[label] = {
            region: int(np.logical_and(label_mask, decisions == region).sum())
            for region in ["accept", "boundary", "reject"]
        }

    result = {
        "source_cases": cases_path,
        "calibration_source": "current cases.json",
        "note": "Use validation-set calibration for paper-facing test results; this file is for post-hoc diagnostics.",
        "accept_quantile": args.accept_quantile,
        "reject_quantile": args.reject_quantile,
        "accept_threshold": accept_threshold,
        "reject_threshold": reject_threshold,
        "sample_count": int(len(cases)),
        "overall_accuracy": float(correct.mean()),
        "mean_uncertainty": float(scores.mean()),
        "mean_correct_uncertainty": float(scores[correct].mean()) if correct.any() else None,
        "mean_error_uncertainty": float(scores[~correct].mean()) if (~correct).any() else None,
        "regions": {
            region: summarize_region(decisions, correct, scores, region)
            for region in ["accept", "boundary", "reject"]
        },
        "label_region_counts": label_region_counts,
        "decision_counts": dict(Counter(decisions.tolist())),
    }

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.run_dir, output_path)
    write_json(result, output_path)
    print(json.dumps(result["regions"], ensure_ascii=True, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
