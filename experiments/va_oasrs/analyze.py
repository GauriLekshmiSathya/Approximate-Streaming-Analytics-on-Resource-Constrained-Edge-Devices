"""Compute paired confidence intervals and simple allocation diagnostics."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/va_oasrs"))
    args = parser.parse_args()
    rows = read(args.results / "raw_results.csv")
    paired = defaultdict(dict)
    for row in rows:
        if (row["method"] in ("OASRS",) or row["method"].startswith("VA-OASRS")
                or row["method"].startswith("Static VA-OASRS")
                or row["method"].startswith("DV-OASRS")):
            key = (row["dataset"], row["budget"], row["query"], row["seed"])
            paired[key][row["method"]] = float(row["relative_error"])
    comparisons = []
    groups = defaultdict(list)
    for (dataset, budget, query, seed), vals in paired.items():
        for vm, error in vals.items():
            if (vm.startswith("VA-OASRS") or vm.startswith("Static VA-OASRS")
                    or vm.startswith("DV-OASRS")) and "OASRS" in vals:
                groups[(dataset, budget, query, vm)].append(vals["OASRS"] - error)
    for (dataset, budget, query, method), diffs in sorted(groups.items()):
        n = len(diffs); mean = sum(diffs)/n
        sd = math.sqrt(sum((x-mean)**2 for x in diffs)/(n-1)) if n > 1 else 0.0
        critical = T975.get(n-1, 1.96) if n > 1 else float("nan")
        half = critical * sd / math.sqrt(n) if n > 1 else float("nan")
        comparisons.append({"dataset":dataset,"budget":budget,"query":query,"method":method,
                            "paired_runs":n,"mean_OASRS_minus_VA_relative_error":mean,
                            "ci95_low":mean-half,"ci95_high":mean+half})
    target = args.results / "processed" / "paired_comparisons.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]) if comparisons else ["dataset"])
        writer.writeheader(); writer.writerows(comparisons)


if __name__ == "__main__":
    main()
