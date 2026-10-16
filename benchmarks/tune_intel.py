"""Budget and per-stratum allocation study on the Intel Berkeley dataset.

Uses a 100-record-per-sensor pilot to estimate stratum frequencies and
temperature variability, then compares equal, proportional, and Neyman
allocations at fixed total sample budgets. Repeated reservoir seeds measure
run-to-run error. The full file is replayed for each configuration.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from streamapprox.aggregators import approximate_mean
from streamapprox.estimators import estimate_mean
from streamapprox.oasrs import OASRS

FIELDS = ("temperature", "humidity", "light", "voltage")
POLICIES = ("equal", "proportional", "neyman")


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


def records(path):
    with open(path, "rt") as stream:
        for line in stream:
            record = common.parse_intel_line(line)
            if record is not None:
                yield record


def collect_pilot(path, per_sensor):
    pilot = defaultdict(RunningStats)
    for record in records(path):
        stats = pilot[record["mote_id"]]
        if stats.n < per_sensor:
            stats.add(record["temperature"])
        if pilot and all(s.n >= per_sensor for s in pilot.values()) and len(pilot) >= 54:
            break
    return pilot


def allocate(budget, strata, policy, pilot):
    """Integer Hamilton allocation with at least one slot per observed stratum."""
    ids = sorted(strata)
    if budget < len(ids):
        raise ValueError(f"budget {budget} is below {len(ids)} discovered strata")
    weights = {}
    for sid in ids:
        if policy == "equal":
            weight = 1.0
        elif policy == "proportional":
            weight = float(strata[sid])
        else:
            stats = pilot.get(sid)
            sd = math.sqrt(stats.variance) if stats else 0.0
            weight = float(strata[sid]) * sd
        weights[sid] = max(weight, 1e-12)
    remaining = budget - len(ids)
    total_weight = sum(weights.values())
    exact = {sid: remaining * weights[sid] / total_weight for sid in ids}
    caps = {sid: 1 + int(exact[sid]) for sid in ids}
    leftovers = budget - sum(caps.values())
    for sid in sorted(ids, key=lambda x: exact[x] - int(exact[x]), reverse=True)[:leftovers]:
        caps[sid] += 1
    return caps


def exact_dataset(path):
    counts = defaultdict(int)
    sums = {field: 0.0 for field in FIELDS}
    total = 0
    for record in records(path):
        total += 1
        counts[record["mote_id"]] += 1
        for field in FIELDS:
            sums[field] += record[field]
    return total, counts, {field: sums[field] / total for field in FIELDS}


def run_trial(path, budget, policy, capacities, seed, exact_means):
    sampler = OASRS(1, stratum_key="mote_id", seed=seed,
                    stratum_capacity=lambda sid: capacities[sid])
    started = time.perf_counter()
    processed = 0
    for record in records(path):
        sampler.update(record)
        processed += 1
    elapsed = max(1e-9, time.perf_counter() - started)
    snapshot = sampler.snapshot()
    rows = []
    for field in FIELDS:
        result = estimate_mean(snapshot, field)
        empirical = approximate_mean(snapshot, field)
        truth = exact_means[field]
        abs_error = abs(empirical - truth)
        rel_error = abs_error / abs(truth) if truth else abs_error
        rows.append({
            "policy": policy,
            "budget": budget,
            "seed": seed,
            "query": field + "_mean",
            "exact_mean": truth,
            "estimate": empirical,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "estimated_se": result.standard_error,
            "ci95_contains_exact": abs_error <= 1.96 * result.standard_error,
            "sampled_items": snapshot.sample_size,
            "allocated_capacity": sum(capacities.values()),
            "records": processed,
            "elapsed_seconds": elapsed,
            "throughput_records_per_second": processed / elapsed,
        })
    return rows


def write_svg(summary, path, query):
    policies = list(POLICIES)
    points = [(x["budget"], x["mean_relative_error"], x["policy"])
              for x in summary if x["query"] == query]
    if not points:
        return
    width, height = 850, 500
    left, right, top, bottom = 90, 30, 45, 70
    xmin, xmax = min(x[0] for x in points), max(x[0] for x in points)
    ymin, ymax = 0.0, max(x[1] for x in points) * 1.1 or 1.0
    sx = lambda x: left + (x - xmin) / max(1, xmax - xmin) * (width-left-right)
    sy = lambda y: height-bottom - (y-ymin) / max(1e-12, ymax-ymin) * (height-top-bottom)
    colors = {"equal": "#2867a6", "proportional": "#d96b27", "neyman": "#16845b"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{query}: accuracy vs total sample budget</text>',
           f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#333"/>',
           f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif">Total sample budget</text>',
           f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">Mean relative error</text>']
    for i in range(5):
        y = ymax * i / 4
        out += [f'<path d="M {left} {sy(y):.2f} H {width-right}" stroke="#ddd"/>',
                f'<text x="{left-8}" y="{sy(y)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>']
    for policy in policies:
        series = sorted((x, y) for x, y, p in points if p == policy)
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in series)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{colors[policy]}" stroke-width="3"/>')
        for x, y in series:
            out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{colors[policy]}"/>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/data.txt"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[270, 540, 1080, 2160])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-per-sensor", type=int, default=100)
    parser.add_argument("--target-relative-error", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("results/intel-budget-study"))
    args = parser.parse_args()
    if args.runs < 1 or args.pilot_per_sensor < 2:
        parser.error("runs must be positive and pilot-per-sensor must be at least 2")
    pilot = collect_pilot(str(args.file), args.pilot_per_sensor)
    pilot = {sid: stats for sid, stats in pilot.items() if stats.n >= 2}
    total, counts, exact_means = exact_dataset(str(args.file))
    raw = []
    for budget in args.budgets:
        for policy in POLICIES:
            capacities = allocate(budget, counts, policy, pilot)
            for repeat in range(args.runs):
                raw.extend(run_trial(str(args.file), budget, policy, capacities,
                                     args.seed + repeat, exact_means))
    args.output.mkdir(parents=True, exist_ok=True)
    raw_fields = list(raw[0])
    with (args.output / "raw_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw)
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["query"], row["budget"], row["policy"])].append(row)
    summary = []
    for (query, budget, policy), rows in sorted(grouped.items()):
        errors = [row["relative_error"] for row in rows]
        summary.append({
            "query": query,
            "budget": budget,
            "policy": policy,
            "runs": len(rows),
            "mean_relative_error": statistics.fmean(errors),
            "sd_relative_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "median_relative_error": statistics.median(errors),
            "mean_ci95_coverage": statistics.fmean(float(r["ci95_contains_exact"]) for r in rows),
            "mean_sampled_items": statistics.fmean(r["sampled_items"] for r in rows),
            "mean_throughput_records_per_second": statistics.fmean(r["throughput_records_per_second"] for r in rows),
            "mean_estimated_se": statistics.fmean(r["estimated_se"] for r in rows),
        })
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    feasible = [row for row in summary if row["query"] == "temperature_mean"
                and row["mean_relative_error"] <= args.target_relative_error
                and row["mean_ci95_coverage"] >= 0.9]
    chosen = min(feasible, key=lambda row: (row["mean_sampled_items"], row["mean_relative_error"])) if feasible else min(
        (row for row in summary if row["query"] == "temperature_mean"),
        key=lambda row: (row["mean_relative_error"], row["mean_sampled_items"]))
    pareto = []
    temp = [row for row in summary if row["query"] == "temperature_mean"]
    for row in temp:
        dominated = any(other["mean_sampled_items"] <= row["mean_sampled_items"]
                        and other["mean_relative_error"] <= row["mean_relative_error"]
                        and (other["mean_sampled_items"] < row["mean_sampled_items"]
                             or other["mean_relative_error"] < row["mean_relative_error"])
                        for other in temp)
        if not dominated:
            pareto.append(row)
    study = {
        "dataset": str(args.file), "records": total, "strata": len(counts),
        "pilot_records_per_stratum": args.pilot_per_sensor, "runs_per_configuration": args.runs,
        "seed_start": args.seed, "budgets": args.budgets, "policies": POLICIES,
        "allocation_note": "Neyman uses pilot N_i * sample standard deviation for temperature; proportional uses pilot population counts. Both require pilot statistics, unlike equal allocation.",
        "best_configuration": chosen,
        "recommendation_rule": f"For temperature mean, choose the lowest realized sample cost meeting mean relative error <= {args.target_relative_error:.1%} and at least 90% empirical 95%-CI coverage; if none meet both, choose lowest error.",
        "temperature_pareto_front": pareto,
        "interpretation": "Relative error is empirical against full-dataset exact means. CI coverage is empirical across repeated reservoir seeds. Throughput is single-process Python on this machine and is not comparable to the paper's Spark/Flink cluster figures.",
    }
    (args.output / "study.json").write_text(json.dumps(study, indent=2))
    plots = args.output / "plots"
    plots.mkdir(exist_ok=True)
    for query in ("temperature_mean", "humidity_mean", "light_mean", "voltage_mean"):
        write_svg(summary, plots / (query + ".svg"), query)
    print(json.dumps({"records": total, "strata": len(counts), "configurations": len(args.budgets)*len(POLICIES),
                      "runs": args.runs, "best": chosen, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
