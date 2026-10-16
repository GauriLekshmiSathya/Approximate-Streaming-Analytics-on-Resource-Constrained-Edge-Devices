"""Reproducible fixed-budget OASRS/VA-OASRS benchmark on deterministic data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import sys as _sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from streamapprox.allocation import allocate_capacities
from streamapprox.aggregators import approximate_mean, approximate_sum
from streamapprox.estimators import estimate_mean as estimate_mean_ci
from streamapprox.oasrs import OASRS
from streamapprox.srs import SRSSampler
from streamapprox.va_oasrs import VAOASRS
from streamapprox.dv_oasrs import DVOASRS


def make_records(kind: str, n: int, seed: int, variance_ratio: float = 10.0,
                 skew: tuple[float, ...] | None = None) -> list[dict]:
    rng = random.Random(seed)
    if kind in ("uniform", "variance_skew"):
        proportions = (1 / 5,) * 5
    elif kind in ("population_skew", "combined_skew"):
        proportions = skew or (0.60, 0.20, 0.10, 0.05, 0.05)
    elif kind == "distribution_shift":
        proportions = (0.2,) * 5
    else:
        raise ValueError(kind)
    cumulative = []
    total = 0.0
    for p in proportions:
        total += p
        cumulative.append(total)
    stds = [1.0] * 5
    if kind in ("variance_skew", "combined_skew"):
        stds = [1.0, 1.0, math.sqrt(variance_ratio), math.sqrt(variance_ratio), math.sqrt(variance_ratio)]
    records = []
    for i in range(n):
        draw = rng.random()
        sid = next(j for j, edge in enumerate(cumulative) if draw <= edge)
        std = stds[sid]
        if kind == "distribution_shift" and i >= n // 2 and sid in (3, 4):
            std *= 8.0
        records.append({
            "sensor_id": f"S{sid + 1}",
            "value": rng.gauss(10.0 + sid * 2.0, std),
            "timestamp": i,
        })
    return records


def exact(records: list[dict]) -> tuple[float, float, int]:
    return sum(r["value"] for r in records), statistics.fmean(r["value"] for r in records), len(records)


def capacities(records: list[dict], budget: int) -> dict[str, int]:
    strata = set()
    for record in records:
        strata.add(record["sensor_id"])
    # Equal fixed-capacity OASRS cost function; it uses only configured stratum
    # keys and budget, never final stream frequencies or values.
    return allocate_capacities(budget, {sid: 1.0 for sid in strata}, minimum=1)


def run_sampler(method: str, records: list[dict], budget: int, seed: int,
                interval: int) -> tuple[object, float, float, float, dict]:
    started = time.perf_counter()
    if method == "Exact":
        sm = sum(r["value"] for r in records)
        return None, sm, sm / len(records), time.perf_counter() - started, {}
    if method == "Uniform Reservoir":
        sampler = SRSSampler(sample_size=budget, seed=seed)
        sampler.update_batch(records)
        return sampler, sampler.approximate_sum("value"), sampler.approximate_mean("value"), time.perf_counter() - started, {}
    if method == "OASRS":
        caps = capacities(records, budget)
        sampler = OASRS(sample_size=1, stratum_key="sensor_id", seed=seed,
                        stratum_capacity=lambda sid: caps[sid])
        sampler.update_batch(records)
        snap = sampler.snapshot()
    elif method in ("VA-OASRS", "Static VA-OASRS"):
        sampler = VAOASRS(sample_size=budget, stratum_key="sensor_id", value_key="value",
                          seed=seed, reallocation_interval=interval, burn_in_records=interval)
        sampler.update_batch(records)
        snap = sampler.finalize()
    elif method == "DV-OASRS":
        sampler = DVOASRS(sample_size=budget, stratum_key="sensor_id", value_key="value",
                          seed=seed, reallocation_interval=interval)
        sampler.update_batch(records)
        snap = sampler.finalize()
    else:
        raise ValueError(method)
    elapsed = time.perf_counter() - started
    extra = {}
    if method == "DV-OASRS":
        extra = {
            "variance_seconds": sampler.variance_seconds,
            "allocation_seconds": sampler.allocation_seconds,
            "priority_seconds": sampler.priority_seconds,
            "candidate_pool_bytes_estimate": sum(
                _sys.getsizeof(state.priority_heap) +
                sum(_sys.getsizeof(entry) for entry in state.priority_heap)
                for state in sampler.strata.values()),
            "allocation_events": sampler.allocation_events,
        }
    return sampler, approximate_sum(snap, "value"), approximate_mean(snap, "value"), elapsed, extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/va_oasrs")
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[200, 1000, 2000, 4000])
    parser.add_argument("--intervals", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--variance-ratios", type=float, nargs="+", default=[1, 2, 5, 10, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = ["uniform", "population_skew", "variance_skew", "combined_skew", "distribution_shift"]
    raw: list[dict] = []
    allocation_rows: list[dict] = []
    for dataset_index, dataset in enumerate(datasets):
        for seed in args.seeds[:args.runs]:
            records = make_records(dataset, args.records, seed + dataset_index * 1000)
            exact_sum, exact_mean, count = exact(records)
            for budget in args.budgets:
                interval_values = args.intervals if dataset == "distribution_shift" else [args.intervals[0]]
                methods = ["Exact", "Uniform Reservoir", "OASRS"] + [
                    f"Static VA-OASRS@{i}" for i in interval_values
                ] + [f"DV-OASRS@{i}" for i in interval_values]
                for label in methods:
                    method, interval = (label.split("@", 1)[0], int(label.split("@", 1)[1])) if "@" in label else (label, args.intervals[0])
                    sampler, estimate_sum, estimate_mean, elapsed, extra = run_sampler(method, records, budget, seed, interval)
                    sample_count = count if method == "Exact" else sampler.sample_size
                    allocated = sample_count
                    for query, estimate, truth in (("sum", estimate_sum, exact_sum), ("mean", estimate_mean, exact_mean)):
                        error = abs(estimate - truth)
                        ci_result = None
                        if method in ("OASRS", "VA-OASRS", "Static VA-OASRS", "DV-OASRS"):
                            ci_result = estimate_mean_ci(sampler.snapshot(), "value") if query == "mean" else None
                        raw.append({
                            "dataset": dataset, "seed": seed, "budget": budget,
                            "sampling_fraction": budget / count, "method": label,
                            "query": query, "exact": truth, "estimate": estimate,
                            "absolute_error": error,
                            "relative_error": error / abs(truth) if truth else error,
                            "records": count, "sampled": sample_count,
                            "actual_sampling_fraction": sample_count / count,
                            "elapsed_seconds": elapsed,
                            "throughput_records_per_second": count / max(elapsed, 1e-12),
                            "latency_us_per_record": elapsed * 1e6 / count,
                            "interval": interval if method in ("VA-OASRS", "Static VA-OASRS", "DV-OASRS") else "",
                            "estimated_se": ci_result.standard_error if ci_result else "",
                            "ci95_contains_exact": (error <= 1.96 * ci_result.standard_error) if ci_result else "",
                            **extra,
                        })
                    if method in ("VA-OASRS", "Static VA-OASRS", "DV-OASRS"):
                        for event_id, event in enumerate(sampler.allocation_history):
                            capacities_now = event.get("capacities", {})
                            scores_now = event.get("scores", {})
                            for sid in event.get("counts", event.get("capacities", {})):
                                allocation_rows.append({
                                    "dataset": dataset, "seed": seed, "budget": budget,
                                    "method": label,
                                    "interval": interval, "event_id": event_id,
                                    "records_seen": event["records_seen"], "stratum": sid,
                                    "population": event.get("counts", {}).get(sid, ""),
                                    "online_variance": event.get("welford_variances", {}).get(sid, ""),
                                    "capacity": capacities_now.get(sid, 0),
                                    "score": scores_now.get(sid, ""),
                                    "ema_variance": event.get("ema_variances", {}).get(sid, ""),
                                })
    heterogeneity_rows = []
    heterogeneity_budget = args.budgets[0]
    for ratio in args.variance_ratios:
        for seed in args.seeds[:args.runs]:
            records = make_records("variance_skew", args.records, seed + int(ratio * 10000), variance_ratio=ratio)
            exact_sum, exact_mean, count = exact(records)
            for method in ("Uniform Reservoir", "OASRS", "Static VA-OASRS", "DV-OASRS"):
                interval = args.intervals[0]
                sampler, _, estimate_mean, elapsed, extra = run_sampler(method, records, heterogeneity_budget, seed, interval)
                error = abs(estimate_mean - exact_mean)
                heterogeneity_rows.append({"variance_ratio": ratio, "seed": seed, "method": method,
                                           "budget": heterogeneity_budget, "exact_mean": exact_mean,
                                           "estimate_mean": estimate_mean,
                                           "relative_error": error / abs(exact_mean),
                                           "throughput_records_per_second": count / elapsed,
                                           "sampled": sampler.sample_size})
    population_skews = {
        "low": (0.2, 0.2, 0.2, 0.2, 0.2),
        "moderate": (0.5, 0.2, 0.1, 0.1, 0.1),
        "high": (20/28, 5/28, 2/28, 1/28, 0.0),
        "extreme": (100/113, 10/113, 2/113, 1/113, 0.0),
    }
    population_rows = []
    for level, weights in population_skews.items():
        # Keep all five strata active even for very small proportions.
        adjusted = tuple(max(p, 1e-6) for p in weights)
        norm = sum(adjusted)
        adjusted = tuple(p/norm for p in adjusted)
        for seed in args.seeds[:args.runs]:
            records = make_records("population_skew", args.records,
                                   seed + len(level)*100, skew=adjusted)
            _, truth, count = exact(records)
            for method in ("Uniform Reservoir", "OASRS", "Static VA-OASRS", "DV-OASRS"):
                sampler, _, estimate, elapsed, extra = run_sampler(method, records,
                    args.budgets[0], seed, args.intervals[0])
                population_rows.append({"population_skew": level, "seed": seed,
                    "method": method, "budget": args.budgets[0], "exact_mean": truth,
                    "estimate_mean": estimate, "relative_error": abs(estimate-truth)/abs(truth),
                    "throughput_records_per_second": count/elapsed,
                    "sampled": sampler.sample_size, **extra})
    raw_path = args.output / "raw_results.csv"
    with raw_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in raw for k in row}))
        writer.writeheader()
        writer.writerows(raw)
    allocation_path = args.output / "allocations.csv"
    with allocation_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in allocation_rows for k in row}))
        writer.writeheader()
        writer.writerows(allocation_rows)
    with (args.output / "variance_heterogeneity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in heterogeneity_rows for k in row}))
        writer.writeheader(); writer.writerows(heterogeneity_rows)
    with (args.output / "population_skew.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in population_rows for k in row}))
        writer.writeheader(); writer.writerows(population_rows)
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["dataset"], row["budget"], row["method"], row["query"])].append(row)
    summary = []
    for (dataset, budget, method, query), rows in sorted(grouped.items()):
        errors = [r["relative_error"] for r in rows]
        n = len(errors)
        mean = statistics.fmean(errors)
        sd = statistics.stdev(errors) if n > 1 else 0.0
        summary.append({
            "dataset": dataset, "budget": budget, "method": method, "query": query,
            "runs": n, "mean_relative_error": mean, "sd_relative_error": sd,
            "ci95_low": max(0.0, mean - 2.776 * sd / math.sqrt(n)) if n == 5 else "",
            "ci95_high": mean + 2.776 * sd / math.sqrt(n) if n == 5 else "",
            "mean_throughput_records_per_second": statistics.fmean(r["throughput_records_per_second"] for r in rows),
            "mean_latency_us_per_record": statistics.fmean(r["latency_us_per_record"] for r in rows),
            "mean_sampled": statistics.fmean(r["sampled"] for r in rows),
            "mean_actual_sampling_fraction": statistics.fmean(r["actual_sampling_fraction"] for r in rows),
            "mean_ci95_coverage": statistics.fmean(float(r["ci95_contains_exact"]) for r in rows if r["ci95_contains_exact"] != "") if any(r["ci95_contains_exact"] != "" for r in rows) else "",
        })
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    processed = args.output / "processed"
    processed.mkdir(exist_ok=True)
    with (processed / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    paired = defaultdict(list)
    by_key = {(r["dataset"], r["budget"], r["seed"], r["query"], r["method"]): float(r["relative_error"]) for r in raw}
    for dataset, budget, seed, query, method in list(by_key):
        if method not in ("Exact", "OASRS"):
            key = (dataset, budget, seed, query)
            if (dataset, budget, seed, query, "OASRS") in by_key:
                paired[key + (method,)].append(by_key[(dataset, budget, seed, query, "OASRS")] - by_key[(dataset, budget, seed, query, method)])
    paired_rows = []
    for key, differences in sorted(paired.items()):
        dataset, budget, _seed, query, method = key
        mean_diff = statistics.fmean(differences)
        sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
        half = 2.776 * sd / math.sqrt(len(differences)) if len(differences) == 5 else ""
        paired_rows.append({"dataset": dataset, "budget": budget, "query": query, "method": method,
                            "paired_runs": len(differences), "mean_OASRS_minus_method_relative_error": mean_diff,
                            "ci95_low": max(-math.inf, mean_diff-half) if half != "" else "",
                            "ci95_high": mean_diff+half if half != "" else ""})
    with (processed / "paired_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)
    (args.output / "config.json").write_text(json.dumps(vars(args) | {"datasets": datasets}, default=str, indent=2))
    print(f"Wrote {raw_path}, {allocation_path}, and summary.csv")


if __name__ == "__main__":
    main()
