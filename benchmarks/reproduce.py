"""Reproduce the paper's local, single-process microbenchmarks.

The original work compared Spark Streaming, Flink, Spark SRS/STS and native
execution on a 17-node cluster. This script uses the same generated workload,
window/query definitions and Python process for Exact, global SRS and OASRS;
it explicitly does not claim distributed scalability or JVM engine results.
"""

import argparse
import csv
import html
import json
import math
import statistics
import time
from pathlib import Path

from streamapprox.aggregators import approximate_mean, approximate_sum
from streamapprox.estimators import estimate_mean
from streamapprox.oasrs import OASRS
from streamapprox.srs import SRSSampler
from streamapprox.synthetic import (
    compute_ground_truth,
    create_paper_gaussian_stream,
    create_paper_skewed_gaussian_stream,
    create_paper_skewed_poisson_stream,
    create_paper_poisson_stream,
)
from streamapprox.windows import TimeWindowManager


FRACTIONS = (0.10, 0.20, 0.40, 0.60, 0.80)
ARRIVALS = (
    {"A": 8000, "B": 2000, "C": 100},
    {"A": 3000, "B": 3000, "C": 3000},
    {"A": 100, "B": 2000, "C": 8000},
)


def benchmark_sample(records, *, experiment, dataset, fraction, seed):
    truth = compute_ground_truth(records)
    strata_count = max(1, len(truth["stratum_counts"]))
    target = max(strata_count, int(len(records) * fraction))
    rows = []
    for method in ("Exact", "SRS", "OASRS"):
        started = time.perf_counter()
        if method == "Exact":
            approx_sum, approx_mean = truth["sum"], truth["mean"]
            sample_size = len(records)
            standard_error = 0.0
        elif method == "SRS":
            sampler = SRSSampler(target, seed=seed)
            sampler.update_batch(records)
            approx_sum = sampler.approximate_sum("value")
            approx_mean = sampler.approximate_mean("value")
            sample_size = sampler.sample_size
            standard_error = float("nan")
        else:
            per_stratum = max(1, math.ceil(target / strata_count))
            sampler = OASRS(per_stratum, stratum_key="sensor_id", seed=seed)
            sampler.update_batch(records)
            snap = sampler.snapshot()
            approx_sum = approximate_sum(snap, "value")
            approx_mean = approximate_mean(snap, "value")
            standard_error = estimate_mean(snap, "value").standard_error
            sample_size = snap.sample_size
        elapsed = max(1e-9, time.perf_counter() - started)
        rows.append({
            "experiment": experiment,
            "dataset": dataset,
            "method": method,
            "sampling_fraction": fraction,
            "window_size": "",
            "slide": "",
            "input_rate": "",
            "throughput": len(records) / elapsed,
            "latency": elapsed,
            "sampled_items": sample_size,
            "total_items": len(records),
            "exact_result": truth["mean"],
            "approximate_result": approx_mean,
            "accuracy_loss": abs(approx_mean - truth["mean"]) / abs(truth["mean"])
            if truth["mean"] else (0.0 if approx_mean == 0 else float("inf")),
            "estimated_error": standard_error,
        })
    return rows


def benchmark_windows(records, *, experiment, seed):
    rows = []
    for method in ("Exact", "OASRS"):
        for duration in (10.0, 20.0, 30.0, 40.0):
            manager = TimeWindowManager(
                window_duration=duration,
                slide_interval=5.0,
                sample_size=100,
                stratum_key="sensor_id",
                timestamp_key="timestamp",
                value_key="value",
                seed=seed,
                evaluation_mode=True,
            )
            started = time.perf_counter()
            output = list(manager.process_stream(records))
            elapsed = max(1e-9, time.perf_counter() - started)
            if method == "Exact":
                values = [(x.exact_mean, x.exact_mean, 0.0, x.exact_count) for x in output]
            else:
                values = [(x.mean_result.estimate, x.exact_mean,
                           x.mean_result.standard_error, x.sample_size) for x in output]
            for estimate, exact, se, sampled in values:
                loss = abs(estimate - exact) / abs(exact) if exact else (0 if estimate == 0 else float("inf"))
                rows.append({
                    "experiment": experiment,
                    "dataset": "gaussian_8000_2000_100",
                    "method": method,
                    "sampling_fraction": "",
                    "window_size": duration,
                    "slide": 5.0,
                    "input_rate": "",
                    "throughput": len(records) / elapsed,
                    "latency": elapsed / max(1, len(output)),
                    "sampled_items": sampled,
                    "total_items": "",
                    "exact_result": exact,
                    "approximate_result": estimate,
                    "accuracy_loss": loss,
                    "estimated_error": se,
                })
    return rows


def benchmark_batch_intervals(records, *, seed):
    """Model Spark's batch boundaries, sampling each batch with all baselines."""
    rows = []
    for interval in (0.25, 1.0):
        batches = {}
        for record in records:
            batch_id = int(float(record["timestamp"]) // interval)
            batches.setdefault(batch_id, []).append(record)
        nonempty = [batch for _, batch in sorted(batches.items())]
        for method in ("Exact", "SRS", "OASRS"):
            started = time.perf_counter()
            estimates = []
            errors = []
            standard_errors = []
            sampled_total = 0
            for batch_index, batch in enumerate(nonempty):
                exact = sum(x["value"] for x in batch) / len(batch)
                if method == "Exact":
                    estimate, se, sampled = exact, 0.0, len(batch)
                elif method == "SRS":
                    n = max(1, int(0.6 * len(batch)))
                    sampler = SRSSampler(n, seed=seed + batch_index)
                    sampler.update_batch(batch)
                    estimate = sampler.approximate_mean("value")
                    se, sampled = float("nan"), sampler.sample_size
                else:
                    n_strata = max(1, len({x["sensor_id"] for x in batch}))
                    n = max(1, math.ceil(0.6 * len(batch) / n_strata))
                    sampler = OASRS(n, stratum_key="sensor_id", seed=seed + batch_index)
                    sampler.update_batch(batch)
                    snap = sampler.snapshot()
                    estimate = approximate_mean(snap, "value")
                    se = estimate_mean(snap, "value").standard_error
                    sampled = snap.sample_size
                estimates.append(estimate)
                sampled_total += sampled
                errors.append(abs(estimate - exact) / abs(exact) if exact else 0.0)
                standard_errors.append(se)
            elapsed = max(1e-9, time.perf_counter() - started)
            rows.append({
                "experiment": "batch_interval",
                "dataset": "gaussian_8000_2000_100",
                "method": method,
                "sampling_fraction": 0.6,
                "window_size": interval,
                "slide": interval,
                "input_rate": 1.0 / interval,
                "throughput": len(records) / elapsed,
                "latency": elapsed / max(1, len(nonempty)),
                "sampled_items": sampled_total,
                "total_items": len(records),
                "exact_result": statistics.fmean(
                    sum(x["value"] for x in b) / len(b) for b in nonempty
                ),
                "approximate_result": statistics.fmean(estimates),
                "accuracy_loss": statistics.fmean(errors),
                "estimated_error": statistics.fmean(standard_errors),
            })
    return rows


def write_outputs(rows, outdir, config):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    if not rows:
        return
    with (outdir / "raw_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for key in sorted({(x["experiment"], x["method"]) for x in rows}):
        values = [x for x in rows if (x["experiment"], x["method"]) == key]
        grouped[f"{key[0]}:{key[1]}"] = {
            "runs": len(values),
            "mean_throughput": statistics.fmean(float(x["throughput"]) for x in values),
            "mean_accuracy_loss": statistics.fmean(float(x["accuracy_loss"]) for x in values),
            "mean_estimated_error": statistics.fmean(
                float(x["estimated_error"]) for x in values
                if x["estimated_error"] != "" and math.isfinite(float(x["estimated_error"]))
            ) if any(x["estimated_error"] != "" and math.isfinite(float(x["estimated_error"])) for x in values) else None,
        }
    (outdir / "summary.json").write_text(json.dumps({"groups": grouped}, indent=2, allow_nan=False))
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    fractions = [x for x in rows if x["sampling_fraction"] != ""]
    for metric, ylabel, filename in (("throughput", "Items per second", "throughput.svg"),
                                     ("accuracy_loss", "Relative mean error", "accuracy_loss.svg")):
        series_by_method = {
            method: sorted((x for x in fractions if x["method"] == method),
                           key=lambda x: float(x["sampling_fraction"]))
            for method in ("Exact", "SRS", "OASRS")
        }
        plot_series = [(method, [(float(x["sampling_fraction"]) * 100,
                                  float(x[metric])) for x in series])
                       for method, series in series_by_method.items() if series]
        if not plot_series:
            continue
        width, height = 840, 520
        left, right, top, bottom = 90, 25, 45, 75
        xvals = [p[0] for _, points in plot_series for p in points]
        yvals = [p[1] for _, points in plot_series for p in points if math.isfinite(p[1])]
        xmin, xmax = min(xvals), max(xvals)
        ymin, ymax = (min(yvals), max(yvals)) if yvals else (0.0, 1.0)
        if ymin == ymax:
            ymax = ymin + 1.0
        colors = {"Exact": "#68707a", "SRS": "#d96b27", "OASRS": "#2867a6"}
        sx = lambda x: left + (x - xmin) / max(1e-9, xmax - xmin) * (width - left - right)
        sy = lambda y: height - bottom - (y - ymin) / max(1e-9, ymax - ymin) * (height - top - bottom)
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(metric.replace("_", " ").title())} vs sampling fraction</text>',
                 f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#333"/>',
                 f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Sampling fraction (%)</text>',
                 f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(ylabel)}</text>']
        for tick in range(5):
            y = ymin + (ymax - ymin) * tick / 4
            py = sy(y)
            parts.append(f'<path d="M {left} {py:.2f} H {width-right}" stroke="#ddd"/>')
            parts.append(f'<text x="{left-8}" y="{py+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>')
        for method, points in plot_series:
            valid = [(x, y) for x, y in points if math.isfinite(y)]
            coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in valid)
            if coords:
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>')
            for x, y in valid:
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{colors[method]}"/>')
        for index, (method, _) in enumerate(plot_series):
            y = top + index * 22
            parts.append(f'<path d="M {width-165} {y} h 24" stroke="{colors[method]}" stroke-width="3"/>')
            parts.append(f'<text x="{width-133}" y="{y+4}" font-family="sans-serif" font-size="12">{method}</text>')
        parts.append('</svg>')
        (plots / filename).write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/paper-reproduction"))
    parser.add_argument("--quick", action="store_true", help="small smoke suite")
    args = parser.parse_args()
    if args.records < 1 or args.runs < 1:
        parser.error("records and runs must be positive")
    records_n = min(args.records, 5000) if args.quick else args.records
    runs = min(args.runs, 1) if args.quick else args.runs
    rows = []
    for run in range(runs):
        for arrival in ARRIVALS:
            records = list(create_paper_gaussian_stream(
                records_n, arrival_rates=arrival, seed=args.seed + run,
                time_step=0.01))
            label = "arrival_" + "_".join(str(arrival[x]) for x in ("A", "B", "C"))
            fractions = (0.2, 0.6) if args.quick else FRACTIONS
            for fraction in fractions:
                rows.extend(benchmark_sample(records, experiment="sampling_fraction_" + label,
                                             dataset=label, fraction=fraction,
                                             seed=args.seed + run))
        if args.quick:
            rows.extend(benchmark_windows(records, experiment="window_size", seed=args.seed + run))
        else:
            skew = list(create_paper_skewed_gaussian_stream(records_n, seed=args.seed + run))
            for fraction in FRACTIONS:
                rows.extend(benchmark_sample(skew, experiment="skew_80_19_1",
                                             dataset="skew_80_19_1", fraction=fraction,
                                             seed=args.seed + run))
            rows.extend(benchmark_windows(records, experiment="window_size", seed=args.seed + run))
    config = {
        "paper": "StreamApprox, Middleware 2017, DOI 10.1145/3135974.3135989",
        "runs": runs,
        "records_per_run": records_n,
        "seed": args.seed,
        "sampling_fractions": list(FRACTIONS),
        "arrival_rate_ratios": [list(x.values()) for x in ARRIVALS],
        "window_sizes_seconds": [10, 20, 30, 40],
        "slide_seconds": 5,
    }
    write_outputs(rows, args.output, config)
    print(json.dumps({"rows": len(rows), "output": str(args.output), "runs": runs}))


if __name__ == "__main__":
    main()
