"""Build DV-OASRS figures from saved CSV measurements (no hand-entered results)."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dv_oasrs/plots"
SYN = ROOT / "results/dv_oasrs/synthetic"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_chart(path: Path, title: str, xlabel: str, ylabel: str,
                series: dict[str, list[tuple[float, float, float | None]]],
                xlabels: dict[float, str] | None = None) -> None:
    width, height = 960, 590
    left, right, top, bottom = 100, 35, 58, 100
    xs = [x for points in series.values() for x, _, _ in points]
    ys = [y for points in series.values() for _, y, _ in points]
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmin -= 0.5; xmax += 0.5
    ymin, ymax = min(0.0, min(ys)), max(ys)
    if ymin == ymax:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.1
    ymin -= pad * 0.2; ymax += pad
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (width-left-right)
    sy = lambda y: height-bottom - (y-ymin) / (ymax-ymin) * (height-top-bottom)
    colors = ["#245b9e", "#d06425", "#16845b", "#8953a6", "#b33b45", "#555555"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>']
    for tick in range(6):
        y = ymin + (ymax-ymin)*tick/5
        out += [f'<path d="M {left} {sy(y):.1f} H {width-right}" stroke="#e5e7eb"/>',
                f'<text x="{left-12}" y="{sy(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.3g}</text>']
    out += [f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#333" stroke-width="1.3"/>',
            f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(xlabel)}</text>',
            f'<text transform="translate(25 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(ylabel)}</text>']
    for idx, (label, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = sorted(points)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.7"/>')
        for x, y, err in points:
            if err is not None:
                ylo, yhi = sy(y-err), sy(y+err)
                out.append(f'<path d="M {sx(x):.1f} {ylo:.1f} V {yhi:.1f} M {sx(x)-4:.1f} {ylo:.1f} H {sx(x)+4:.1f} M {sx(x)-4:.1f} {yhi:.1f} H {sx(x)+4:.1f}" stroke="{color}"/>')
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
    labels = xlabels or {}
    tick_values = sorted(set(xs))
    if len(tick_values) > 12:
        tick_values = tick_values[::max(1, len(tick_values)//10)]
    for x in tick_values:
        out.append(f'<text x="{sx(x):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(labels.get(x, f"{x:g}"))}</text>')
    legends = list(series)
    per_row = max(1, int((width-left-right)/190))
    for idx, label in enumerate(legends):
        row, col = divmod(idx, per_row)
        lx, ly = left + col*190, height-58 + row*18
        out += [f'<path d="M {lx} {ly} h 22" stroke="{colors[idx%len(colors)]}" stroke-width="3"/>',
                f'<text x="{lx+28}" y="{ly+4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>']
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))


def aggregate(rows: list[dict[str, str]], xkey: str, ykey: str,
              groupkey: str, filter_fn=lambda row: True) -> dict[str, list[tuple[float, float, float | None]]]:
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        if filter_fn(row):
            values[(row[groupkey], float(row[xkey]))].append(float(row[ykey]))
    result: dict[str, list[tuple[float, float, float | None]]] = defaultdict(list)
    for (group, x), ys in values.items():
        mean = sum(ys)/len(ys)
        # Small-sample t multiplier for 5-seed runs; normal approximation otherwise.
        t = 2.776 if len(ys) == 5 else 1.96
        sd = math.sqrt(sum((y-mean)**2 for y in ys)/(len(ys)-1)) if len(ys) > 1 else 0.0
        result[group].append((x, mean, t*sd/math.sqrt(len(ys))))
    return dict(result)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read(SYN / "summary.csv")
    raw = read(SYN / "raw_results.csv")
    hetero = read(SYN / "variance_heterogeneity.csv")
    pop = read(SYN / "population_skew.csv")
    allocations = read(SYN / "allocations.csv")
    intel = read(ROOT / "results/dv_oasrs/intel/allocation_over_time.csv")
    late = read(ROOT / "results/dv_oasrs/late-strata/allocation_over_time.csv")

    for name, metric, ylabel in (("variance_skew", "mean_relative_error", "Mean relative error (unitless)"),
                                 ("variance_skew", "mean_throughput_records_per_second", "Replay throughput (records/s)")):
        rows = [r for r in summary if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["method"] != "Exact"]
        write_chart(OUT / ("sampling_budget_error.svg" if metric == "mean_relative_error" else "sampling_budget_throughput.svg"),
                    "Variance-skew stream: " + ("error vs sample budget" if metric == "mean_relative_error" else "throughput vs sample budget"),
                    "Global sampling budget (records)", ylabel,
                    aggregate(rows, "budget", metric, "method"))

    latency = [r for r in summary if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["method"] != "Exact"]
    write_chart(OUT / "latency_vs_budget.svg", "Variance-skew stream: replay latency vs sample budget",
                "Global sampling budget (records)", "Replay time (microseconds/input record)",
                aggregate(latency, "budget", "mean_latency_us_per_record", "method"))

    # The raw file stores two query rows per run; use one row per run for the sampler microtimings.
    overhead_rows = []
    for r in raw:
        if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["method"] == "DV-OASRS@500":
            for key, label in (("variance_seconds", "Variance update"), ("allocation_seconds", "Allocation"), ("priority_seconds", "Priority reservoir")):
                overhead_rows.append({"budget": r["budget"], "component": label,
                                      "seconds_per_record_us": float(r[key]) * 1e6 / float(r["records"])})
    write_chart(OUT / "sampling_overhead_breakdown.svg", "DV-OASRS instrumented component time",
                "Global sampling budget (records)", "Component time (microseconds/input record)",
                aggregate(overhead_rows, "budget", "seconds_per_record_us", "component"))

    write_chart(OUT / "variance_heterogeneity_error.svg", "Mean error under increasing variance heterogeneity",
                "Highest-to-lowest stratum variance ratio", "Mean relative error (unitless)",
                aggregate(hetero, "variance_ratio", "relative_error", "method"))
    paired: dict[tuple[float, str], list[float]] = defaultdict(list)
    lookup = {(float(r["variance_ratio"]), int(r["seed"]), r["method"]): float(r["relative_error"]) for r in hetero}
    for ratio, seed, method in list(lookup):
        if method != "OASRS":
            paired[(ratio, method)].append(lookup[(ratio, seed, "OASRS")] - lookup[(ratio, seed, method)])
    improvement_rows = [{"variance_ratio": x, "method": method, "improvement": v}
                        for (x, method), values in paired.items() for v in values]
    write_chart(OUT / "variance_heterogeneity_improvement.svg", "Paired error reduction relative to OASRS",
                "Highest-to-lowest stratum variance ratio", "OASRS error − method error (positive is improvement)",
                aggregate(improvement_rows, "variance_ratio", "improvement", "method"))
    order = {"low": 1, "moderate": 2, "high": 3, "extreme": 4}
    for row in pop:
        row["skew_order"] = str(order[row["population_skew"]])
    write_chart(OUT / "population_skew_error.svg", "Error under population skew (variance approximately controlled)",
                "Population imbalance", "Mean relative error (unitless)",
                aggregate(pop, "skew_order", "relative_error", "method"),
                {1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"})

    # Latest per-stratum state from one fixed budget/seed; event data is online as of that event.
    alloc = [r for r in allocations if r["dataset"] == "variance_skew" and r["method"] == "DV-OASRS@500" and r["budget"] == "200" and r["seed"] == "11"]
    latest = {}
    for row in alloc:
        sid = row["stratum"]
        if sid not in latest or int(row["records_seen"]) > int(latest[sid]["records_seen"]): latest[sid] = row
    ordered_latest = [latest[sid] for sid in sorted(latest)]
    write_chart(OUT / "per_stratum_variance.svg", "Online Welford variance by stratum",
                "Stratum index", "Variance (value units squared)",
                {"Welford variance": [(float(i+1), float(r["online_variance"]), None) for i, r in enumerate(ordered_latest)]})
    write_chart(OUT / "per_stratum_allocation.svg", "DV-OASRS capacity by stratum",
                "Stratum index", "Allocated sample records",
                {"DV-OASRS capacity": [(float(i+1), float(r["capacity"]), None) for i, r in enumerate(ordered_latest)]})

    # Variance-shift allocation trajectory, one seed and budget to avoid overplotting.
    shift = [r for r in allocations if r["dataset"] == "distribution_shift" and r["method"] == "DV-OASRS@500" and r["budget"] == "200" and r["seed"] == "11"]
    for row in shift:
        row["records_k"] = str(float(row["records_seen"])/1000)
    write_chart(OUT / "allocation_over_time.svg", "DV-OASRS allocation through a variance shift",
                "Records observed (thousands)", "Allocated samples per stratum",
                aggregate(shift, "records_k", "capacity", "stratum"))
    late_dv = [r for r in late if r["method"] == "DV-OASRS" and r["stratum"] in ("S5", "S6")]
    for row in late_dv: row["records_k"] = str(float(row["records_seen"])/1000)
    write_chart(OUT / "late_strata_allocation.svg", "Allocation to late-arriving strata",
                "Records observed (thousands)", "Allocated samples",
                aggregate(late_dv, "records_k", "capacity", "stratum"))
    sensor = [r for r in intel if r["mote_id"] == "54" and r["seed"] == "100" and r["budget"] == "270"]
    for row in sensor: row["records_m"] = str(float(row["records_seen"])/1e6)
    write_chart(OUT / "intel_late_sensor_allocation.svg", "DV-OASRS allocation to Intel mote 54",
                "Valid records observed (millions)", "Allocated sample slots",
                aggregate(sensor, "records_m", "capacity", "mote_id"))

    # CI coverage, system replay rates and a descriptive Pareto curve.
    coverage = [r for r in summary if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["mean_ci95_coverage"] not in ("", "None") and r["method"] != "Exact"]
    write_chart(OUT / "confidence_interval_coverage.svg", "Empirical 95% interval coverage for mean estimates",
                "Global sampling budget (records)", "Coverage fraction across five seeds",
                aggregate(coverage, "budget", "mean_ci95_coverage", "method"))
    bound_groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in raw:
        if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["estimated_se"] and r["method"] != "Exact":
            bound_groups[(r["method"], r["budget"])].append((1.96*float(r["estimated_se"]), float(r["absolute_error"])))
    bound_series: dict[str, list[tuple[float, float, float | None]]] = defaultdict(list)
    for (method, _budget), values in bound_groups.items():
        bound_series[method].append((sum(x for x, _ in values)/len(values), sum(y for _, y in values)/len(values), None))
    write_chart(OUT / "actual_error_vs_estimated_bound.svg", "Observed mean absolute error vs estimated 95% bound",
                "Mean estimated 95% error bound (value units)", "Mean actual absolute error (value units)",
                dict(bound_series))
    perf = [r for r in summary if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["method"] != "Exact"]
    pareto = {}
    for row in perf:
        pareto.setdefault(row["method"], []).append((float(row["mean_throughput_records_per_second"]), float(row["mean_relative_error"]), None))
    write_chart(OUT / "accuracy_throughput_pareto.svg", "Accuracy–throughput trade-off (variance-skew replay)",
                "Python replay throughput (records/s)", "Mean relative error (unitless)", pareto)
if __name__ == "__main__":
    main()
