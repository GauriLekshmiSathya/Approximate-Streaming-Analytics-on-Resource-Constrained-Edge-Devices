"""Generate SVG figures and CSV summaries directly from benchmark raw results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def line_plot(rows, x_key, y_key, group_key, title, xlabel, ylabel, path):
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 55, 80
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda x: left + (x - xmin) / max(1e-12, xmax - xmin) * (width-left-right)
    sy = lambda y: height-bottom - (y-ymin) / max(1e-12, ymax-ymin) * (height-top-bottom)
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    palette = ["#2867a6", "#d96b27", "#16845b", "#8e5aa7", "#b23b48", "#666666"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
           f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#333"/>',
           f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif">{xlabel}</text>',
           f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{ylabel}</text>']
    for i in range(6):
        y = ymin + (ymax-ymin)*i/5
        out.extend([f'<path d="M {left} {sy(y):.2f} H {width-right}" stroke="#e5e5e5"/>',
                    f'<text x="{left-8}" y="{sy(y)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>'])
    for index, (name, series) in enumerate(sorted(groups.items())):
        color = palette[index % len(palette)]
        series = sorted(series, key=lambda row: float(row[x_key]))
        coords = " ".join(f'{sx(float(row[x_key])):.2f},{sy(float(row[y_key])):.2f}' for row in series)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in series:
            out.append(f'<circle cx="{sx(float(row[x_key])):.2f}" cy="{sy(float(row[y_key])):.2f}" r="4" fill="{color}"/>')
        out.append(f'<text x="{left+index*155}" y="{height-50}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/va_oasrs"))
    args = parser.parse_args()
    raw = read_csv(args.results / "raw_results.csv")
    processed = args.results / "processed"
    plots = args.results / "plots"
    processed.mkdir(exist_ok=True); plots.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["dataset"], row["budget"], row["method"], row["query"])].append(row)
    summary = []
    for (dataset, budget, method, query), values in sorted(grouped.items()):
        errors = [float(row["relative_error"]) for row in values]
        mean = sum(errors) / len(errors)
        summary.append({"dataset": dataset, "budget": budget, "method": method,
                        "query": query, "runs": len(values), "mean_relative_error": mean,
                        "sd_relative_error": (sum((x-mean)**2 for x in errors)/(len(errors)-1))**0.5 if len(errors)>1 else 0.0,
                        "mean_throughput_records_per_second": sum(float(r["throughput_records_per_second"]) for r in values)/len(values),
                        "mean_latency_us_per_record": sum(float(r["latency_us_per_record"]) for r in values)/len(values),
                        "mean_sampled": sum(float(r["sampled"]) for r in values)/len(values)})
    with (processed / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    for dataset in sorted({r["dataset"] for r in summary}):
        for query in ("mean", "sum"):
            points = [r for r in summary if r["dataset"] == dataset and r["query"] == query and r["method"] != "Exact"]
            if not points:
                continue
            line_plot(points, "budget", "mean_relative_error", "method",
                      f"{dataset}: relative error vs budget ({query})", "Global sample budget (records)",
                      "Mean relative error (unitless)", plots / f"{dataset}_{query}_error.svg")
            line_plot(points, "budget", "mean_throughput_records_per_second", "method",
                      f"{dataset}: throughput vs budget", "Global sample budget (records)",
                      "Throughput (input records/s)", plots / f"{dataset}_{query}_throughput.svg")
    if (args.results / "variance_heterogeneity.csv").exists():
        hetero = read_csv(args.results / "variance_heterogeneity.csv")
        line_plot(hetero, "variance_ratio", "relative_error", "method",
                  "Relative error vs variance heterogeneity", "Variance ratio",
                  "Relative error (unitless)", plots / "variance_heterogeneity.svg")
    if (args.results / "population_skew.csv").exists():
        pop = read_csv(args.results / "population_skew.csv")
        # Category x-axis is mapped to a stable ordinal for plotting.
        order = {"low": 1, "moderate": 2, "high": 3, "extreme": 4}
        for row in pop:
            row["skew_order"] = order[row["population_skew"]]
        line_plot(pop, "skew_order", "relative_error", "method",
                  "Relative error vs population skew", "Population skew (low to extreme)",
                  "Relative error (unitless)", plots / "population_skew_sensitivity.svg")


if __name__ == "__main__":
    main()
