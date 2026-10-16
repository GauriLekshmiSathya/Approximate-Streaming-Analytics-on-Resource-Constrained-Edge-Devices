"""Create research-report charts/tables from measured CSV output."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "results/va_oasrs/synthetic-full"
INTEL = ROOT / "results/va_oasrs/intel2"
PLOTS = ROOT / "results/va_oasrs/plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def svg_line(rows, x, y, group, title, xlabel, ylabel, path):
    w, h = 900, 500
    l, r, t, b = 85, 25, 50, 70
    xv = [float(a[x]) for a in rows]
    yv = [float(a[y]) for a in rows]
    xmin, xmax, ymin, ymax = min(xv), max(xv), min(yv), max(yv)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    margin = (ymax-ymin)*0.1
    ymin -= margin; ymax += margin
    X = lambda v: l+(v-xmin)/(xmax-xmin)*(w-l-r)
    Y = lambda v: h-b-(v-ymin)/(ymax-ymin)*(h-t-b)
    groups = defaultdict(list)
    for a in rows: groups[a[group]].append(a)
    colors = ["#2867a6", "#d96b27", "#16845b", "#8e5aa7", "#b23b48", "#555"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{w/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
           f'<path d="M{l} {t} V{h-b} H{w-r}" stroke="#333" fill="none"/>',
           f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-family="sans-serif">{xlabel}</text>',
           f'<text transform="translate(20 {h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{ylabel}</text>']
    for i in range(6):
        val = ymin+(ymax-ymin)*i/5
        out.append(f'<path d="M{l} {Y(val)} H{w-r}" stroke="#ddd"/><text x="{l-5}" y="{Y(val)+4}" text-anchor="end" font-size="11">{val:.3g}</text>')
    for k,(name,series) in enumerate(sorted(groups.items())):
        series.sort(key=lambda a:float(a[x])); color=colors[k%len(colors)]
        pts=" ".join(f'{X(float(a[x]))},{Y(float(a[y]))}' for a in series)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for a in series: out.append(f'<circle cx="{X(float(a[x]))}" cy="{Y(float(a[y]))}" r="4" fill="{color}"/>')
        out.append(f'<text x="{l+k*165}" y="{h-45}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    out.append('</svg>'); path.write_text("\n".join(out))


syn = read(SYN / "summary.csv")
intel = read(INTEL / "summary.csv")
for ds in sorted({r["dataset"] for r in syn}):
    for query in ("mean",):
        data=[r for r in syn if r["dataset"]==ds and r["query"]==query and r["method"]!="Exact"]
        svg_line(data,"budget","mean_relative_error","method",f"{ds}: relative error vs sample budget", "Sample budget (records)", "Mean relative error",PLOTS/f"{ds}-error.svg")
        svg_line(data,"budget","mean_throughput_records_per_second","method",f"{ds}: processing throughput vs sample budget","Sample budget (records)","Throughput (input records/s)",PLOTS/f"{ds}-throughput.svg")
for query in ("temperature_mean", "light_mean"):
    data=[r for r in intel if r["query"]==query]
    if data:
        svg_line(data,"budget","mean_relative_error","method",f"Intel {query}: relative error vs sample budget","Sample budget (records)","Mean relative error",PLOTS/f"intel-{query}-error.svg")

hetero = read(SYN / "variance_heterogeneity.csv")
hetero_summary = defaultdict(list)
for row in hetero:
    hetero_summary[(row["variance_ratio"], row["method"])].append(float(row["relative_error"]))
hetero_points = [{"variance_ratio": key[0], "method": key[1],
                  "mean_relative_error": sum(vals)/len(vals)}
                 for key, vals in hetero_summary.items()]
svg_line(hetero_points,"variance_ratio","mean_relative_error","method",
         "Relative error vs within-stratum variance ratio","Maximum/low stratum standard-deviation ratio",
         "Mean relative error",PLOTS/"variance-heterogeneity.svg")

allocation_rows = read(SYN / "allocations.csv")
allocation = [r for r in allocation_rows if r["dataset"]=="variance_skew" and int(r["budget"])==200 and int(r["seed"])==11]
if allocation:
    w,h=850,480; left=80; right=35; top=55; bottom=80
    ids=[r["stratum"] for r in allocation]
    maxima=max(max(float(r["online_variance"]) for r in allocation),max(float(r["capacity"]) for r in allocation))
    slot=(w-left-right)/len(ids)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">','<rect width="100%" height="100%" fill="white"/>',
         '<text x="425" y="25" text-anchor="middle" font-size="18" font-family="sans-serif">Variance and VA-OASRS allocation (variance-skew, budget 200)</text>',
         f'<path d="M{left} {top} V{h-bottom} H{w-right}" stroke="#333" fill="none"/>',
         f'<text x="425" y="{h-20}" text-anchor="middle" font-family="sans-serif">Stratum</text>',
         f'<text transform="translate(22 250) rotate(-90)" text-anchor="middle" font-family="sans-serif">Observed scale (variance or records)</text>']
    for i,row in enumerate(sorted(allocation,key=lambda r:r["stratum"])):
        x=left+i*slot+slot*.18
        variance=float(row["online_variance"]); cap=float(row["capacity"])
        bh=(h-top-bottom)*variance/maxima; ch=(h-top-bottom)*cap/maxima
        out += [f'<rect x="{x}" y="{h-bottom-bh}" width="{slot*.28}" height="{bh}" fill="#d96b27"/>',
                f'<rect x="{x+slot*.32}" y="{h-bottom-ch}" width="{slot*.28}" height="{ch}" fill="#2867a6"/>',
                f'<text x="{left+i*slot+slot*.38}" y="{h-bottom+20}" text-anchor="middle" font-family="sans-serif">{row["stratum"]}</text>']
    out += ['<text x="85" y="45" fill="#d96b27" font-family="sans-serif">Orange: sample variance</text>',
            '<text x="300" y="45" fill="#2867a6" font-family="sans-serif">Blue: allocated capacity</text>','</svg>']
    (PLOTS/"stratum-allocation.svg").write_text("\n".join(out))


def fmt(x, digits=4): return f"{float(x):.{digits}g}"

pop = [r for r in syn if r["dataset"]=="population_skew" and r["query"]=="mean"]
var = [r for r in syn if r["dataset"]=="variance_skew" and r["query"]=="mean"]
shift = [r for r in syn if r["dataset"]=="distribution_shift" and r["query"]=="mean"]
report = r'''# Variance-Aware Adaptive Stratified Reservoir Sampling for Apache Flink

## Abstract

This report evaluates a Python reference prototype of variance-aware online adaptive stratified reservoir sampling (VA-OASRS), motivated by the OASRS design in StreamApprox. The prototype uses online per-stratum variance and count estimates to make a single fixed-budget allocation after a burn-in interval. Five-seed synthetic experiments and a five-seed replay of the Intel sensor dataset were conducted. The measurements show that variance-aware allocation can help under some settings, but the current prototype does not establish a consistent improvement over equal-capacity OASRS. It also has materially lower single-process throughput and failed empirical confidence-interval coverage in the Intel study. This repository does not contain a Flink job, so no Flink measurements are claimed.

## 1. Introduction

Processing every record can constrain throughput in stream analytics. StreamApprox proposes Online Adaptive Stratified Reservoir Sampling (OASRS), retaining independent reservoir samples by source and weighting retained items by observed per-source population. This project asks whether online variance estimates can inform the per-stratum sample allocation and reduce estimator error at a fixed sample budget.

Contributions implemented in this repository are a Python OASRS reference, a VA-OASRS fixed-budget prototype, and a reproducible initial evaluation over synthetic and Intel streams. These results are not an Apache Flink implementation or distributed systems evaluation.

## 2. Background

### 2.1 Sampling and OASRS

StreamApprox Algorithm 3 discovers source sub-streams, obtains each stratum capacity from `getSampleSize(sampleSize, S)`, independently applies reservoir sampling, tracks counts, and weights the samples. The paper does not prescribe the cost function's global-to-stratum capacity mapping. In the described weighted mean, stratum i contributes in proportion to its observed count C_i. The source paper also estimates variance using sample variance within each stratum.

### 2.2 Scope of the baseline

The existing `OASRS` class implements one fixed capacity per stratum with Algorithm R. For this evaluation, an additional fixed-budget population-proportional allocation is used as a control. It must not be confused with a uniquely specified budget-allocation algorithm in the paper: the capacity mapping is an evaluation choice because the paper delegates that mapping to its cost function.

## 3. Proposed Method

### 3.1 Allocation

After a configurable burn-in, VA-OASRS allocates a fixed budget B using observed counts C_i and sample standard deviations s_i:

\[
n_i = m + \operatorname{Hamilton}\left((B-Xm)\frac{C_i s_i}{\sum_j C_j s_j}\right),
\]

where X is the number of active strata and m is a one-record minimum. Integer residuals use largest remainders; capacities are capped by observed populations when applicable. Until variance is estimable, the implementation uses s_i=1 as an online prior. Zero total score falls back to equal allocation.

### 3.2 Sampling, weights, and limitations

Once capacities are allocated, each stratum's reservoir uses Algorithm R. A sampled value represents C_i/Y_i observations, so weighted sum is the sum of sample values times C_i/Y_i and weighted mean divides by total count. The capacity allocation is frozen after its first burn-in decision. Repeated resizing of a live reservoir is not implemented because preserving an exact probability sample across changing capacities requires tracking inclusion probabilities or an appropriate priority sampling design. This freeze is a material deviation from a continuously adaptive method.

The online variance is computed with Welford's recurrence in O(1) state per stratum. Allocation costs O(X log X) for integer remainder ordering; each record is O(1) expected reservoir work. State is O(B+X) for fixed B.

```text
for each online record (source i, value x):
    C_i += 1; update Welford variance for x
    if C_total reaches burn_in:
        allocate B capacities from C_i * s_i, minimum 1
        initialize each stratum reservoir from subsequent records
    else if allocation is frozen:
        update stratum i's Algorithm-R reservoir
at query time:
    estimate = sum_i (C_i / Y_i) * sum(sample_i)
    mean = estimate / sum_i C_i
```

### 3.3 Deviations from StreamApprox

1. A Python in-process implementation replaces the paper's Flink/Spark operators.
2. The baseline uses a fixed population-proportional global budget because the paper does not specify `getSampleSize`'s distribution rule.
3. VA-OASRS adds online Welford variance to the allocation score `C_i s_i`.
4. VA-OASRS waits for a burn-in and freezes allocation once; it does not adapt repeatedly through a window.
5. Largest-remainder rounding, minimum per active stratum, and observed-population caps are explicit engineering rules.
6. Normal-approximation confidence intervals are evaluated empirically and are not asserted to be rigorous deterministic bounds.

## 4. Implementation

The code is organized as a sampling library and offline benchmark driver. The natural source key is `sensor_id` for synthetic streams and Intel `mote_id` for the real stream. The data path for a future Flink implementation would be Kafka source → parse → key by source → keyed reservoir/variance state → weighted query/window aggregation → metrics sink. This architecture is not implemented here; keyed state, checkpoint recovery, event-time windows, parallelism, and exactly-once behavior remain future work.

## 5. Experimental Methodology

The primary run used the repository virtual environment on the available macOS host, Python 3.14, and the project code. Flink and Kafka versions are not applicable because neither runtime was used in these measurements. Synthetic experiments used deterministic Python random seeds 11, 23, 37, 53, and 71; each dataset had 10,000 records, five source strata, and budgets 200 and 1,000. Distribution shift doubled the scale of the last two strata after half the stream. The benchmark measured exact, global uniform reservoir, fixed-budget equal-capacity OASRS, and VA-OASRS.

The Intel dataset run parsed 2,210,084 valid records across 52 natural `mote_id` strata. It used five reservoir seeds and budgets 270, 540, 1,080, and 2,160. The source duration and event-time span were not derived in this run; the benchmark is a replay, not a wall-clock source simulation. Query fields were temperature, humidity, light, and voltage means.

All raw rows, configurations, allocations, summaries, and charts are generated from scripts. Confidence intervals across runs use a Student-t interval when n=5. Reported throughput is local single-process Python replay throughput and should not be interpreted as Flink throughput.

## 6. Results

### 6.1 Synthetic results

![Combined-skew mean relative error](../plots/combined_skew-error.svg)

In the 10,000-record synthetic study, at budget 200 on the variance-skew stream, VA-OASRS mean relative error averaged 1.80% (SD 1.63%) versus 3.21% (SD 2.38%) for equal-capacity OASRS. At budget 1,000, the corresponding values were 1.03% (SD 0.89%) and 1.13% (SD 0.71%); the difference is small relative to run-to-run variation. At budget 200 on uniform data, VA-OASRS averaged 0.42% while OASRS averaged 0.62%. On the combined-skew stream at budget 1,000, VA-OASRS averaged 1.27% compared with 0.77% for OASRS, a negative result for VA-OASRS.

These five-run summaries are descriptive. They do not establish statistical significance; a paired confidence-interval CSV is generated separately, and only five seeds are available. The method helps some conditions and hurts others.

### 6.2 Real dataset

![Intel temperature mean relative error](../plots/intel-temperature_mean-error.svg)

The Intel VA-OASRS run is **invalid as a fixed-budget comparison**. The source order is highly nonstationary: the last of the 52 sensors first appears after 2,181,367 valid records. The current one-shot allocator freezes before all sensors are discovered and therefore cannot allocate them. The generated Intel VA-OASRS numbers are retained as a failure diagnostic, not used as evidence of accuracy. Population-proportional OASRS and uniform-reservoir numbers are separately measured, but do not repair this design limitation.

The equal-capacity baseline's temperature error averaged 5.84%, 3.78%, 4.56%, and 1.14% at budgets 270, 540, 1,080, and 2,160. Across fields its coverage ranged between 0.4 and 1.0 depending on budget and query. Those figures use five trials.

### 6.3 Performance

![Combined-skew throughput](../plots/combined_skew-throughput.svg)

In local synthetic replay, VA-OASRS throughput was approximately 1.4–1.7 million input records/s, equal-capacity OASRS approximately 2.3–3.0 million records/s, and the uniform reservoir approximately 4.6–7.2 million records/s across the tested budgets and datasets. The offline driver includes list construction and aggregation as well as ingestion, so these are process-level Python timings, not isolated per-record operator costs. The cost of maintaining online variance and a burn-in dominates this prototype's observed overhead. No CPU utilization or memory profiling was performed in the current study.

### 6.4 Allocation and distribution shift

Raw per-stratum allocations and observed variances are in `results/va_oasrs/synthetic/allocations.csv`. The distribution-shift experiment compares 500-record and 2,000-record burn-in intervals. As allocation is frozen after burn-in, it is a stress test of stale allocation and does not demonstrate drift detection. Full error-vs-heterogeneity ratios, the 1–80% fraction sweep, histogram queries, confidence-bound plots, and a complete accuracy/throughput Pareto analysis are not included in this initial evaluation.

## 7. Ablation Study

The current benchmark compares fixed equal-capacity OASRS with OASRS plus online variance allocation. It does not isolate variance tracking overhead without allocation, nor does it test alternative variance priors or repeated allocation. The specified ablation is therefore incomplete.

## 8. Overhead Analysis

Measured process throughput is lower for VA-OASRS in this implementation. The code also records variance-update and allocation wall time as separate sampler fields, but those microtimings are not currently exported per trial. No claim about CPU utilization or memory savings is made. Given the Intel error and coverage results, the current evidence does not justify the added overhead.

## 9. Limitations

The evaluation is a Python prototype, not Flink. VA-OASRS has a single burn-in allocation and cannot handle repeated capacity changes with provably correct inclusion probabilities. A global sample budget smaller than the number of active strata is unsupported. Variance estimates are query-field specific. The Intel source-order skew means a short burn-in cannot discover all strata; late strata need dynamically allocated capacity. The Intel run is not valid evidence for VA-OASRS accuracy. The 95% interval results are therefore not used to assess calibrated coverage. Histogram, COUNT, and SUM confidence coverage are not tested. System metrics are incomplete, and the full requested sensitivity matrix has not been run.

## 10. Threats to Validity

Synthetic Gaussian distributions and the chosen stratum means/variances may not represent deployment streams. The Intel dataset's file replay order and one-machine Python runtime affect observed throughput. Five seeds provide limited statistical power. The capacity-mapping ambiguity in the paper means equal-capacity OASRS is an explicit control choice, not a paper-mandated allocation. In the Intel runner, `mote_id` is the source stratum. The full dataset is used to calculate ground truth and baseline population allocation. VA-OASRS uses only online pilot information, but the one-shot allocation policy fails under this dataset's delayed stratum discovery.

## 11. Conclusion

This prototype provides a first online variance-allocation experiment, but it does not answer the research question affirmatively. VA-OASRS improved some synthetic cases and regressed others. On the Intel sensor data it had larger temperature error and zero observed confidence-interval coverage across the five seeds, with substantially lower replay throughput. The present design does not justify a claim of statistically significant or unconditional improvement. A proper Flink implementation and a corrected, validated variance/inclusion-probability design are needed before making the final research claim.

## Reproducibility artifacts

- Synthetic raw data: `results/va_oasrs/synthetic/raw_results.csv`
- Synthetic allocation observations: `results/va_oasrs/synthetic/allocations.csv`
- Synthetic summaries: `results/va_oasrs/synthetic/summary.csv`
- Intel raw results and summary: `results/va_oasrs/intel2/`
- Benchmark code: `experiments/va_oasrs/`
- VA-OASRS implementation: `src/streamapprox/va_oasrs.py`

## References

Do Le Quoc, Ruichuan Chen, Pramod Bhatotia, Christof Fetzer, Volker Hilt, and Thorsten Strufe. “StreamApprox: Approximate Computing for Stream Analytics.” ACM Middleware 2017. DOI: [10.1145/3135974.3135989](https://doi.org/10.1145/3135974.3135989).
'''
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report/report.md").write_text(report)
