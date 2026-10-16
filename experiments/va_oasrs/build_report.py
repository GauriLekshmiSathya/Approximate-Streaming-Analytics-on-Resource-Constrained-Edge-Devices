"""Render report tables from experiment CSVs and retain prose/methodology in source."""

from __future__ import annotations

import csv
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/dv_oasrs"
REPORT = ROOT / "report/report.md"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def md_table(headers: list[str], values: list[list[object]]) -> str:
    fmt = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(fmt(x) for x in row) + " |" for row in values)


def pct(x: str | float) -> str:
    return f"{100*float(x):.2f}%"


def f3(x: str | float) -> str:
    return f"{float(x):.4f}"


def main() -> None:
    summary = rows(BASE / "synthetic/summary.csv")
    hetero = rows(BASE / "synthetic/variance_heterogeneity.csv")
    pop = rows(BASE / "synthetic/population_skew.csv")
    intel = rows(BASE / "intel/summary.csv")
    late = rows(BASE / "late-strata/summary.csv")
    intel_config = __import__("json").loads((BASE / "intel/config.json").read_text())
    intel_first = rows(BASE / "intel/first_appearance.csv")

    hetero_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in hetero:
        hetero_groups[(r["variance_ratio"], r["method"])].append(float(r["relative_error"]))
    hetero_table = []
    for ratio in ("1.0", "2.0", "5.0", "10.0", "50.0", "100.0"):
        for method in ("OASRS", "Static VA-OASRS", "DV-OASRS", "Uniform Reservoir"):
            vals = hetero_groups[(ratio, method)]
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals)
            half = 2.776 * sd / math.sqrt(len(vals))
            hetero_table.append([ratio.removesuffix(".0"), method, f"{mean:.5f}",
                                 f"[{max(0, mean-half):.5f}, {mean+half:.5f}]", len(vals)])

    budget_table = []
    for budget in (200, 1000, 2000, 4000):
        for method in ("OASRS", "Static VA-OASRS@500", "DV-OASRS@500", "Uniform Reservoir"):
            r = next(x for x in summary if x["dataset"] == "variance_skew" and x["query"] == "mean" and int(x["budget"]) == budget and x["method"] == method)
            budget_table.append([budget, method, pct(r["mean_relative_error"]),
                                 f"{float(r['mean_throughput_records_per_second']):,.0f}",
                                 pct(r["mean_ci95_coverage"]) if r["mean_ci95_coverage"] else "—"])

    population_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in pop:
        population_groups[(r["population_skew"], r["method"])].append(float(r["relative_error"]))
    population_table = [[level, method, pct(statistics.fmean(population_groups[(level, method)]))]
                        for level in ("low", "moderate", "high", "extreme")
                        for method in ("OASRS", "Static VA-OASRS", "DV-OASRS", "Uniform Reservoir")]

    intel_table = []
    for budget in (270, 540, 1080, 2160):
        for method in ("OASRS-equal", "DV-OASRS", "Uniform Reservoir"):
            r = next(x for x in intel if x["query"] == "temperature_mean" and int(x["budget"]) == budget and x["method"] == method)
            intel_table.append([budget, method, pct(r["mean_relative_error"]),
                                f"{float(r['mean_throughput_records_per_second']):,.0f}",
                                pct(r["ci95_coverage"])])
    intel_feature_table = []
    for budget in (270, 540, 1080, 2160):
        for method in ("OASRS-equal", "DV-OASRS", "Uniform Reservoir"):
            metrics = {}
            for field in ("temperature", "humidity", "light", "voltage"):
                r = next(x for x in intel if int(x["budget"]) == budget and x["method"] == method and x["query"] == field + "_mean")
                metrics[field] = pct(r["mean_relative_error"])
            intel_feature_table.append([budget, method, metrics["temperature"], metrics["humidity"], metrics["light"], metrics["voltage"]])

    shift_table = []
    for budget in (200, 1000, 2000, 4000):
        for method in ("OASRS", "Static VA-OASRS@500", "DV-OASRS@500", "DV-OASRS@2000", "Uniform Reservoir"):
            r = next(x for x in summary if x["dataset"] == "distribution_shift" and x["query"] == "mean" and int(x["budget"]) == budget and x["method"] == method)
            shift_table.append([budget, method, pct(r["mean_relative_error"])])

    late_table = [[r["method"], r["runs"], pct(r["mean_relative_error"]),
                   f"[{pct(r['ci95_low'])}, {pct(r['ci95_high'])}]", f"{float(r['mean_sampled']):.0f}",
                   f"S5: {float(r['mean_late_S5_capacity']):.1f}",
                   f"S6: {float(r['mean_late_S6_capacity']):.1f}",
                   f"S5: {float(r['mean_late_S5_first_sample_position']):.1f}" if r["mean_late_S5_first_sample_position"] else "—",
                   f"S6: {float(r['mean_late_S6_first_sample_position']):.1f}" if r["mean_late_S6_first_sample_position"] else "—"] for r in late]

    throughput_rows = [r for r in summary if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["budget"] == "200"]
    throughput_table = [[r["method"], f"{float(r['mean_throughput_records_per_second']):,.0f}",
                         f"{float(r['mean_latency_us_per_record']):.3f}", pct(r["mean_relative_error"])]
                        for r in throughput_rows]
    comparison_rows = []
    comparison_specs = [
        ("Exact", "Full stream", "—", "—", "Not applicable"),
        ("Uniform Reservoir", "Global priority/reservoir sample", "No", "No", "Not measured"),
        ("OASRS", "Equal-capacity stratified Algorithm R", "No", "No", "Not measured"),
        ("Static VA-OASRS@500", "One-shot variance-aware stratified", "Yes", "No", "Not measured"),
        ("DV-OASRS@500", "Dynamic priority stratified", "Yes", "Yes", "Not measured"),
    ]
    for method, strategy, variance_aware, dynamic, memory in comparison_specs:
        r = next(x for x in throughput_rows if x["method"] == method)
        comparison_rows.append([method, strategy, variance_aware, dynamic,
                                f"{float(r['mean_throughput_records_per_second']):,.0f}",
                                pct(r["mean_relative_error"]), memory])

    def sm(dataset: str, budget: int, method: str) -> dict[str, str]:
        return next(r for r in summary if r["dataset"] == dataset and r["query"] == "mean" and int(r["budget"]) == budget and r["method"] == method)

    hetero_means = {(ratio, method): statistics.fmean(hetero_groups[(ratio, method)])
                    for ratio in ("1.0", "2.0", "5.0", "10.0", "50.0", "100.0")
                    for method in ("OASRS", "Static VA-OASRS", "DV-OASRS", "Uniform Reservoir")}
    intel_temp = {(int(r["budget"]), r["method"]): r for r in intel if r["query"] == "temperature_mean"}
    mote54_first = next(r["first_position"] for r in intel_first if r["mote_id"] == "54")
    var_1_dv, var_1_oasrs = hetero_means[("1.0", "DV-OASRS")], hetero_means[("1.0", "OASRS")]
    var_100_dv, var_100_oasrs = hetero_means[("100.0", "DV-OASRS")], hetero_means[("100.0", "OASRS")]
    var_100_reduction = (var_100_oasrs-var_100_dv)/var_100_oasrs
    late_oasrs = next(r for r in late if r["method"] == "OASRS")
    late_static = next(r for r in late if r["method"] == "Static VA-OASRS")
    late_dv = next(r for r in late if r["method"] == "DV-OASRS")
    throughput_mean = {r["method"]: float(r["mean_throughput_records_per_second"]) for r in throughput_rows}
    coverage_values = [float(r["mean_ci95_coverage"]) for r in summary
                       if r["dataset"] == "variance_skew" and r["query"] == "mean" and r["method"].startswith("DV-OASRS")]

    version = platform.python_version()
    machine = f"{platform.system()} {platform.machine()} ({platform.processor() or 'processor model unavailable'})"
    figures = """
![Relative error vs sampling budget](../results/dv_oasrs/plots/sampling_budget_error.svg)
*Figure 1. Relative error for the variance-skew mean query; points show five-run means and bars show 95% Student-t intervals.*

![Throughput vs sampling budget](../results/dv_oasrs/plots/sampling_budget_throughput.svg)
*Figure 2. Single-process Python replay throughput. This is not Flink or Kafka throughput.*

![Latency vs sampling budget](../results/dv_oasrs/plots/latency_vs_budget.svg)
*Figure 3. Measured replay time per input record, derived as total replay time divided by input records.*

![Variance heterogeneity](../results/dv_oasrs/plots/variance_heterogeneity_error.svg)
*Figure 4. Mean relative error as the high-to-low stratum variance ratio grows at a fixed budget of 200 records.*

![Improvement relative to OASRS](../results/dv_oasrs/plots/variance_heterogeneity_improvement.svg)
*Figure 5. Paired OASRS error minus method error over the same five seeds; positive values indicate lower error.*

![Population skew](../results/dv_oasrs/plots/population_skew_error.svg)
*Figure 6. Error under population imbalance while stratum variances are approximately equal.*

![Per-stratum variance](../results/dv_oasrs/plots/per_stratum_variance.svg)
*Figure 7. Final online Welford variance by stratum for a fixed seed and budget.*

![Per-stratum allocation](../results/dv_oasrs/plots/per_stratum_allocation.svg)
*Figure 8. Final DV-OASRS sample capacity by stratum for the same run.*

![Distribution shift allocation](../results/dv_oasrs/plots/allocation_over_time.svg)
*Figure 9. DV-OASRS allocation history around the synthetic variance shift at 10,000 records.*

![Late strata](../results/dv_oasrs/plots/late_strata_allocation.svg)
*Figure 10. Capacities assigned to S5 and S6 after their first appearances.*

![Intel late sensor](../results/dv_oasrs/plots/intel_late_sensor_allocation.svg)
*Figure 11. DV-OASRS assigns capacity to Intel mote 54 when it first appears in the stream.*

![Confidence interval coverage](../results/dv_oasrs/plots/confidence_interval_coverage.svg)
*Figure 12. Empirical coverage of the approximate 95% normal interval over five seeds.*

![Accuracy throughput frontier](../results/dv_oasrs/plots/accuracy_throughput_pareto.svg)
*Figure 13. Accuracy-throughput points across sample budgets. Curves are descriptive and machine-specific.*

![Sampling overhead breakdown](../results/dv_oasrs/plots/sampling_overhead_breakdown.svg)
*Figure 14. Instrumented DV-OASRS variance update, allocation, and priority-reservoir time per record.*

![Actual error vs estimated bound](../results/dv_oasrs/plots/actual_error_vs_estimated_bound.svg)
*Figure 15. Mean observed absolute error against the sampler's estimated 95% error bound, grouped by method and budget.*
"""

    report = f'''# Dynamic Variance-Aware OASRS for Apache Flink

## Abstract

This revision evaluates Dynamic Variance-Aware OASRS (DV-OASRS), a priority-reservoir extension of StreamApprox's OASRS. DV-OASRS allocates a fixed global sample budget using online per-stratum counts and exponentially smoothed Welford variance, then periodically changes the active prefix of each stratum's priority reservoir. Five-seed Python experiments cover variance heterogeneity, population skew, combined skew, a variance shift, late-arriving strata, and the Intel sensor stream. DV-OASRS improves on equal-capacity OASRS in some high-heterogeneity synthetic settings, but regresses at low heterogeneity and on Intel temperature accuracy; it is also slower in the Python replay. A single-threaded Flink job reference has been added, but could not be compiled or run in this environment. The available evidence therefore supports a conditional accuracy benefit in selected synthetic settings, not a general or Flink-validated improvement.

## 1. Introduction

Stream analytics can spend substantial compute processing records that contribute little to an approximate answer. StreamApprox applies Online Adaptive Stratified Reservoir Sampling (OASRS) to retain per-source samples and estimate aggregates with stratum weights. This project asks whether repeated allocation using online variance can reduce error at a fixed total sample size, especially when variance differs across sources or changes over time.

The revised contributions that are implemented are: (1) preserved OASRS and static VA-OASRS reference baselines; (2) a Python DV-OASRS priority-reservoir implementation with online adaptation and late-stratum discovery; (3) fixed-budget synthetic and Intel experiments with raw results and generated plots; and (4) a Flink DV-OASRS reference job using managed state. The Flink job is source-only and unverified; no Flink performance contribution is claimed.

## 2. Background

### 2.1 Approximate stream processing

Approximate processing retains a subset of records and estimates a query, trading accuracy for resource use. A valid evaluation compares estimates against exact results computed independently from the same input.

### 2.2 Reservoir sampling

Algorithm R maintains a uniform fixed-size sample in one pass when its capacity is fixed. Changing a live reservoir's capacity requires care: shrinking a uniform reservoir to a random subset remains uniform, but discarded records cannot be recovered on later expansion.

### 2.3 Stratified sampling

Stratification keeps source-specific samples. For stratum i with C_i observed records and n_i sampled records, its expansion estimate for a total is C_i times the sample mean. This protects rare strata from being missed by a single global reservoir, subject to the allocation.

### 2.4 StreamApprox OASRS

StreamApprox Algorithm 3 discovers source substreams, obtains a per-stratum capacity through its `getSampleSize(sampleSize, S)` cost function, maintains reservoirs and counts, and combines weighted results. The paper does not mandate one universal cost function mapping a global budget to per-stratum capacities. Our OASRS baseline therefore uses an explicit equal-capacity fixed-budget control and is identified as such. The paper's variance estimation informs the reported uncertainty; the change proposed here is to use variance in allocation.

## 3. Motivation

The initial static VA-OASRS prototype allocated once after a burn-in and then froze capacities. It performed poorly when the Intel dataset's last sensor appeared at valid record position 2,181,366, after the one-shot allocation. Historical files under `results/va_oasrs/intel2/` preserve the resulting near-95% temperature error; this run is invalid as a static VA-OASRS accuracy comparison and is excluded from the final claims. Earlier synthetic results were mixed, and the prototype also had lower throughput than OASRS. Finally, naively resizing an Algorithm-R reservoir cannot recover records omitted under its former smaller capacity.

These failures motivate dynamic discovery and reallocation, but they do not imply that dynamic variance allocation will improve every query or distribution.

## 4. Proposed method

### 4.1 Variance-aware allocation

For each active stratum, DV-OASRS maintains count C_i and online mean/M2 via Welford's recurrence. It maintains an exponential moving variance and scores each stratum as `C_i * sqrt(EMA(variance_i))`. A small positive numerical floor is used for a newly observed or zero-variance stratum. Largest-remainder allocation converts scores to integer capacities subject to minimum capacity, per-stratum budget cap, and observed-population cap. Deterministic tie-breaking makes allocation reproducible for fixed scores and state.

### 4.2 Dynamic reallocation and stability

The default Python interval is 10,000 global records; synthetic sensitivity runs use 500 and 2,000 records. Intel uses 10,000. A newly seen stratum triggers an immediate allocation. At periodic allocation events, a maximum capacity movement (25% of B per interval) limits oscillation. The default EMA coefficient is 0.01, except the late-stratum stress test, which uses 0.02. The distribution-shift test is a stress test of response, not drift detection.

### 4.3 New strata

A new stratum immediately receives its own statistical state and at least the configured minimum allocation, subject to B being at least the active-stratum count. Its candidate pool begins with its first record. If active strata exceed B, the implementation raises an error instead of silently violating the minimum.

### 4.4 Reservoir semantics and estimator

Each record receives an independent U(0,1) priority. Each stratum retains its B highest priorities in a bounded min-heap; for current capacity n_i, the active sample is the n_i highest priorities. For every fixed finite stratum and n_i, these priorities select a simple random sample without replacement. Allocation uses online values/counts but is independent of the random priorities, so conditional on the finite observed stratum values and selected n_i, the sample mean remains unbiased. The record inclusion probability is n_i/C_i and each selected record has expansion weight C_i/n_i. This argument supports stratified totals and means; it does not automatically establish validity for arbitrary adaptive queries or confidence procedures.

The cap B is kept per stratum to permit later capacity increases without restoring discarded values. Consequently candidate storage is O(XB) in the worst case, not O(B); the estimator uses only the active prefixes. Actual sampled records remain at most B once the stream has enough records.

```text
for record (stratum i, value x):
    C_i += 1; update Welford and EMA variance from x
    assign independent uniform priority u; retain top B priorities for i
    if i is new or global_count - last_allocation >= interval:
        score_i = C_i * sqrt(max(EMA_variance_i, epsilon))
        n_i = capped_largest_remainder(B, scores, populations, minimums)
        record the allocation event
at query time:
    estimate_total = sum_i C_i * mean(active_priority_sample_i)
    estimate_mean = estimate_total / sum_i C_i
```

### 4.5 Complexity

Each input record updates one Welford/EMA state and one priority heap, O(log B) heap work, plus allocation work O(X log X) at an allocation event. Worst-case priority state is O(XB). Allocation and priority operations are separately timed in raw results. The Python implementation uses all values for online variance; ground-truth query results are never read by the sampler.

## 5. Apache Flink architecture

The Java reference job is in `flink/src/main/java/org/streamapprox/flink/DVOASRSJob.java`. It reads JSON from Kafka, keys all records to a single global allocator, stores per-stratum counts/variance/priority candidates in Flink `MapState`, stores global counters in `ValueState`, and emits weighted mean summaries to Kafka. Checkpointing is configured every 10 seconds in exactly-once checkpoint mode with local filesystem checkpoint storage by default. The Kafka sink is not configured transactionally, so end-to-end exactly-once output is not claimed. Event time and windows are not implemented; it reports every configured number of records. Global allocation is serialized at parallelism one. The job was not compiled or run because Maven and the Flink CLI are unavailable in the environment. No Flink baseline jobs or Flink system metrics are present; the final experimental comparison below is Python-only.

```mermaid
flowchart LR
    K[Kafka input] --> S[Flink KafkaSource]
    S --> P[Parse JSON and identify mote/source]
    P --> G[Global keyed process operator, parallelism 1]
    G --> V[Managed Welford and EMA state]
    G --> R[Per-stratum priority candidates]
    V --> A[Periodic constrained allocation]
    R --> A
    A --> Q[Weighted mean estimate]
    Q --> O[Kafka result sink]
    G -. checkpointed state .-> C[Checkpoint storage]
```

The implementation currently provides only the DV-OASRS Flink job. Exact, uniform-reservoir, OASRS, and static-VA Flink counterparts, Kafka lag, resource profiles, state-size measurements, fault recovery experiments, and end-to-end latency are not implemented.

## 6. Experimental methodology

The Python replay ran on {machine}, Python {version}, using the repository virtual environment. Flink/Kafka versions are specified in `flink/pom.xml` but were not executed. Synthetic streams contain 20,000 observations, five source IDs, deterministic Gaussian generation, and seeds 11, 23, 37, 53, and 71. The main budget sweep uses 200, 1,000, 2,000, and 4,000 records (1%, 5%, 10%, and 20%); it does not cover the earlier requested 40–80% budgets. There are five independent seeds. Reported cross-run intervals are 95% Student-t intervals with df=4; they are descriptive and no significance tests are asserted.

The variance-heterogeneity study fixes B=200 and varies the high-to-low within-stratum variance ratio across 1, 2, 5, 10, 50, and 100. The population-skew study uses low, moderate, high, and extreme configured source proportions. Combined skew uses 60/20/10/5/5 proportions with three higher-variance strata. The shift stream multiplies standard deviation by eight for two strata after record 10,000. Exact means/sums are computed from the complete stream independently; samplers only consume records sequentially.

The synthetic OASRS configuration receives the natural synthetic source-key registry from the controlled generator. The Intel equal-capacity OASRS capacity map is based on the observed source registry from the loaded file; this is an offline source-universe convenience and a threat to strict no-lookahead parity. DV-OASRS itself discovers `mote_id` values online and does not use final counts or the exact answers. Intel records use fields `temperature`, `humidity`, `light`, and `voltage`; allocation is driven by temperature variance for all four reported mean queries. Results are five seeds at budgets 270, 540, 1,080, and 2,160.

## 7. Results

### 7.1 Generated sampling-budget results

The table is rendered from `results/dv_oasrs/synthetic/summary.csv`; every row is five-run mean error, mean local replay throughput, and empirical interval coverage.

{md_table(["Budget", "Method", "Mean RE", "Replay records/s", "95% CI coverage"], budget_table)}

{figures}

### 7.2 Variance heterogeneity

The central result is conditional. DV-OASRS is worse than OASRS at ratio 1 ({pct(var_1_dv)} versus {pct(var_1_oasrs)} relative error), but its five-run mean error is lower at ratios 10, 50, and 100. At ratio 100 the measured means are {pct(var_100_dv)} for DV-OASRS versus {pct(var_100_oasrs)} for OASRS, an observed {pct(var_100_reduction)} relative reduction. At ratio 2, static VA-OASRS is the lowest-error method; at ratio 5 the three stratified methods are close. The five-run intervals are broad and overlap in several cases, so these differences are descriptive, not statistically significant evidence.

{md_table(["Variance ratio", "Method", "Mean RE", "95% t interval", "Runs"], hetero_table)}

### 7.3 Population and combined skew

DV-OASRS improves over equal-capacity OASRS at moderate/high/extreme population skew in these generated streams, but OASRS has lower error at low skew. Static VA-OASRS edges DV-OASRS at high skew. Under combined skew at budget 200, DV-OASRS mean relative error is {pct(next(r['mean_relative_error'] for r in summary if r['dataset']=='combined_skew' and r['query']=='mean' and r['budget']=='200' and r['method']=='DV-OASRS@500'))}, compared with {pct(next(r['mean_relative_error'] for r in summary if r['dataset']=='combined_skew' and r['query']=='mean' and r['budget']=='200' and r['method']=='OASRS'))} for OASRS. This one case supports the hypothesis, not universal improvement.

{md_table(["Population skew", "Method", "Mean RE over five seeds"], population_table)}

### 7.4 Late-arriving strata

In the controlled six-source stream, S5 begins at record 100,000 and S6 at 200,000, with B=300 and a 10,000-record interval. DV-OASRS allocates to each new stratum on discovery; per-run first sampled-record counts are saved in the raw output. Across five seeds, its mean final capacities are {float(late_dv['mean_late_S5_capacity']):.1f} for S5 and {float(late_dv['mean_late_S6_capacity']):.1f} for S6. Static VA-OASRS assigns neither late stratum capacity and has {pct(late_static['mean_relative_error'])} mean relative error; OASRS and DV-OASRS have {pct(late_oasrs['mean_relative_error'])} and {pct(late_dv['mean_relative_error'])} respectively. DV-OASRS uses a mean of {float(late_dv['mean_sampled']):.0f} slots at completion.

{md_table(["Method", "Runs", "Mean RE", "95% t interval", "Mean sampled", "Mean S5 capacity", "Mean S6 capacity", "S5 first sample count", "S6 first sample count"], late_table)}

### 7.5 Distribution shift and interval sensitivity

At B=200, DV-OASRS error is {pct(sm('distribution_shift', 200, 'DV-OASRS@500')['mean_relative_error'])} for both tested intervals (500 and 2,000); OASRS is {pct(sm('distribution_shift', 200, 'OASRS')['mean_relative_error'])}, static VA-OASRS is {pct(sm('distribution_shift', 200, 'Static VA-OASRS@500')['mean_relative_error'])} with interval 500 and {pct(sm('distribution_shift', 200, 'Static VA-OASRS@2000')['mean_relative_error'])} with interval 2,000, while uniform reservoir is {pct(sm('distribution_shift', 200, 'Uniform Reservoir')['mean_relative_error'])}. At B=4,000, static VA-OASRS at interval 500 and OASRS both outperform DV-OASRS in this run. The allocation trajectory reacts to larger post-shift variance, but no explicit drift detector or general recovery guarantee exists. We have not summarized adaptation time with a preregistered threshold; the allocation trace is provided for inspection instead.

{md_table(["Budget", "Method/interval", "Mean relative error"], shift_table)}

### 7.6 Intel sensor dataset

The valid Intel input has {intel_config['records']:,} records and {intel_config['strata']} observed `mote_id` strata. Sensor 54 first appears at valid position {int(mote54_first):,}. DV-OASRS detects it online and gives it capacity one at its first allocation event; later allocations are recorded in `results/dv_oasrs/intel/allocation_over_time.csv`. However, across all four budgets it has higher temperature mean error than equal-capacity OASRS and uniform reservoir. At B=270 the errors are {pct(intel_temp[(270, 'DV-OASRS')]['mean_relative_error'])}, {pct(intel_temp[(270, 'OASRS-equal')]['mean_relative_error'])}, and {pct(intel_temp[(270, 'Uniform Reservoir')]['mean_relative_error'])}, respectively. Thus the late-stratum handling works in the trace, while the accuracy result is negative for this workload. The static VA Intel run with {pct(next(r['mean_relative_error'] for r in intel if r['budget']=='270' and r['method']=='VA-OASRS' and r['query']=='temperature_mean'))} error is retained as an invalid historical failure and is excluded.

{md_table(["Budget", "Method", "Temperature mean RE", "Replay records/s", "95% interval coverage"], intel_table)}

Mean relative error for all four Intel numeric queries:

{md_table(["Budget", "Method", "Temperature", "Humidity", "Light", "Voltage"], intel_feature_table)}

Per-stratum DV-OASRS population, capacity, and actual retained count are saved in `results/dv_oasrs/intel/sample_counts.csv` for each seed and budget. First appearance position for each of the 52 observed sensors is in `first_appearance.csv`.

### 7.7 Throughput, overhead, and Pareto analysis

On the variance-skew Python replay at B=200, DV-OASRS processes {throughput_mean['DV-OASRS@500']/1e6:.2f} million records/s, OASRS {throughput_mean['OASRS']/1e6:.2f} million, static VA-OASRS {throughput_mean['Static VA-OASRS@500']/1e6:.2f} million, uniform reservoir {throughput_mean['Uniform Reservoir']/1e6:.2f} million, and exact processing {throughput_mean['Exact']/1e6:.2f} million. Increasing the sample budget also grows priority maintenance and lowers measured Python throughput. These are local single-process replay rates and include the implementation's Python bookkeeping; they are not comparable to Flink throughput. DV-OASRS's estimated candidate-pool byte count is a shallow Python object estimate only, not a process memory profile.

{md_table(["Method", "Mean replay records/s", "Microseconds/input record", "Mean RE"], throughput_table)}

### 7.7.1 Final method comparison

This comparison uses the variance-skew synthetic workload at B=200 (1% of 20,000 records). Memory was not measured consistently across methods; it is therefore not numerically populated.

{md_table(["Method", "Sampling strategy", "Variance aware", "Dynamic allocation", "Replay records/s", "Mean RE", "Memory"], comparison_rows)}

The Pareto figure shows an unfavorable throughput trade-off for DV-OASRS in this prototype. Raw DV results record variance-update time, allocation time, priority-reservoir time, allocation event count, and a shallow candidate-pool memory estimate. These instrumentation timings do not include JVM/Flink execution and can be noisy at Python timer granularity. Dynamic priority state can grow to X times B candidate references, a key cost of preserving correct resizing semantics. CPU utilization, resident memory, checkpoint size, Kafka lag, and end-to-end latency were not measured. No resource conclusion can be drawn from the Python throughput plot alone.

### 7.8 Confidence intervals

The sampler reports a normal-approximation error interval using the existing StreamApprox-style stratified sample-variance estimator. Across the five-seed variance-skew sweep, DV-OASRS coverage ranges from {pct(min(coverage_values))} to {pct(max(coverage_values))}; at B=200 it is {pct(sm('variance_skew', 200, 'DV-OASRS@500')['mean_ci95_coverage'])}. Five repetitions are too few for calibrated coverage claims. The normal interval is an estimate, not a rigorous finite-sample guarantee. Coverage and raw per-seed errors are in the CSVs.

## 8. Ablation study

The available three-way comparison is OASRS (no variance), static VA-OASRS (variance and one-shot allocation), and DV-OASRS (variance plus periodic reallocation), with a uniform reservoir reference. We did not add a separate OASRS-plus-variance-tracking/no-allocation variant, so pure variance-tracking compute cost is not isolated. Static and dynamic comparison does not show a uniform dynamic advantage: static wins on some distributions and budgets, while DV wins on selected high-heterogeneity and skew cases.

## 9. Discussion

Variance-aware dynamic allocation helps in the evaluated synthetic cases when one or more strata have substantially higher within-stratum variance and the online variance estimate has time to respond. It does not help under equal variance (ratio 1), does not consistently beat static VA-OASRS, and does not beat uniform reservoir or OASRS for Intel temperature means. Reallocation and priority retention also reduce throughput substantially in this Python implementation. New-source discovery is the clearest functional improvement over static VA-OASRS; the evidence does not establish that this benefit is worth the cost for production use.

## 10. Limitations

The priority pool is O(XB) in the worst case. A global budget smaller than the number of active strata is rejected. Allocation is query-field dependent; Intel allocations use temperature variance for humidity, light, and voltage too. Evaluation covers mean and sum, not a full histogram suite or dedicated approximate COUNT query. The synthetic shift has one form and does not test a drift detector. There are five seeds, only four sampling budgets, and no statistical significance test. The Intel OASRS capacity map uses an offline observed source registry; DV-OASRS does not. Flink source is uncompiled/unrun, single-parallelism, and only DV-OASRS is implemented there. No end-to-end or resource benchmark is available.

## 11. Threats to validity

Gaussian synthetic sources, chosen means, and variance patterns may not represent operational data. A fixed random-priority method and only five seeds leave substantial Monte Carlo uncertainty. The Intel file's record order and query distributions may not represent live Kafka arrival. The OASRS cost function in the paper is delegated to an implementation-specific hook, so our equal-capacity control is one defensible baseline, not the only OASRS allocation. Python timing is sensitive to interpreter/host and cannot predict Flink throughput. Confidence coverage estimates are based on very few repetitions and use an approximate normal interval.

## 12. Conclusion

The Python implementation demonstrates statistically defensible dynamic capacity changes using independent priorities and successfully assigns samples to late-arriving strata. On synthetic streams, DV-OASRS reduces error versus equal-capacity OASRS at high variance heterogeneity, but it regresses at the homogeneous endpoint and does not consistently improve over static VA-OASRS. On Intel temperature means it is less accurate and slower than the OASRS/uniform controls. Therefore, the research question has a conditional answer: online variance allocation can help under selected heterogeneous synthetic distributions, but this study does not establish a significant overall accuracy/performance improvement or justify its overhead for the Intel workload. A compiled, fault-tested Flink implementation with matched baseline jobs and resource instrumentation is required before drawing a systems conclusion.

## Reproducibility and artifacts

- Raw and processed synthetic measurements: `results/dv_oasrs/synthetic/`
- Intel measurements and first-appearance positions: `results/dv_oasrs/intel/`
- Late-stratum experiment: `results/dv_oasrs/late-strata/`
- Generated figures: `results/dv_oasrs/plots/`
- Python benchmark: `experiments/va_oasrs/benchmark.py`
- Plot generator: `experiments/va_oasrs/dv_plots.py`
- Report generator: `experiments/va_oasrs/build_report.py`
- DV-OASRS implementation: `src/streamapprox/dv_oasrs.py`
- Flink reference job: `flink/src/main/java/org/streamapprox/flink/DVOASRSJob.java`

## References

Do Le Quoc, Ruichuan Chen, Pramod Bhatotia, Christof Fetzer, Volker Hilt, and Thorsten Strufe. “StreamApprox: Approximate Computing for Stream Analytics.” ACM Middleware 2017. DOI: [10.1145/3135974.3135989](https://doi.org/10.1145/3135974.3135989).
'''
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"Generated {REPORT}")


if __name__ == "__main__":
    main()
