# StreamApprox: Dynamic Variance-Aware OASRS

This repository contains a Python implementation and evaluation of **Dynamic Variance-Aware Online Adaptive Stratified Reservoir Sampling (DV-OASRS)**, along with OASRS and static VA-OASRS comparisons. It also contains a reference Apache Flink job that has not yet been built or run.

The research question is whether periodically allocating a fixed sample budget using online stratum population and variance can reduce query error enough to justify its processing cost. The answer in the current experiments is **conditional**: DV-OASRS helps on some high-variance synthetic streams, but not uniformly, and it is slower in the measured Python replay. See the [research report](report/report.md) for the full methodology, tables, plots, negative results, and limitations.

## Methods

| Method | Description |
| --- | --- |
| Exact | Processes every record; supplies offline ground truth. |
| Uniform Reservoir | One global reservoir with a fixed budget. |
| OASRS | Per-stratum Algorithm-R reservoirs. The benchmark uses equal per-stratum capacities as an explicit fixed-budget control; this is an evaluation choice because StreamApprox delegates capacity mapping to its cost function. |
| Static VA-OASRS | The preserved one-shot variance-aware prototype. It allocates once after burn-in and then freezes. |
| DV-OASRS | The proposed dynamic method. It periodically reallocates using online population counts and smoothed variance. |

DV-OASRS assigns each record an independent random priority and retains the top *B* priorities per stratum, where *B* is the global budget. A new capacity selects a prefix of that priority reservoir, avoiding naive resizing of Algorithm-R state. The estimator uses each stratum's observed count and sample mean. New strata trigger immediate reallocation; periodic allocations use the configured record interval. The default Python candidate state can grow to **O(number of active strata × B)**, which is a meaningful memory cost.

The sampler and supporting library are in [`src/streamapprox/`](src/streamapprox/). The existing static implementation remains in [`va_oasrs.py`](src/streamapprox/va_oasrs.py); the dynamic implementation is [`dv_oasrs.py`](src/streamapprox/dv_oasrs.py).

## Results at a glance

At a synthetic variance ratio of 100 and budget 200, DV-OASRS mean relative error was 1.19%, compared with 2.90% for OASRS. At variance ratio 1, DV-OASRS was worse: 0.29% versus 0.17%. On the Intel temperature mean at budget 270, DV-OASRS error was 8.34%, versus 7.73% for OASRS and 3.19% for uniform reservoir. The dynamic method handled the late-arriving Intel sensor, but did not improve accuracy on that workload.

These are five-run descriptive results; the study does not claim statistical significance. Python replay throughput is not Flink throughput. The earlier static VA-OASRS Intel result is retained under `results/va_oasrs/intel2/` as an **invalid comparison** because it missed a late sensor; it is excluded from the current accuracy claims.

## Setup

Use Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the sampler-focused tests:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_dv_oasrs.py tests/test_va_oasrs.py
```

The repository-wide test suite currently also has four failing `TimeWindowManager` tests unrelated to DV-OASRS; see the report or test output for details.

## Results reproduction guide

Run these commands from the repository root. They reproduce the synthetic comparisons, five-seed late-stratum study, Intel replay, figures, and Markdown report. The Intel step reads `data/data.txt` by default; pass `--file` to use another copy of the Intel sensor dataset.

```bash
.venv/bin/python experiments/va_oasrs/benchmark.py \
  --output results/dv_oasrs/synthetic \
  --records 20000 --runs 5 \
  --budgets 200 1000 2000 4000 \
  --intervals 500 2000 \
  --variance-ratios 1 2 5 10 50 100 \
  --seeds 11 23 37 53 71

.venv/bin/python experiments/va_oasrs/late_strata.py \
  --records 400000 --budget 300 \
  --seeds 7 11 23 37 53 --interval 10000 \
  --output results/dv_oasrs/late-strata

.venv/bin/python experiments/va_oasrs/run_intel.py \
  --file data/data.txt --runs 5 \
  --budgets 270 540 1080 2160 \
  --dynamic-interval 10000 \
  --output results/dv_oasrs/intel

.venv/bin/python experiments/va_oasrs/dv_plots.py
.venv/bin/python experiments/va_oasrs/build_report.py
```

The synthetic benchmark includes uniform, population-skew, variance-skew, combined-skew, and distribution-shift streams. It saves raw per-run rows, allocation histories, summaries, paired comparisons, and configuration files. The late-stratum run records first appearances and allocations for S5/S6. The Intel run uses `mote_id` as the natural stratum and writes per-stratum sample counts and first appearance positions.

Re-running a command overwrites the corresponding generated files in its output directory. Keep a copy of those files if you need to compare different configurations. The plots and report are generated from the saved CSVs; do not edit reported measurements by hand.

### Output locations

| Artifact | Path |
| --- | --- |
| Synthetic raw results, allocations, config, and summaries | `results/dv_oasrs/synthetic/` |
| Intel raw results, per-sensor samples, and allocations | `results/dv_oasrs/intel/` |
| Late-stratum raw runs and allocation traces | `results/dv_oasrs/late-strata/` |
| Generated SVG figures | `results/dv_oasrs/plots/` |
| Research-paper-style report | `report/report.md` |

The report generator uses the current result files. Run the benchmark commands before regenerating the report if you changed experiment settings.

## Python API example

```python
from streamapprox.aggregators import approximate_mean
from streamapprox.dv_oasrs import DVOASRS

records = [
    {"sensor_id": "sensor-a", "value": 18.2},
    {"sensor_id": "sensor-b", "value": 21.4},
]
sampler = DVOASRS(
    sample_size=200,
    stratum_key="sensor_id",
    value_key="value",
    seed=11,
    reallocation_interval=10_000,
)
sampler.update_batch(records)
estimate = approximate_mean(sampler.finalize(), "value")
print(estimate)
```

Allocation is driven only by online records. Exact results are computed separately in the experiment code for evaluation. When a stream has more active strata than its budget can represent under the minimum-per-stratum rule, DV-OASRS raises an error rather than silently dropping a stratum.

## Apache Flink status

The reference job is in [`flink/`](flink/README.md), with Maven configuration in `flink/pom.xml`. It uses a Kafka source, one globally keyed process operator, managed Flink state, periodic checkpoints, and a Kafka result sink. Global allocation currently requires parallelism one. Checkpointing is configured as exactly-once for Flink state, but the Kafka sink is not transactional; end-to-end exactly-once output is not claimed.

The job has **not been compiled or run** in the current environment because Maven and the Flink CLI are unavailable. Flink versions of Exact, Uniform Reservoir, OASRS, and static VA-OASRS are not implemented, and there are no Flink throughput, latency, CPU, memory, checkpoint-size, or Kafka-lag measurements. Do not treat the Python benchmark as Flink performance evidence.

Once Maven and a matching Flink/Kafka deployment are available, the reference jar can be packaged with:

```bash
mvn -f flink/pom.xml package
```

Job configuration and limitations are documented in [`flink/README.md`](flink/README.md).

## Project map

```text
src/streamapprox/       OASRS, static VA-OASRS, DV-OASRS, estimators, queries
experiments/va_oasrs/   Synthetic, Intel, late-stratum, plotting, report scripts
flink/                  Unverified Java DV-OASRS reference job
results/dv_oasrs/       Raw and processed experiment outputs
results/dv_oasrs/plots/ Generated figures
report/report.md        Research report generated from saved results
tests/                  Unit and regression tests
```

## Citation

Baseline work:

> Do Le Quoc, Ruichuan Chen, Pramod Bhatotia, Christof Fetzer, Volker Hilt, and Thorsten Strufe. “StreamApprox: Approximate Computing for Stream Analytics.” ACM Middleware 2017. [https://doi.org/10.1145/3135974.3135989](https://doi.org/10.1145/3135974.3135989)
