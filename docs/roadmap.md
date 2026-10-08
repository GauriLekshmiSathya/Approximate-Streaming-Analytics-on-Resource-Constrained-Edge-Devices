# StreamApprox Baseline Implementation Roadmap

## 1. Project Context & Reference Paper

- **Base Paper**: *"StreamApprox: Approximate Computing for Stream Analytics"*, Do Le Quoc, Ruichuan Chen, Pramod Bhatotia, Christof Fetzer, Volker Hilt, Thorsten Strufe. ACM Middleware 2017. DOI: `10.1145/3135974.3135989`.
- **Target Context**: Baseline approximate stream analytics layer for B.Tech project on resource-constrained edge devices.
- **Scope & Constraints**:
  - Pure algorithmic reproduction of the baseline paper without unverified extensions.
  - Strict exclusion of proposed ML controllers, reinforcement learning, or dynamic resource-aware controllers during this baseline phase.
  - Decoupled from heavy infrastructure (Kafka/Flink) to allow pure mathematical and algorithmic validation in Python.

---

## 2. Mathematical Foundation & Paper Formulations

### 2.1 Algorithm 1: Reservoir Sampling (Vitter Algorithm R)
For a stratum sample size $N_i$ and arriving item sequence $x_1, x_2, \dots, x_{C_i}$:
- If current reservoir count $|R_i| < N_i$: append $x_{C_i}$ to $R_i$.
- If $|R_i| = N_i$ (i.e., $C_i > N_i$):
  - Draw random coin flip with probability $p = \frac{N_i}{C_i}$.
  - If heads, replace a uniformly chosen index $j \in [0, N_i - 1]$ with $x_{C_i}$.
  - If tails, discard $x_{C_i}$.
- **Invariant**: Every item seen so far has an exact, uniform inclusion probability of $\min\left(1, \frac{N_i}{C_i}\right)$.

### 2.2 Stratum Weights (Equation 1)
For each stratum $S_i$ with arrival count $C_i$ and reservoir capacity $N_i$:
$$
W_i = \begin{cases}
\frac{C_i}{N_i} & \text{if } C_i > N_i \\
1 & \text{if } C_i \le N_i
\end{cases}
$$
$W_i$ reflects the expansion factor: each sample represents $W_i$ actual items from that stratum.

### 2.3 Approximate Linear Queries (Equations 2, 3, 4)
For $X$ strata $\{S_1, \dots, S_X\}$, with $Y_i = |R_i| \le N_i$ sampled values $\{I_{i,1}, \dots, I_{i,Y_i}\}$:
- **Stratum Approximate Sum**:
  $$\widehat{SUM}_i = \left(\sum_{j=1}^{Y_i} I_{i,j}\right) \times W_i$$
- **Global Approximate Sum**:
  $$\widehat{SUM} = \sum_{i=1}^X \widehat{SUM}_i$$
- **Stratum Weight Ratio**:
  $$\omega_i = \frac{C_i}{\sum_{k=1}^X C_k}$$
- **Global Approximate Mean**:
  $$\widehat{MEAN} = \frac{\widehat{SUM}}{\sum_{i=1}^X C_i} = \sum_{i=1}^X (\omega_i \times \overline{I}_i)$$
  where $\overline{I}_i = \frac{1}{Y_i} \sum_{j=1}^{Y_i} I_{i,j}$ when $Y_i > 0$.

### 2.4 Error & Variance Estimation (Equations 5, 6, 7, 8, 9)
- **Sample Variance within Stratum** (Equation 7):
  $$s_i^2 = \frac{1}{Y_i - 1} \sum_{j=1}^{Y_i} (I_{i,j} - \overline{I}_i)^2$$
- **Estimated Variance of Approximate Sum** (Equation 6):
  $$\widehat{Var}(\widehat{SUM}) = \sum_{i=1}^X \left[ C_i \times (C_i - Y_i) \times \frac{s_i^2}{Y_i} \right]$$
  *Note*: When $C_i \le N_i$, $C_i - Y_i = 0$, reflecting exact sampling with zero variance contribution.
- **Estimated Variance of Approximate Mean** (Equation 9):
  $$\widehat{Var}(\widehat{MEAN}) = \sum_{i=1}^X \left[ \omega_i^2 \times \frac{s_i^2}{Y_i} \times \frac{C_i - Y_i}{C_i} \right]$$
- **Standard Error (SE)**:
  $$SE = \sqrt{\widehat{Var}}$$
- **Confidence Intervals / Error Bound** (68-95-99.7 rule, Section 3.3):
  $$\text{Bound} = z \times SE \quad (z \in \{1.0, 1.96, 2.0, 2.576, 3.0\})$$

---

## 3. Implementation Phases Overview

| Phase | Title | Core Objective | Key Deliverables | Status |
|---|---|---|---|---|
| **Phase 1** | **Reservoir & Stratum** | Single-reservoir sampling (Alg. 1) & Stratum tracking ($C_i, N_i, Y_i, W_i$) | `models.py`, `reservoir.py`, `strata.py`, `test_reservoir.py`, `test_strata.py` | **COMPLETED** |
| **Phase 2** | **OASRS Multi-Strata** | Online Adaptive Stratified Reservoir Sampling orchestrator (Alg. 3) | `oasrs.py`, `test_oasrs.py`, `test_stratification.py` | **COMPLETED** |
| **Phase 3** | **Weighted Aggregations** | Linear queries ($\widehat{SUM}, \widehat{MEAN}, \widehat{COUNT}$) using stratum weights | `aggregators.py`, `test_aggregators.py`, `test_weights.py` | **COMPLETED** |
| **Phase 4** | **Variance & Error Estimation** | Finite-population stratified variance & confidence intervals | `estimators.py`, `test_estimators.py` | Queued |
| **Phase 5** | **Windowing Abstraction** | Sliding and tumbling window processing models | `windows.py`, `test_windows.py` | Queued |
| **Phase 6** | **Comprehensive Testing** | End-to-end invariant validation, edge cases, skew scenarios | Extended tests in `tests/` | Queued |
| **Phase 7** | **Synthetic Data Generators** | Gaussian, Poisson, Uniform, and skewed streams from paper §5.1 & §5.7 | `examples/synthetic_stream.py` | Queued |
| **Phase 8** | **SRS vs OASRS Comparison** | Simple Random Sampling baseline to quantify stratification advantage | `benchmarks/benchmark_accuracy.py` | Queued |
| **Phase 9** | **Resource & Profiling Instrumentation** | Latency, throughput, CPU, RSS memory benchmarks & CSV/JSON export | `benchmarks/benchmark_sampling.py`, `benchmark_resources.py` | Queued |

---

## 4. Phase-by-Phase Detailed Plan

### Phase 1: Reservoir and Stratum (Current Phase)
- Implement `Reservoir` with capacity $N$, item insertion, uniform random replacement, deterministic seed option, and invariants.
- Implement `Stratum` with stratum identity, counter $C_i$, sample size $Y_i$, reservoir of capacity $N_i$, and dynamic weight $W_i$.
- Data models for snapshots (`StratumSnapshot`).
- Unit tests verifying capacity invariants, replacement probabilities, weight calculations, and edge cases.

### Phase 2: OASRS (Online Adaptive Stratified Reservoir Sampling)
- Multi-stratum manager accepting arbitrary stream items.
- Configurable stratum key extractor function (e.g., `lambda r: r['sensor_id']`).
- On-the-fly stratum discovery without prior knowledge of stream keys.
- Capacity allocation strategy across strata (uniform baseline allocation or budget-based).
- Unit tests with balanced and imbalanced stream arrival patterns.

### Phase 3: Weighted Aggregations
- Query evaluators operating on sampler snapshots:
  - `approximate_sum(snapshot, value_key=...)`
  - `approximate_mean(snapshot, value_key=...)`
  - `approximate_count(snapshot)`
- Support stratum-level breakdown and global roll-ups.
- Tests comparing exact vs approximate aggregations on controlled streams.

### Phase 4: Variance and Error Estimation
- Sample variance $s_i^2$ calculation per stratum.
- Stratified variance for sums (Equation 6) and means (Equation 9).
- Standard error and configurable confidence bounds ($z$-scores).
- Edge-case defenses: $Y_i \in \{0, 1\}$, $C_i \le N_i$, zero sample variance, zero exact result.
- Tests ensuring non-negative variance, $SE = \sqrt{Var}$, and mathematical rigor.

### Phase 5: Windowing Abstractions
- Sliding and tumbling window managers.
- Time-based and count-based triggers.
- Window state lifecycle: ingest $\to$ snapshot $\to$ slide/reset.
- Tests for multi-window streams and state isolation across windows.

### Phase 6: Comprehensive Testing Suite
- Extreme skew streams (e.g. 100,000 : 10,000 : 100).
- Invariant assertions: reservoir size $\le N_i$ at all steps.
- Numerical stability tests with large and small numbers.

### Phase 7: Synthetic Data Stream Generators
- Generators reproducing paper evaluation setups:
  - Gaussian streams: $(\mu=10, \sigma=5), (\mu=1000, \sigma=50), (\mu=10000, \sigma=500)$.
  - Poisson streams: $\lambda=10, 1000, 10^8$.
  - Skew distributions: $80\% : 19.99\% : 0.01\%$.
- Reproducible random seeds.

### Phase 8: Simple Random Sampling (SRS) Baseline
- Implementation of standard reservoir sampling over the combined unstratified stream.
- Comparison metrics: Accuracy loss $\frac{|\hat{y} - y|}{|y|}$ across varying sampling fractions ($10\%, 20\%, 40\%, 60\%, 80\%, 100\%$).
- Demonstrate minority stratum starvation in SRS vs representation in OASRS.

### Phase 9: Resource Measurements & Profiling
- Real-time instrumentation:
  - Throughput (items/sec)
  - Processing latency (microseconds per record)
  - Peak RSS memory (`resource.getrusage`)
  - CPU process time
- Machine-readable output formats (CSV, JSON).
- Final algorithm documentation (`docs/algorithm.md`).

---

## 5. Paper Alignment & Ambiguity Notes

1. **Virtual Cost Function (§2.3, §4.2.1)**:
   The paper assumes a virtual cost function converts query budgets into sample sizes, but explicitly states in Section 4.2.1 that it was not implemented in their evaluation. For the baseline, we support explicit `sample_size` and `sampling_fraction`.
2. **Stratum Sample Size Allocation**:
   In Algorithm 3, `N <- getSampleSize(sampleSize, S)` is called. When strata are discovered on the fly in unbounded streams, a fixed reservoir capacity per stratum $N_i = N$ or budget partition $N / |S|$ can be used. We document this clearly.
3. **Pipelined vs Micro-batching**:
   OASRS operates record-at-a-time (pipelined) while supporting batch/window cuts, ensuring full compatibility with both Spark micro-batch and Flink pipelined models.
