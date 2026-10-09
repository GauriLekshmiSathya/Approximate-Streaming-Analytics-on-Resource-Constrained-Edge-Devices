# Algorithm Documentation: StreamApprox Baseline Reproduction

**Base Paper**: *"StreamApprox: Approximate Computing for Stream Analytics"*, Do Le Quoc, Ruichuan Chen, Pramod Bhatotia, Christof Fetzer, Volker Hilt, Thorsten Strufe. *ACM Middleware 2017*. DOI: [10.1145/3135974.3135989](https://doi.org/10.1145/3135974.3135989).

---

## 1. What Reservoir Sampling Is

Reservoir sampling (*Jeffrey S. Vitter, 1985*) is an online, single-pass randomized algorithm designed to maintain a uniform random sample of size $N$ from an unbounded data stream of unknown total length $C$.

### Mechanics (Algorithm 1)
For an arriving stream of items $x_1, x_2, \dots, x_C$:
1. **Filling Phase ($i \le N$)**: Insert item $x_i$ directly into the reservoir buffer.
2. **Replacement Phase ($i > N$)**:
   - Accept item $x_i$ into the sample with probability:
     $$p = \frac{N}{i}$$
   - If accepted, choose a uniform random index $j \in \{0, 1, \dots, N-1\}$ and replace the existing item at $j$ with $x_i$.
   - If rejected, discard $x_i$.

### Mathematical Invariant
At any step $C$, every item seen so far has an exact, uniform inclusion probability of $\min\left(1, \frac{N}{C}\right)$.

---

## 2. What Stratified Sampling Is

In stream analytics, input streams typically originate from diverse heterogeneous sources (e.g., distinct IoT sensors, network protocols, geographic boroughs, or user segments). These sub-streams exhibit disparate arrival rates and distinct population distributions.

**Stratified sampling** partitions the population into mutually exclusive, exhaustive sub-populations called *strata* ($S_1, S_2, \dots, S_X$). Independent samples are then drawn from each stratum. This guarantees that:
- Every sub-stream is guaranteed representation in the final sample.
- Sample variance within each stratum is isolated from inter-stratum variance.
- Minority strata with rare but high-impact values are not starved by high-volume majority streams.

---

## 3. Why Normal Reservoir Sampling Underrepresents Minority Streams

Under standard Simple Random Sampling (SRS) using a single global reservoir, every item has an equal probability of retention. When streams are heavily skewed, high-volume strata flood the reservoir, while minority strata face statistical starvation.

### Empirical Example (Paper §5.7)
Consider a stream with:
- Stratum $A$ (high-frequency, low-value): $80\%$ of stream ($80,000$ items, value $\approx 10$)
- Stratum $B$ (medium-frequency): $19.99\%$ of stream ($19,990$ items, value $\approx 1,000$)
- Stratum $C$ (long-tail minority, extreme value): $0.01\%$ of stream ($10$ items, value $\approx 100,000$)

With an SRS reservoir of size $N = 5,000$ ($10\%$ sampling fraction):
- Inclusion probability for each item is $\frac{5000}{100000} = 5\%$.
- Expected retained items from Stratum $C$: $10 \times 0.05 = 0.5$ items.
- The probability that **zero** items from Stratum $C$ are retained is $(1 - 0.05)^{10} \approx 60\%$.
- When Stratum $C$ is missed, the true mean of $\approx 8,206$ is estimated as $\approx 196$, causing an catastrophic accuracy loss of **$97.6\%$**!

OASRS solves this by maintaining an independent reservoir for Stratum $C$, preserving all 10 items ($100\%$ representation) and achieving $< 0.001\%$ accuracy loss.

---

## 4. OASRS Algorithm (Online Adaptive Stratified Reservoir Sampling)

OASRS (Algorithm 3) combines the stream-friendly online properties of reservoir sampling with the variance reduction of stratified sampling:

```text
Algorithm OASRS:
Input: arriving record x_i, per-stratum capacity N_i, stratum key function

1. Extract stratum identifier: sid = stratum_key(x_i)
2. If stratum S_{sid} not yet seen:
     Initialize Stratum S_{sid} with capacity N_{sid} and arrival counter C_{sid} = 0
3. Increment arrival counter: C_{sid} <- C_{sid} + 1
4. If C_{sid} <= N_{sid}:
     Insert x_i into reservoir R_{sid}
5. Else (C_{sid} > N_{sid}):
     With probability N_{sid} / C_{sid}:
       Pick random index j in [0, N_{sid} - 1]
       Replace R_{sid}[j] <- x_i
6. At query epoch / window completion:
     Calculate sampling weight W_{sid} according to Equation 1
```

---

## 5. Data Structures

The implementation uses decoupled, modular data abstractions:

1. **`Reservoir[T]`** ([`reservoir.py`](file:///home/MrShah_21/programming/Approximate-Streaming-Analytics-on-Resource-Constrained-Edge-Devices/src/streamapprox/reservoir.py)):
   - Stores sampled items in a pre-allocated Python list of bounded capacity $N$.
   - Tracks arrival counter $C$ and active sample count $Y = |R| \le N$.
   - Employs dedicated `random.Random` instance for determinism and seed control.

2. **`Stratum[T]`** ([`strata.py`](file:///home/MrShah_21/programming/Approximate-Streaming-Analytics-on-Resource-Constrained-Edge-Devices/src/streamapprox/strata.py)):
   - Encapsulates stratum identity $S_i$ and its underlying `Reservoir`.
   - Computes dynamic weight $W_i$.
   - Produces immutable snapshots `StratumSnapshot`.

3. **`OASRS[T]`** ([`oasrs.py`](file:///home/MrShah_21/programming/Approximate-Streaming-Analytics-on-Resource-Constrained-Edge-Devices/src/streamapprox/oasrs.py)):
   - Manages dictionary of active strata: `Dict[Any, Stratum[T]]`.
   - Dispatches incoming records on-the-fly without locking or synchronization.
   - Creates global immutable snapshots `OASRSSnapshot`.

4. **`StratumSnapshot` & `OASRSSnapshot`** ([`models.py`](file:///home/MrShah_21/programming/Approximate-Streaming-Analytics-on-Resource-Constrained-Edge-Devices/src/streamapprox/models.py)):
   - Frozen, decoupled value objects capturing states at query evaluation time.
   - Prevents downstream aggregation or variance computation from mutating online stream state.

---

## 6. Weight Calculation (Equation 1)

For each stratum $S_i$ with arrival count $C_i$ and maximum capacity $N_i$:
$$
W_i = \begin{cases}
\frac{C_i}{N_i} & \text{if } C_i > N_i \\
1.0 & \text{if } C_i \le N_i
\end{cases}
$$

- When $C_i \le N_i$: Every record received is present in the sample ($Y_i = C_i$). Each sample represents only itself ($W_i = 1$).
- When $C_i > N_i$: The stratum is downsampled to $N_i$ items ($Y_i = N_i$). Each sample statistically represents $W_i = \frac{C_i}{N_i}$ original items.

---

## 7. Approximate Sum (Equations 2 & 3)

For stratum $S_i$ with $Y_i$ sampled values $\{I_{i,1}, I_{i,2}, \dots, I_{i,Y_i}\}$ and weight $W_i$:
$$
\widehat{SUM}_i = \left(\sum_{j=1}^{Y_i} I_{i,j}\right) \times W_i
$$

The global approximate sum across all $X$ active strata is:
$$
\widehat{SUM} = \sum_{i=1}^X \widehat{SUM}_i
$$

---

## 8. Approximate Mean (Equations 4 & 8)

The global approximate mean is the total approximate sum divided by total observed arrivals:
$$
\widehat{MEAN} = \frac{\widehat{SUM}}{\sum_{i=1}^X C_i}
$$

Equivalently, defining the arrival proportion of stratum $S_i$ as:
$$
\omega_i = \frac{C_i}{\sum_{k=1}^X C_k}
$$
The mean is the weighted linear combination of stratum sample means:
$$
\widehat{MEAN} = \sum_{i=1}^X (\omega_i \times \overline{I}_i) \quad \text{where } \overline{I}_i = \frac{1}{Y_i} \sum_{j=1}^{Y_i} I_{i,j}
$$

*Note*: Calculating an unweighted average of pooled reservoir samples produces massive bias under skew. Preserving weights $W_i$ and proportions $\omega_i$ guarantees unbiasedness.

---

## 9. Error Estimation & Variance Bounds (Equations 6, 7, 8, 9)

### 9.1 Stratum Sample Variance (Equation 7)
$$
s_i^2 = \frac{1}{Y_i - 1} \sum_{j=1}^{Y_i} (I_{i,j} - \overline{I}_i)^2
$$
- Handled safely when $Y_i \le 1$ ($s_i^2 = 0$).

### 9.2 Estimated Variance of Approximate Sum (Equation 6)
Applying finite-population sampling theory:
$$
\widehat{Var}(\widehat{SUM}) = \sum_{i=1}^X \left[ C_i \times (C_i - Y_i) \times \frac{s_i^2}{Y_i} \right]
$$
- When $C_i \le N_i$, $C_i - Y_i = 0$, so the stratum contributes **zero variance** (exact sample).
- When $C_i > N_i$, variance scales with the finite population correction $(C_i - Y_i)$.

### 9.3 Estimated Variance of Approximate Mean (Equation 9)
$$
\widehat{Var}(\widehat{MEAN}) = \sum_{i=1}^X \left[ \omega_i^2 \times \frac{s_i^2}{Y_i} \times \frac{C_i - Y_i}{C_i} \right] \equiv \frac{\widehat{Var}(\widehat{SUM})}{\left(\sum_{k=1}^X C_k\right)^2}
$$

### 9.4 Standard Error & Confidence Bounds
- **Standard Error**: $SE = \sqrt{\widehat{Var}}$
- **Confidence Interval**: $[\text{estimate} - z \cdot SE, \; \text{estimate} + z \cdot SE]$
- Common critical values:
  - $68\%$ ($1\sigma$): $z = 1.0$
  - $95\%$ ($2\sigma$): $z = 1.96$
  - $99.7\%$ ($3\sigma$): $z = 3.0$

---

## 10. Window Processing (Section 2.2 & 3.1)

Windows segment the continuous stream into bounded processing intervals:

1. **`TimeWindowManager`**:
   - Manages sliding windows ($S < D$) and tumbling windows ($S = D$).
   - Allocates an independent `OASRS` sampling instance per active window interval $[t_{\text{start}}, t_{\text{end}})$.
   - Advances watermarks as events arrive and emits completed `WindowResult` objects upon boundary crossing.
   - Guarantees complete state isolation without inter-window state pollution.

2. **`CountWindowManager`**:
   - Manages count-based micro-batches matching Apache Spark Streaming's micro-batch model (§4.1.1).

---

## 11. Distinction: StreamApprox Paper vs Implementation Choices

| Aspect | StreamApprox Paper (Middleware '17) | Our Baseline Implementation | Rationale |
|---|---|---|---|
| **Platform** | Apache Spark Streaming & Apache Flink (Java/Scala) | Clean standalone Python package (`streamapprox`) | Targeted for edge devices; permits decoupled mathematical validation without heavy JVM/Kafka cluster dependencies |
| **Virtual Cost Function** | Assumed to exist; stated as **not implemented** in paper §4.2.1 | Explicit `sample_size` and `sampling_fraction` parameters | Faithful baseline reproduction without inventing unverified ML controllers |
| **Stratum Keying** | Conceptualized as source/sub-stream tagging | Configurable callable or dictionary string key | Supports arbitrary JSON/dict, tuple, or dataclass records |
| **Quantiles** | 68-95-99.7 empirical integer rule ($z \in \{1, 2, 3\}$) | Supports both integer $\sigma$ and formal Gaussian quantiles ($z=1.96, 2.576$) | Statistical precision for formal academic benchmarks |
| **Edge Protections** | Not detailed in paper | Rigorous guards against $Y_i \le 1$, empty streams, $0$ variance, and $\text{exact} = 0$ | Ensures zero `NaN`, `Inf`, or `ZeroDivisionError` in production edge environments |
| **Sampling State** | Partitioned across Spark RDDs / Flink operators | Independent `OASRS` per window interval | Clean lifecycle without distributed synchronization overhead |
