"""Synthetic Data Stream Generators for StreamApprox Benchmarking.

Reproduces the synthetic evaluation experimental setups from Section 5.1, 5.4, and 5.7 of:
"StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Provides:
- Parametric single-distribution stream generators (Uniform, Gaussian, Poisson).
- Multi-strata stream interleaver with configurable arrival rates and skews.
- Exact paper benchmark presets:
    - Gaussian baseline: A ~ N(10, 5), B ~ N(1000, 50), C ~ N(10000, 500) (§5.1)
    - Poisson baseline: A ~ Pois(10), B ~ Pois(1000), C ~ Pois(10^8) (§5.1)
    - Varying arrival rates: 8K:2K:100, 3K:3K:3K, 100:2K:8K (§5.4)
    - Skew distributions: 80% : 19% : 1% and 80% : 19.99% : 0.01% (§5.7)
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class StratumSpec:
    """Specification of a stratum sub-stream distribution and arrival dynamics.

    Attributes
    ----------
    stratum_id : str
        Sub-stream identifier S_i (e.g. 'A', 'B', 'C').
    distribution : str
        'gaussian', 'poisson', 'uniform', or 'constant'.
    params : Dict[str, float]
        Distribution parameters (e.g. {'mean': 10, 'std': 5} or {'lam': 10}).
    weight : float
        Relative arrival rate or proportion weight (e.g. 8000 for A, 2000 for B, 100 for C).
    """

    stratum_id: str
    distribution: str
    params: Dict[str, float]
    weight: float = 1.0


def generate_uniform_stream(
    n_records: int,
    low: float = 0.0,
    high: float = 100.0,
    stratum_id: str = "default",
    start_time: float = 0.0,
    time_step: float = 0.001,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate a stream of uniformly distributed records."""
    rng = np.random.RandomState(seed)
    current_time = start_time
    values = rng.uniform(low=low, high=high, size=n_records)
    for v in values:
        yield {
            "sensor_id": stratum_id,
            "value": float(v),
            "timestamp": float(current_time),
        }
        current_time += time_step


def generate_gaussian_stream(
    n_records: int,
    mean: float = 0.0,
    std: float = 1.0,
    stratum_id: str = "default",
    start_time: float = 0.0,
    time_step: float = 0.001,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate a stream of Gaussian distributed records."""
    rng = np.random.RandomState(seed)
    current_time = start_time
    values = rng.normal(loc=mean, scale=std, size=n_records)
    for v in values:
        yield {
            "sensor_id": stratum_id,
            "value": float(v),
            "timestamp": float(current_time),
        }
        current_time += time_step


def generate_poisson_stream(
    n_records: int,
    lam: float = 10.0,
    stratum_id: str = "default",
    start_time: float = 0.0,
    time_step: float = 0.001,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate a stream of Poisson distributed records."""
    rng = np.random.RandomState(seed)
    current_time = start_time
    values = rng.poisson(lam=lam, size=n_records)
    for v in values:
        yield {
            "sensor_id": stratum_id,
            "value": float(v),
            "timestamp": float(current_time),
        }
        current_time += time_step


def generate_multi_strata_stream(
    strata_specs: List[StratumSpec],
    total_records: int,
    start_time: float = 0.0,
    time_step: float = 0.001,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate an interleaved multi-strata data stream.

    Records arrive randomly according to the relative arrival rate weights of each stratum.
    """
    if not strata_specs:
        return

    rng = np.random.RandomState(seed)
    weights = np.array([s.weight for s in strata_specs], dtype=float)
    if weights.sum() == 0:
        probs = np.ones(len(strata_specs)) / len(strata_specs)
    else:
        probs = weights / weights.sum()

    current_time = start_time

    # Pre-sample stratum selections to maximize generation speed
    stratum_indices = rng.choice(len(strata_specs), size=total_records, p=probs)

    for idx in stratum_indices:
        spec = strata_specs[idx]
        if spec.distribution == "gaussian":
            val = rng.normal(loc=spec.params.get("mean", 0.0), scale=spec.params.get("std", 1.0))
        elif spec.distribution == "poisson":
            val = rng.poisson(lam=spec.params.get("lam", 10.0))
        elif spec.distribution == "uniform":
            val = rng.uniform(low=spec.params.get("low", 0.0), high=spec.params.get("high", 1.0))
        elif spec.distribution == "constant":
            val = spec.params.get("val", 0.0)
        else:
            raise ValueError(f"Unknown distribution: {spec.distribution}")

        yield {
            "sensor_id": spec.stratum_id,
            "value": float(val),
            "timestamp": float(current_time),
        }
        current_time += time_step


# ==============================================================================
# Paper Benchmark Experimental Presets (§5.1, §5.4, §5.7)
# ==============================================================================


def create_paper_gaussian_stream(
    total_records: int = 10000,
    arrival_rates: Optional[Dict[str, float]] = None,
    seed: Optional[int] = 42,
    time_step: float = 0.001,
) -> Iterator[Dict[str, Any]]:
    """Paper Section 5.1 Gaussian benchmark stream.

    Parameters:
      A: mu = 10,    sigma = 5
      B: mu = 1000,  sigma = 50
      C: mu = 10000, sigma = 500
    Default arrival rates: A:B:C = 8000:2000:100 (§5.4)
    """
    rates = arrival_rates or {"A": 8000.0, "B": 2000.0, "C": 100.0}
    specs = [
        StratumSpec("A", "gaussian", {"mean": 10.0, "std": 5.0}, weight=rates.get("A", 1.0)),
        StratumSpec("B", "gaussian", {"mean": 1000.0, "std": 50.0}, weight=rates.get("B", 1.0)),
        StratumSpec("C", "gaussian", {"mean": 10000.0, "std": 500.0}, weight=rates.get("C", 1.0)),
    ]
    return generate_multi_strata_stream(
        strata_specs=specs,
        total_records=total_records,
        time_step=time_step,
        seed=seed,
    )


def create_paper_poisson_stream(
    total_records: int = 10000,
    arrival_rates: Optional[Dict[str, float]] = None,
    seed: Optional[int] = 42,
    time_step: float = 0.001,
) -> Iterator[Dict[str, Any]]:
    """Paper Section 5.1 Poisson benchmark stream.

    Parameters:
      A: lambda = 10
      B: lambda = 1000
      C: lambda = 100,000,000 (10^8)
    Default arrival rates: A:B:C = 8000:2000:100 (§5.4)
    """
    rates = arrival_rates or {"A": 8000.0, "B": 2000.0, "C": 100.0}
    specs = [
        StratumSpec("A", "poisson", {"lam": 10.0}, weight=rates.get("A", 1.0)),
        StratumSpec("B", "poisson", {"lam": 1000.0}, weight=rates.get("B", 1.0)),
        StratumSpec("C", "poisson", {"lam": 100000000.0}, weight=rates.get("C", 1.0)),
    ]
    return generate_multi_strata_stream(
        strata_specs=specs,
        total_records=total_records,
        time_step=time_step,
        seed=seed,
    )


def create_paper_skewed_gaussian_stream(
    total_records: int = 10000,
    seed: Optional[int] = 42,
    time_step: float = 0.001,
) -> Iterator[Dict[str, Any]]:
    """Paper Section 5.7 Skewed Gaussian benchmark stream.

    Distribution:
      A: mu = 100,   sigma = 10   (80% of items)
      B: mu = 1000,  sigma = 100  (19% of items)
      C: mu = 10000, sigma = 1000 (1% of items)
    """
    specs = [
        StratumSpec("A", "gaussian", {"mean": 100.0, "std": 10.0}, weight=0.80),
        StratumSpec("B", "gaussian", {"mean": 1000.0, "std": 100.0}, weight=0.19),
        StratumSpec("C", "gaussian", {"mean": 10000.0, "std": 1000.0}, weight=0.01),
    ]
    return generate_multi_strata_stream(
        strata_specs=specs,
        total_records=total_records,
        time_step=time_step,
        seed=seed,
    )


def create_paper_skewed_poisson_stream(
    total_records: int = 10000,
    seed: Optional[int] = 42,
    time_step: float = 0.001,
) -> Iterator[Dict[str, Any]]:
    """Paper Section 5.7 Skewed Poisson benchmark stream.

    Distribution:
      A: lambda = 10         (80.00% of items)
      B: lambda = 1000       (19.99% of items)
      C: lambda = 100000000  (0.01% of items)
    """
    specs = [
        StratumSpec("A", "poisson", {"lam": 10.0}, weight=0.80),
        StratumSpec("B", "poisson", {"lam": 1000.0}, weight=0.1999),
        StratumSpec("C", "poisson", {"lam": 100000000.0}, weight=0.0001),
    ]
    return generate_multi_strata_stream(
        strata_specs=specs,
        total_records=total_records,
        time_step=time_step,
        seed=seed,
    )


def compute_ground_truth(
    records: List[Dict[str, Any]], value_key: str = "value", stratum_key: str = "sensor_id"
) -> Dict[str, Any]:
    """Compute exact ground truth sum, mean, count, and per-stratum breakdowns."""
    total_count = len(records)
    if total_count == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "stratum_counts": {},
            "stratum_sums": {},
            "stratum_means": {},
        }

    total_sum = sum(float(r[value_key]) for r in records)
    total_mean = total_sum / total_count

    s_counts: Dict[str, int] = {}
    s_sums: Dict[str, float] = {}

    for r in records:
        sid = str(r[stratum_key])
        val = float(r[value_key])
        s_counts[sid] = s_counts.get(sid, 0) + 1
        s_sums[sid] = s_sums.get(sid, 0.0) + val

    s_means = {sid: s_sums[sid] / s_counts[sid] for sid in s_counts}

    return {
        "count": total_count,
        "sum": total_sum,
        "mean": total_mean,
        "stratum_counts": s_counts,
        "stratum_sums": s_sums,
        "stratum_means": s_means,
    }
