"""Dynamic variance-aware OASRS using per-stratum priority reservoirs.

Each record receives an independent U(0, 1) priority. A stratum retains its
largest ``B`` priorities, where B is the global sample budget. At any update,
the active sample is the largest n_i priorities in that bounded candidate set.
For every fixed n_i <= B this is a simple random sample without replacement
from the C_i records seen in that stratum. Reallocation changes only n_i; it
does not resize or invalidate the underlying priority reservoir.

The estimator is the stratified expansion estimator: sum_i C_i * sample_mean_i.
Conditional on the observed finite stratum values and the allocation (which is
computed without using priorities), each stratum sample mean is unbiased.
Memory is O(XB) candidates in the worst case; this is an explicit trade-off.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union

from streamapprox.allocation import allocate_capacities
from streamapprox.models import OASRSSnapshot, StratumSnapshot

T = TypeVar("T")
KeySpec = Union[str, Callable[[T], Any]]


class _PriorityStratum(Generic[T]):
    def __init__(self, key: Any) -> None:
        self.key = key
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.ema_mean = 0.0
        self.ema_variance = 0.0
        self.priority_heap: list[tuple[float, int, T]] = []
        self.capacity = 0
        self.serial = 0

    @property
    def variance(self) -> float:
        return max(0.0, self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def update_statistics(self, value: float, variance_alpha: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

        if self.count == 1:
            self.ema_mean = value
            self.ema_variance = 0.0
        else:
            old_mean = self.ema_mean
            self.ema_mean += variance_alpha * (value - self.ema_mean)
            self.ema_variance = ((1.0 - variance_alpha) *
                                 (self.ema_variance + variance_alpha * (value - old_mean) ** 2))

    def retain_priority(self, priority: float, item: T, max_candidates: int) -> bool:
        self.serial += 1
        entry = (priority, self.serial, item)
        if len(self.priority_heap) < max_candidates:
            heapq.heappush(self.priority_heap, entry)
            return True
        elif priority > self.priority_heap[0][0]:
            heapq.heapreplace(self.priority_heap, entry)
            return True
        return False

    def top_items(self, n: int) -> List[T]:
        n = min(max(0, n), self.count, len(self.priority_heap))
        return [entry[2] for entry in heapq.nlargest(n, self.priority_heap)]

    def snapshot(self) -> StratumSnapshot[T]:
        samples = self.top_items(self.capacity)
        y = len(samples)
        return StratumSnapshot(
            stratum_id=self.key,
            samples=samples,
            total_seen=self.count,
            capacity=self.capacity,
            weight=self.count / y if y else 1.0,
        )


class DVOASRS(Generic[T]):
    """Dynamic Variance-Aware OASRS (DV-OASRS).

    ``reallocation_interval`` is measured in global records. New strata trigger
    an immediate allocation cycle so they receive at least their minimum
    capacity without waiting for an epoch boundary. Welford variance is kept
    for diagnostics; allocation uses an EMA variance to respond to shifts.
    """

    def __init__(
        self,
        sample_size: int,
        stratum_key: KeySpec[T] = "sensor_id",
        value_key: KeySpec[T] = "value",
        seed: Optional[int] = None,
        reallocation_interval: int = 10_000,
        minimum_per_stratum: int = 1,
        variance_alpha: float = 0.01,
        max_fraction_per_stratum: float = 1.0,
        max_change_fraction: float = 0.25,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if reallocation_interval < 1:
            raise ValueError("reallocation_interval must be positive")
        if minimum_per_stratum < 0:
            raise ValueError("minimum_per_stratum cannot be negative")
        if not 0.0 < variance_alpha <= 1.0:
            raise ValueError("variance_alpha must be in (0, 1]")
        if not 0.0 < max_fraction_per_stratum <= 1.0:
            raise ValueError("max_fraction_per_stratum must be in (0, 1]")
        if not 0.0 < max_change_fraction <= 1.0:
            raise ValueError("max_change_fraction must be in (0, 1]")
        self.budget = sample_size
        self.key_fn = self._resolver(stratum_key)
        self.value_fn = self._resolver(value_key)
        self.rng = random.Random(seed)
        self.interval = reallocation_interval
        self.minimum = minimum_per_stratum
        self.alpha = variance_alpha
        self.max_fraction = max_fraction_per_stratum
        self.max_change_fraction = max_change_fraction
        self.strata: Dict[Any, _PriorityStratum[T]] = {}
        self.total_seen = 0
        self.last_allocation_record = 0
        self.allocation_history: List[Dict[str, Any]] = []
        self.allocation_seconds = 0.0
        self.priority_seconds = 0.0
        self.variance_seconds = 0.0
        self.allocation_events = 0
        self.late_stratum_first_sample: Dict[Any, int] = {}

    @staticmethod
    def _resolver(spec: KeySpec) -> Callable[[Any], Any]:
        if callable(spec):
            return spec
        if isinstance(spec, str):
            def extract(item: Any) -> Any:
                if isinstance(item, dict):
                    return item[spec]
                return getattr(item, spec)
            return extract
        raise TypeError("key specification must be a string or callable")

    def update(self, record: T) -> bool:
        """Process one record; return whether it entered the B-sized priority pool."""
        key = self.key_fn(record)
        is_new = key not in self.strata
        if is_new:
            if self.minimum and len(self.strata) + 1 > self.budget // self.minimum:
                raise ValueError("global budget cannot maintain minimum capacity for each active stratum")
            self.strata[key] = _PriorityStratum(key)
        state = self.strata[key]
        value = float(self.value_fn(record))
        self.total_seen += 1
        variance_started = time.perf_counter()
        state.update_statistics(value, self.alpha)
        self.variance_seconds += time.perf_counter() - variance_started
        priority_started = time.perf_counter()
        candidate_retained = state.retain_priority(self.rng.random(), record, self.budget)
        self.priority_seconds += time.perf_counter() - priority_started

        if is_new or self.total_seen - self.last_allocation_record >= self.interval:
            self.reallocate()
        if is_new and state.capacity > 0:
            self.late_stratum_first_sample[key] = self.total_seen
        # True means retained in the bounded per-stratum priority pool, not
        # necessarily in the current query sample of size n_i.
        return candidate_retained

    def update_batch(self, records: Iterable[T]) -> int:
        """Process records and count insertions/replacements in priority pools."""
        retained = 0
        for record in records:
            retained += int(self.update(record))
        return retained

    def reallocate(self) -> Dict[Any, int]:
        if not self.strata:
            return {}
        if self.budget < self.minimum * len(self.strata):
            raise ValueError("global sample budget cannot represent all active strata")
        variance_started = time.perf_counter()
        scores = {
            key: state.count * math.sqrt(max(state.ema_variance, 1e-12))
            for key, state in self.strata.items()
        }
        self.variance_seconds += time.perf_counter() - variance_started
        allocation_started = time.perf_counter()
        max_cap = max(self.minimum, int(self.budget * self.max_fraction))
        population_caps = {key: min(state.count, max_cap) for key, state in self.strata.items()}
        target = allocate_capacities(
            self.budget, scores, minimum=self.minimum, maximums=population_caps
        )
        target_total = sum(target.values())

        # Hysteresis: limit a stratum's move at each periodic interval, except
        # a new stratum gets its minimum immediately. Then project back to the
        # exact feasible budget with deterministic score/remainder ordering.
        step = max(1, int(math.ceil(self.budget * self.max_change_fraction)))
        bounded: Dict[Any, int] = {}
        lower_bounds: Dict[Any, int] = {}
        upper_bounds: Dict[Any, int] = {}
        for key, desired in target.items():
            old = self.strata[key].capacity
            lo = self.minimum if old == 0 else max(self.minimum, old - step)
            hi = min(population_caps[key], old + step) if old else min(population_caps[key], self.minimum)
            lower_bounds[key] = min(lo, population_caps[key])
            upper_bounds[key] = max(lower_bounds[key], hi)
            bounded[key] = min(hi, max(lo, desired))

        while sum(bounded.values()) < target_total:
            candidates = [k for k in target if bounded[k] < upper_bounds[k]]
            if not candidates:
                break
            key = max(candidates, key=lambda k: (target[k] - bounded[k], scores[k], str(k)))
            bounded[key] += 1
        while sum(bounded.values()) > target_total:
            candidates = [k for k in target if bounded[k] > lower_bounds[k]]
            if not candidates:
                break
            key = min(candidates, key=lambda k: (target[k] - bounded[k], scores[k], str(k)))
            bounded[key] -= 1

        for key, capacity in bounded.items():
            self.strata[key].capacity = capacity
        self.allocation_seconds += time.perf_counter() - allocation_started
        self.last_allocation_record = self.total_seen
        self.allocation_events += 1
        self.allocation_history.append({
            "records_seen": self.total_seen,
            "target_capacities": dict(target),
            "capacities": dict(bounded),
            "scores": dict(scores),
            "counts": {k: state.count for k, state in self.strata.items()},
            "welford_variances": {k: state.variance for k, state in self.strata.items()},
            "ema_variances": {k: s.ema_variance for k, s in self.strata.items()},
            "sampled": self.sample_size,
        })
        return bounded

    @property
    def sample_size(self) -> int:
        return sum(min(s.capacity, s.count) for s in self.strata.values())

    @property
    def num_strata(self) -> int:
        return len(self.strata)

    def snapshot(self) -> OASRSSnapshot[T]:
        return OASRSSnapshot(strata={key: state.snapshot() for key, state in self.strata.items()})

    def finalize(self) -> OASRSSnapshot[T]:
        if self.total_seen and self.last_allocation_record != self.total_seen:
            self.reallocate()
        return self.snapshot()
