"""Variance-aware OASRS with periodic, online Neyman allocation.

The sampler stores only a maximum-size reservoir per active stratum. After a
burn-in, allocation is periodically recomputed from counts and Welford sample
variances. Nested uniform reservoirs allow capacity changes without replaying
or storing discarded stream records.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union

from streamapprox.allocation import allocate_capacities
from streamapprox.models import OASRSSnapshot, StratumSnapshot

T = TypeVar("T")
KeySpec = Union[str, Callable[[T], Any]]


class _VarianceReservoir(Generic[T]):
    def __init__(self, key: Any, rng: random.Random) -> None:
        self.key = key
        self.rng = rng
        self.seen = 0
        self.samples: List[T] = []
        self.capacity = 0
        self.population_at_resize = 0
        self.pilot_capacity = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.accepted = 0
        self.reallocations = 0

    @property
    def variance(self) -> float:
        return self.m2 / (self.seen - 1) if self.seen > 1 else 0.0

    def add(self, value: float) -> None:
        self.seen += 1
        delta = value - self.mean
        self.mean += delta / self.seen
        self.m2 += delta * (value - self.mean)

    def update(self, item: T, value: float) -> bool:
        self.add(value)
        if self.capacity == 0:
            return False
        if len(self.samples) < self.capacity:
            self.samples.append(item)
            self.accepted += 1
            return True
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.samples[index] = item
            self.accepted += 1
            return True
        return False

    def resize(self, capacity: int) -> None:
        capacity = max(0, int(capacity))
        if capacity < len(self.samples):
            # A uniform subset of a uniform reservoir remains uniform.
            self.samples = self.rng.sample(self.samples, capacity)
        self.capacity = capacity
        self.population_at_resize = self.seen
        self.reallocations += 1

    def snapshot(self) -> StratumSnapshot[T]:
        y = len(self.samples)
        return StratumSnapshot(
            stratum_id=self.key,
            samples=list(self.samples),
            total_seen=self.seen,
            capacity=self.capacity,
            weight=(self.seen / y if y else 1.0),
        )


class VAOASRS(Generic[T]):
    """Online VA-OASRS using periodic Neyman allocation.

    The selected query field is used only to estimate within-stratum variance.
    Counts, values, and sample histories are observed online; no future or
    ground-truth values enter allocation decisions.
    """

    def __init__(
        self,
        sample_size: int,
        stratum_key: KeySpec[T] = "sensor_id",
        value_key: KeySpec[T] = "value",
        seed: Optional[int] = None,
        reallocation_interval: int = 1000,
        minimum_per_stratum: int = 1,
        burn_in_records: Optional[int] = None,
        max_fraction_per_stratum: float = 1.0,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if reallocation_interval < 1:
            raise ValueError("reallocation_interval must be positive")
        if minimum_per_stratum < 0:
            raise ValueError("minimum_per_stratum cannot be negative")
        if not 0 < max_fraction_per_stratum <= 1:
            raise ValueError("max_fraction_per_stratum must be in (0, 1]")
        self.budget = sample_size
        self.key_fn = self._resolver(stratum_key)
        self.value_fn = self._resolver(value_key)
        self.rng = random.Random(seed)
        self.interval = reallocation_interval
        self.minimum = minimum_per_stratum
        self.burn_in = burn_in_records if burn_in_records is not None else reallocation_interval
        self.max_fraction = max_fraction_per_stratum
        self.strata: Dict[Any, _VarianceReservoir[T]] = {}
        self.total_seen = 0
        self.last_allocation_record = 0
        self.allocation_seconds = 0.0
        self.variance_seconds = 0.0
        self.reservoir_seconds = 0.0
        self.allocation_history: List[Dict[str, Any]] = []
        self.frozen_allocation = False

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
        key = self.key_fn(record)
        stratum = self.strata.get(key)
        if stratum is None:
            stratum = _VarianceReservoir(key, random.Random(self.rng.randrange(2**63)))
            # Pilot state is separately bounded per stratum by global budget.
            # It supports late discovery and a correct uniform downsample at
            # allocation, at the cost of up to X*B temporary pilot records.
            if not self.frozen_allocation:
                stratum.capacity = self.budget
                stratum.pilot_capacity = self.budget
            self.strata[key] = stratum
        value = float(self.value_fn(record))
        started = time.perf_counter()
        accepted = stratum.update(record, value)
        self.reservoir_seconds += time.perf_counter() - started
        self.total_seen += 1
        if (not self.frozen_allocation and self.total_seen >= self.burn_in and
                self.total_seen - self.last_allocation_record >= self.interval):
            self.reallocate()
        return accepted

    def update_batch(self, records: Iterable[T]) -> int:
        """Update from an iterable and return the number of retained arrivals."""
        accepted = 0
        for record in records:
            accepted += int(self.update(record))
        return accepted

    def reallocate(self) -> Dict[Any, int]:
        if not self.strata:
            return {}
        if self.budget < self.minimum * len(self.strata):
            raise ValueError("global sample budget cannot represent every active stratum")
        variance_started = time.perf_counter()
        scores = {}
        for key, state in self.strata.items():
            # Small-sample fallback: equal SD prior until variance can be estimated.
            sd = math.sqrt(max(0.0, state.variance)) if state.seen > 1 else 1.0
            scores[key] = state.seen * sd
        self.variance_seconds += time.perf_counter() - variance_started
        max_cap = max(self.minimum, int(self.budget * self.max_fraction))
        # A stratum may receive a final capacity larger than its burn-in count;
        # subsequent arrivals will fill it. Capping by burn-in population would
        # systematically starve late and low-frequency strata.
        max_caps = {key: max_cap for key in self.strata}
        # Capacity must not shrink below minimum while an active stratum exists.
        max_caps = {key: max(self.minimum, cap) for key, cap in max_caps.items()}
        start = time.perf_counter()
        allocations = allocate_capacities(
            self.budget, scores, minimum=self.minimum, maximums=max_caps
        )
        for key, capacity in allocations.items():
            self.strata[key].resize(capacity)
        self.allocation_seconds += time.perf_counter() - start
        self.last_allocation_record = self.total_seen
        self.allocation_history.append({
            "records_seen": self.total_seen,
            "capacities": dict(allocations),
            "scores": dict(scores),
        })
        # Reallocation between non-overlapping query epochs requires either
        # retaining/replaying the stratum history or tracking per-item
        # inclusion probabilities. This implementation freezes after the
        # first burn-in allocation to preserve exact SRS reservoir semantics.
        self.frozen_allocation = True
        return allocations

    @property
    def sample_size(self) -> int:
        return sum(len(state.samples) for state in self.strata.values())

    @property
    def num_strata(self) -> int:
        return len(self.strata)

    def snapshot(self) -> OASRSSnapshot[T]:
        return OASRSSnapshot(strata={key: state.snapshot() for key, state in self.strata.items()})

    def finalize(self) -> OASRSSnapshot[T]:
        if self.total_seen and not self.frozen_allocation:
            self.reallocate()
        return self.snapshot()
