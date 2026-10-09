"""Simple Random Sampling (SRS) Baseline for Stream Analytics.

Fulfills Phase 8 and Section 11 of project requirements.

Simple Random Sampling draws items from the combined unstratified stream
using a single global reservoir of capacity N.
Unlike OASRS, SRS has no knowledge of strata boundaries or sub-stream sources,
which allows empirical demonstration of:
1. Minority stratum starvation under skewed arrival rates.
2. Accuracy loss degradation on heavy-tailed distributions.
"""

import random
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from streamapprox.aggregators import ValueExtractor, _extract_numeric_value
from streamapprox.reservoir import Reservoir

T = TypeVar("T")


class SRSSampler(Generic[T]):
    """Unstratified Simple Random Sampling baseline maintaining a single global reservoir.

    Parameters
    ----------
    sample_size : int
        Maximum reservoir capacity N. Must be >= 1.
    seed : Optional[int], default=None
        Seed for deterministic sampling.
    """

    def __init__(self, sample_size: int, seed: Optional[int] = None) -> None:
        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size!r}")

        self._sample_size: int = sample_size
        self._reservoir: Reservoir[T] = Reservoir[T](capacity=sample_size, seed=seed)

    @property
    def capacity(self) -> int:
        """Global reservoir capacity N."""
        return self._reservoir.capacity

    @property
    def total_seen(self) -> int:
        """Total records processed C."""
        return self._reservoir.total_seen

    @property
    def sample_size(self) -> int:
        """Current number of items in reservoir Y."""
        return self._reservoir.sample_size

    @property
    def weight(self) -> float:
        """Global sampling weight: W = C / N if C > N else 1.0."""
        c = self._reservoir.total_seen
        n = self._reservoir.capacity
        if c > n:
            return float(c) / float(n)
        return 1.0

    @property
    def samples(self) -> List[T]:
        """Shallow copy of sampled records."""
        return self._reservoir.items

    def update(self, record: T) -> bool:
        """Process incoming stream record online."""
        return self._reservoir.update(record)

    def update_batch(self, records: Iterable[T]) -> int:
        """Process iterable stream online."""
        accepted = 0
        for r in records:
            if self.update(r):
                accepted += 1
        return accepted

    def approximate_sum(self, value_key: ValueExtractor = None) -> float:
        """Approximate sum: W * sum(samples)."""
        if self.sample_size == 0 or self.total_seen == 0:
            return 0.0
        raw_sum = sum(_extract_numeric_value(x, value_key) for x in self.samples)
        return float(raw_sum * self.weight)

    def approximate_mean(self, value_key: ValueExtractor = None) -> float:
        """Approximate mean: approximate_sum / total_seen == unweighted sample mean."""
        if self.sample_size == 0 or self.total_seen == 0:
            return 0.0
        raw_sum = sum(_extract_numeric_value(x, value_key) for x in self.samples)
        return float(raw_sum / self.sample_size)

    def stratum_representation(
        self, stratum_key: Union[Callable[[T], Any], str] = "sensor_id"
    ) -> Dict[Any, int]:
        """Count how many items from each stratum were retained in the global sample."""
        extractor = (
            stratum_key
            if callable(stratum_key)
            else (lambda r: r[stratum_key] if isinstance(r, dict) else getattr(r, stratum_key))
        )
        counts: Dict[Any, int] = {}
        for x in self.samples:
            sid = extractor(x)
            counts[sid] = counts.get(sid, 0) + 1
        return counts

    def reset(self) -> None:
        """Reset reservoir buffer and counter."""
        self._reservoir.reset()

    def __len__(self) -> int:
        return self.sample_size

    def __repr__(self) -> str:
        return (
            f"SRSSampler(capacity={self.capacity}, "
            f"sampled={self.sample_size}, "
            f"total_seen={self.total_seen})"
        )
