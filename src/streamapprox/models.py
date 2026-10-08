"""Core data models and type definitions for StreamApprox."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StratumSnapshot(Generic[T]):
    """Immutable snapshot of a stratum's sampling state at a specific query epoch.

    Corresponds to the state of stratum S_i in StreamApprox (Middleware '17):
    - stratum_id: Identifier of stratum S_i.
    - samples: The Y_i items currently held in the reservoir (Y_i <= N_i).
    - total_seen: Total records C_i received by stratum S_i so far.
    - capacity: Maximum reservoir size N_i for stratum S_i.
    - weight: Sampling weight W_i (Equation 1).
    """

    stratum_id: Any
    samples: List[T] = field(default_factory=list)
    total_seen: int = 0
    capacity: int = 0
    weight: float = 1.0

    @property
    def sample_size(self) -> int:
        """Number of sampled items Y_i currently in the reservoir."""
        return len(self.samples)

    @property
    def is_exact(self) -> bool:
        """True if all arriving items were retained without down-sampling (C_i <= N_i)."""
        return self.total_seen <= self.capacity

    @property
    def sampling_fraction(self) -> float:
        """Empirical sampling fraction Y_i / C_i (1.0 when C_i == 0)."""
        if self.total_seen == 0:
            return 1.0
        return self.sample_size / self.total_seen


@dataclass(frozen=True)
class OASRSSnapshot(Generic[T]):
    """Immutable snapshot of the global OASRS sampling state across all strata.

    Attributes
    ----------
    strata : Dict[Any, StratumSnapshot[T]]
        Mapping of stratum ID to its immutable stratum snapshot.
    """

    strata: Dict[Any, StratumSnapshot[T]] = field(default_factory=dict)

    @property
    def total_seen(self) -> int:
        """Total records sum_i C_i processed across all strata."""
        return sum(s.total_seen for s in self.strata.values())

    @property
    def sample_size(self) -> int:
        """Total records sum_i Y_i currently sampled across all reservoirs."""
        return sum(s.sample_size for s in self.strata.values())

    @property
    def num_strata(self) -> int:
        """Number of distinct strata X."""
        return len(self.strata)

    @property
    def weights(self) -> Dict[Any, float]:
        """Mapping from stratum ID to its sampling weight W_i."""
        return {sid: s.weight for sid, s in self.strata.items()}

    def get_stratum(self, stratum_id: Any) -> Optional[StratumSnapshot[T]]:
        """Retrieve snapshot for a specific stratum."""
        return self.strata.get(stratum_id)

    def __iter__(self) -> Iterator[StratumSnapshot[T]]:
        """Iterate over all stratum snapshots."""
        return iter(self.strata.values())

    def __len__(self) -> int:
        """Return number of strata."""
        return len(self.strata)

    def __getitem__(self, stratum_id: Any) -> StratumSnapshot[T]:
        """Access stratum snapshot by stratum ID."""
        return self.strata[stratum_id]
