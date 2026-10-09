"""Core data models and type definitions for StreamApprox."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

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


@dataclass(frozen=True)
class EstimationResult:
    """Structured result of an approximate query with rigorous statistical error estimation.

    Fulfills Section 3.3 and Section 5 of StreamApprox:
    - estimate: point estimate (approximate sum or mean)
    - variance: estimated sampling variance Var_hat (Equations 6 or 9)
    - standard_error: standard error SE = sqrt(variance)
    - confidence_level: target confidence level (e.g. 0.95)
    - z_score: standard normal critical value z
    - error_bound: margin of error = z * SE
    - stratum_variances: variance contribution per stratum S_i
    """

    estimate: float
    variance: float
    standard_error: float
    confidence_level: float = 0.95
    z_score: float = 1.96
    error_bound: float = 0.0
    stratum_variances: Dict[Any, float] = field(default_factory=dict)

    @property
    def lower_bound(self) -> float:
        """Lower confidence bound (estimate - error_bound)."""
        return self.estimate - self.error_bound

    @property
    def upper_bound(self) -> float:
        """Upper confidence bound (estimate + error_bound)."""
        return self.estimate + self.error_bound

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Confidence interval (lower_bound, upper_bound)."""
        return (self.lower_bound, self.upper_bound)

    def relative_error_bound(self) -> float:
        """Relative error bound = error_bound / |estimate| (handles estimate == 0 safely)."""
        if abs(self.estimate) == 0.0:
            return 0.0
        return float(self.error_bound / abs(self.estimate))


@dataclass(frozen=True)
class WindowResult(Generic[T]):
    """Structured approximate analytics result for a completed window epoch.

    Attributes
    ----------
    window_start : float
        Start timestamp/index of the window interval (inclusive).
    window_end : float
        End timestamp/index of the window interval (exclusive).
    snapshot : OASRSSnapshot[T]
        The immutable OASRS sample snapshot captured at window completion.
    sum_result : EstimationResult
        Approximate sum and rigorous variance / confidence bound.
    mean_result : EstimationResult
        Approximate mean and rigorous variance / confidence bound.
    count_result : float
        Approximate record count in the window.
    """

    window_start: float
    window_end: float
    snapshot: OASRSSnapshot[T]
    sum_result: EstimationResult
    mean_result: EstimationResult
    count_result: float

    @property
    def total_seen(self) -> int:
        """Total records received across all strata in this window."""
        return self.snapshot.total_seen

    @property
    def sample_size(self) -> int:
        """Total records sampled in this window."""
        return self.snapshot.sample_size

    @property
    def num_strata(self) -> int:
        """Number of active strata in this window."""
        return self.snapshot.num_strata

