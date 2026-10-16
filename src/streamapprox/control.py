"""Configurable budget-to-sample mapping and simple adaptive feedback.

The paper assumes a virtual cost function but leaves its concrete form outside
the implementation. These small interfaces are engineering extensions; they
do not claim to reproduce a controller equation from the paper.
"""

from dataclasses import dataclass
from typing import Protocol


class CostFunction(Protocol):
    """Map a non-negative query budget to a per-stratum sample capacity."""

    def sample_size(self, budget: float, *, strata: int = 1) -> int: ...


@dataclass(frozen=True)
class LinearCostFunction:
    """Simple configurable mapping used when no calibrated cost model exists."""

    items_per_budget_unit: float = 1.0
    minimum: int = 1
    maximum: int = 100_000

    def __post_init__(self) -> None:
        if self.items_per_budget_unit <= 0 or self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("invalid linear cost-function bounds")

    def sample_size(self, budget: float, *, strata: int = 1) -> int:
        if budget < 0 or strata < 1:
            raise ValueError("budget must be non-negative and strata must be positive")
        return max(self.minimum, min(self.maximum, int(budget * self.items_per_budget_unit / strata)))


@dataclass
class AdaptiveController:
    """Adjust next interval capacity to target an estimated relative error.

    This proportional update is a documented implementation choice. Actual
    OASRS objects are interval-scoped, so callers apply the returned capacity
    when constructing the next window's sampler.
    """

    target_relative_error: float = 0.05
    minimum_sample_size: int = 1
    maximum_sample_size: int = 100_000
    adjustment: float = 0.5

    def __post_init__(self) -> None:
        if self.target_relative_error <= 0 or self.minimum_sample_size < 1:
            raise ValueError("target error and minimum sample size must be positive")
        if self.maximum_sample_size < self.minimum_sample_size:
            raise ValueError("maximum_sample_size must be >= minimum_sample_size")
        if not 0 < self.adjustment <= 1:
            raise ValueError("adjustment must be in (0, 1]")

    def next_sample_size(self, current: int, estimate: float, standard_error: float) -> int:
        if current < 1 or standard_error < 0:
            raise ValueError("current size must be positive and standard error non-negative")
        if abs(estimate) == 0:
            ratio = float("inf") if standard_error else 0.0
        else:
            ratio = standard_error / abs(estimate)
        if ratio <= self.target_relative_error:
            proposed = max(self.minimum_sample_size, int(current * (1 - self.adjustment)))
        else:
            scale = min(2.0, max(1.1, ratio / self.target_relative_error))
            proposed = int(current * scale)
        return min(self.maximum_sample_size, max(self.minimum_sample_size, proposed))
