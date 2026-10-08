"""Stratum abstraction for Online Adaptive Stratified Reservoir Sampling.

Corresponds to stratum S_i tracking in Section 3.2 of
"StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

For each stratum S_i, OASRS maintains:
- Reservoir of maximum sample size N_i
- Counter C_i: number of records received by the stratum
- Current sample size Y_i: number of records currently sampled (Y_i <= N_i)
- Weight W_i:
    W_i = C_i / N_i  if C_i > N_i
    W_i = 1          if C_i <= N_i
"""

import random
from typing import Any, Generic, List, Optional, TypeVar

from streamapprox.models import StratumSnapshot
from streamapprox.reservoir import Reservoir

T = TypeVar("T")


class Stratum(Generic[T]):
    """State and reservoir management for an individual stratum S_i.

    Parameters
    ----------
    stratum_id : Any
        Unique identifier of this stratum (e.g. sensor ID, region, cluster).
    capacity : int
        Maximum sample size N_i for this stratum's reservoir. Must be >= 1.
    seed : Optional[int], default=None
        Seed for the underlying reservoir's pseudo-random number generator.
    rng : Optional[random.Random], default=None
        Custom random instance passed to the reservoir.

    Attributes
    ----------
    stratum_id : Any
        Identifier of the stratum S_i.
    capacity : int
        Maximum sample size N_i.
    total_seen : int
        Arrival counter C_i measuring records received by this stratum.
    sample_size : int
        Current number of sampled items Y_i held in the reservoir.
    weight : float
        Sampling weight W_i according to Equation 1.
    """

    def __init__(
        self,
        stratum_id: Any,
        capacity: int,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._stratum_id: Any = stratum_id
        self._reservoir: Reservoir[T] = Reservoir(
            capacity=capacity, seed=seed, rng=rng
        )

    @property
    def stratum_id(self) -> Any:
        """Identifier of stratum S_i."""
        return self._stratum_id

    @property
    def capacity(self) -> int:
        """Maximum reservoir sample size N_i."""
        return self._reservoir.capacity

    @property
    def total_seen(self) -> int:
        """Arrival counter C_i for this stratum."""
        return self._reservoir.total_seen

    @property
    def sample_size(self) -> int:
        """Current number of sampled items Y_i."""
        return self._reservoir.sample_size

    @property
    def weight(self) -> float:
        """Sampling weight W_i according to Equation 1:

        W_i = C_i / N_i  if C_i > N_i
        W_i = 1.0        if C_i <= N_i
        """
        c_i = self._reservoir.total_seen
        n_i = self._reservoir.capacity
        if c_i > n_i:
            return float(c_i) / float(n_i)
        return 1.0

    @property
    def samples(self) -> List[T]:
        """A copy of the current sample items Y_i in the reservoir."""
        return self._reservoir.items

    def update(self, item: T) -> bool:
        """Process an incoming record belonging to this stratum.

        Increments C_i and executes reservoir sampling step (Algorithm 1).

        Parameters
        ----------
        item : T
            Incoming data item.

        Returns
        -------
        bool
            True if the item was accepted into the reservoir, False otherwise.
        """
        return self._reservoir.update(item)

    def snapshot(self) -> StratumSnapshot[T]:
        """Produce an immutable snapshot of this stratum's sampling state."""
        return StratumSnapshot(
            stratum_id=self._stratum_id,
            samples=self._reservoir.items,
            total_seen=self._reservoir.total_seen,
            capacity=self._reservoir.capacity,
            weight=self.weight,
        )

    def reset(self) -> None:
        """Reset the stratum state (clears reservoir and resets counter C_i to 0)."""
        self._reservoir.reset()

    def __len__(self) -> int:
        return len(self._reservoir)

    def __repr__(self) -> str:
        return (
            f"Stratum(id={self._stratum_id!r}, "
            f"N_i={self.capacity}, "
            f"C_i={self.total_seen}, "
            f"Y_i={self.sample_size}, "
            f"W_i={self.weight:.4f})"
        )
