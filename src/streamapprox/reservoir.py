"""Reservoir Sampling Implementation.

Fulfills Algorithm 1 of "StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Algorithm 1 specifies:
- Reservoir of capacity N.
- Arriving item x_i:
  - If |reservoir| < N: append x_i.
  - If |reservoir| == N (i.e., i > N):
    - Accept with probability p = N / i.
    - If accepted, replace a uniform random item at index j in [0, N - 1].
"""

import random
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class Reservoir(Generic[T]):
    """Fixed-capacity reservoir sampler maintaining a uniform random sample of a stream.

    Parameters
    ----------
    capacity : int
        Maximum number of items N that can be retained in the reservoir. Must be >= 1.
    seed : Optional[int], default=None
        Seed for the pseudo-random number generator to ensure reproducibility.
    rng : Optional[random.Random], default=None
        Custom random instance. If provided, `seed` is ignored.

    Attributes
    ----------
    capacity : int
        The configured sample capacity N.
    total_seen : int
        Total number of items i processed by this reservoir since initialization or last reset.
    sample_size : int
        Current number of items in the reservoir (|reservoir| <= N).
    """

    def __init__(
        self,
        capacity: int,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(
                f"Reservoir capacity must be a positive integer (>= 1), got {capacity!r}"
            )
        self._capacity: int = capacity
        self._total_seen: int = 0
        self._items: List[T] = []

        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

    @property
    def capacity(self) -> int:
        """Maximum reservoir sample size N."""
        return self._capacity

    @property
    def total_seen(self) -> int:
        """Total number of items received from the stream so far (C_i)."""
        return self._total_seen

    @property
    def sample_size(self) -> int:
        """Number of items currently held in the reservoir (Y_i)."""
        return len(self._items)

    @property
    def is_full(self) -> bool:
        """Whether the reservoir has reached its maximum capacity N."""
        return len(self._items) >= self._capacity

    @property
    def items(self) -> List[T]:
        """A shallow copy of the sampled items currently in the reservoir."""
        return list(self._items)

    def update(self, item: T) -> bool:
        """Process an arriving stream item x_i according to Algorithm 1.

        Parameters
        ----------
        item : T
            The incoming record.

        Returns
        -------
        bool
            True if the item was accepted into the reservoir, False if discarded.
        """
        self._total_seen += 1
        i = self._total_seen

        if len(self._items) < self._capacity:
            # Phase 1: Reservoir is not full yet (|reservoir| < N)
            self._items.append(item)
            return True
        else:
            # Phase 2: Reservoir is full (i > N). Accept with probability p = N / i
            p = self._capacity / i
            if self._rng.random() < p:
                j = self._rng.randrange(0, self._capacity)
                self._items[j] = item
                return True
            return False

    def reset(self) -> None:
        """Reset the reservoir, discarding all stored samples and resetting counter."""
        self._items.clear()
        self._total_seen = 0

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"Reservoir(capacity={self._capacity}, "
            f"sampled={len(self._items)}, "
            f"total_seen={self._total_seen})"
        )
