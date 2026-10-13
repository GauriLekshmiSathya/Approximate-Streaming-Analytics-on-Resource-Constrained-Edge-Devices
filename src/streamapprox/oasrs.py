"""Online Adaptive Stratified Reservoir Sampling (OASRS).

Fulfills Algorithm 3 of "StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Core principles:
- Online execution: records are processed one at a time on-the-fly.
- No storage of the entire stream or window.
- Configurable stratum key extraction (e.g. record["sensor_id"]).
- Independent reservoir per stratum S_i with capacity N_i.
- Dynamic stratum discovery without requiring pre-known strata or stream statistics.
- Exact tracking of arrival counter C_i, sample size Y_i, and sampling weight W_i.
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

from streamapprox.models import OASRSSnapshot
from streamapprox.strata import Stratum

T = TypeVar("T")
StratumKeyExtractor = Union[Callable[[T], Any], str]


class OASRS(Generic[T]):
    """Online Adaptive Stratified Reservoir Sampling (OASRS) coordinator.

    Parameters
    ----------
    sample_size : int
        Default maximum sample size N_i for each stratum's reservoir. Must be >= 1.
    stratum_key : Union[Callable[[T], Any], str]
        Callable extracting stratum identifier from a record, or string key for dict/object access.
    seed : Optional[int], default=None
        Master seed for reproducible pseudo-random sampling.
    stratum_capacity : Optional[Union[int, Callable[[Any], int]]], default=None
        Optional custom per-stratum capacity rule. If an integer, overrides `sample_size`.
        If a callable, `stratum_capacity(stratum_id)` returns N_i for that stratum.
    """

    def __init__(
        self,
        sample_size: int,
        stratum_key: StratumKeyExtractor[T] = lambda record: record["sensor_id"]
        if isinstance(record, dict)
        else getattr(record, "sensor_id"),
        seed: Optional[int] = None,
        stratum_capacity: Optional[Union[int, Callable[[Any], int]]] = None,
    ) -> None:
        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError(
                f"sample_size must be a positive integer (>= 1), got {sample_size!r}"
            )

        self._default_sample_size: int = sample_size
        self._key_extractor: Callable[[T], Any] = self._resolve_key_extractor(stratum_key)
        self._seed: Optional[int] = seed
        self._stratum_capacity_rule = stratum_capacity

        self._master_rng: Optional[random.Random] = (
            random.Random(seed) if seed is not None else None
        )
        self._strata: Dict[Any, Stratum[T]] = {}

    @staticmethod
    def _resolve_key_extractor(key_spec: StratumKeyExtractor[T]) -> Callable[[T], Any]:
        """Convert string or callable key extractor to a uniform callable."""
        if callable(key_spec):
            return key_spec
        elif isinstance(key_spec, str):
            attr_name = key_spec

            def _extract(record: T) -> Any:
                if isinstance(record, dict):
                    if attr_name in record:
                        return record[attr_name]
                    raise KeyError(f"Record dict missing stratum key '{attr_name}': {record}")
                if hasattr(record, attr_name):
                    return getattr(record, attr_name)
                raise AttributeError(f"Record object missing stratum attribute '{attr_name}': {record}")

            return _extract
        else:
            raise TypeError(
                f"stratum_key must be callable or str, got {type(key_spec).__name__}"
            )

    def _determine_capacity(self, stratum_id: Any) -> int:
        """Determine reservoir capacity N_i for a stratum."""
        if self._stratum_capacity_rule is None:
            return self._default_sample_size
        elif isinstance(self._stratum_capacity_rule, int):
            return self._stratum_capacity_rule
        elif callable(self._stratum_capacity_rule):
            cap = self._stratum_capacity_rule(stratum_id)
            if not isinstance(cap, int) or cap < 1:
                raise ValueError(
                    f"stratum_capacity callable returned invalid capacity {cap!r} for stratum {stratum_id!r}"
                )
            return cap
        else:
            return self._default_sample_size

    def _get_or_create_stratum(self, stratum_id: Any) -> Stratum[T]:
        """Retrieve existing stratum or lazily initialize a new one upon first arrival."""
        if stratum_id not in self._strata:
            cap = self._determine_capacity(stratum_id)
            # Derive deterministic stratum seed if master seed was configured
            stratum_seed: Optional[int] = None
            if self._master_rng is not None:
                stratum_seed = self._master_rng.randint(0, 2**31 - 1)

            self._strata[stratum_id] = Stratum(
                stratum_id=stratum_id,
                capacity=cap,
                seed=stratum_seed,
            )
        return self._strata[stratum_id]

    def update(self, record: T) -> bool:
        """Process an incoming record online.

        Extracts stratum key, assigns record to stratum S_i, increments C_i,
        and executes reservoir sampling without storing stream history.

        Parameters
        ----------
        record : T
            Incoming data record.

        Returns
        -------
        bool
            True if record was accepted into the stratum's reservoir, False if rejected.
        """
        stratum_id = self._key_extractor(record)
        stratum = self._get_or_create_stratum(stratum_id)
        return stratum.update(record)

    def update_batch(self, records: Iterable[T]) -> int:
        """Process an iterable stream of records online in arrival order.

        Parameters
        ----------
        records : Iterable[T]
            Sequence or stream generator of records.

        Returns
        -------
        int
            Total number of records accepted into reservoirs.
        """
        accepted = 0
        for record in records:
            if self.update(record):
                accepted += 1
        return accepted

    @property
    def total_seen(self) -> int:
        """Total records sum_i C_i processed across all strata."""
        return sum(s.total_seen for s in self._strata.values())

    @property
    def sample_size(self) -> int:
        """Total records sum_i Y_i currently sampled across all reservoirs."""
        return sum(s.sample_size for s in self._strata.values())

    @property
    def num_strata(self) -> int:
        """Number of active strata X."""
        return len(self._strata)

    @property
    def strata_ids(self) -> List[Any]:
        """List of discovered stratum identifiers."""
        return list(self._strata.keys())

    @property
    def weights(self) -> Dict[Any, float]:
        """Current sampling weights W_i for all active strata."""
        return {sid: s.weight for sid, s in self._strata.items()}

    def get_stratum(self, stratum_id: Any) -> Optional[Stratum[T]]:
        """Retrieve the stratum instance for a specific stratum ID."""
        return self._strata.get(stratum_id)

    def snapshot(self) -> OASRSSnapshot[T]:
        """Produce an immutable snapshot of all strata at current epoch.

        Used for querying approximate aggregations and variance estimation.
        """
        strata_snaps = {
            sid: s.snapshot() for sid, s in self._strata.items()
        }
        return OASRSSnapshot(strata=strata_snaps)

    def reset(self, clear_strata: bool = False) -> None:
        """Reset the sampler state.

        Parameters
        ----------
        clear_strata : bool, default=False
            If True, removes all stratum instances (re-discovers on arrival).
            If False, resets reservoir buffers and counters C_i while keeping stratum registry.
        """
        if clear_strata:
            self._strata.clear()
            if self._seed is not None:
                self._master_rng = random.Random(self._seed)
        else:
            for stratum in self._strata.values():
                stratum.reset()

    def __len__(self) -> int:
        """Total sample size across all reservoirs."""
        return self.sample_size

    def __repr__(self) -> str:
        return (
            f"OASRS(default_sample_size={self._default_sample_size}, "
            f"strata_count={self.num_strata}, "
            f"total_seen={self.total_seen}, "
            f"total_sampled={self.sample_size})"
        )
