"""Configurable Windowing Abstractions for StreamApprox.

Fulfills Section 2.2, 3.1 (Algorithm 2), and Section 6 of
"StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Provides:
- TimeWindowManager: Configurable time-based sliding and tumbling windows.
- CountWindowManager: Count-based micro-batching windows.
- Independent sampling state per window, isolated lifecycle management,
  and automatic query evaluation (sum, mean, error bounds).
"""

import math
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from streamapprox.aggregators import (
    ValueExtractor,
    approximate_count,
)
from streamapprox.estimators import (
    estimate_mean,
    estimate_sum,
)
from streamapprox.models import WindowResult
from streamapprox.oasrs import OASRS, StratumKeyExtractor

T = TypeVar("T")
TimestampExtractor = Optional[Union[Callable[[T], float], str]]


class TimeWindowManager(Generic[T]):
    """Time-based sliding and tumbling window processor for approximate stream analytics.

    Parameters
    ----------
    window_duration : float
        Duration of each window interval D (e.g. 10.0 seconds). Must be > 0.
    slide_interval : Optional[float], default=None
        Sliding step S (e.g. 5.0 seconds). If None or equal to `window_duration`,
        the manager functions as a tumbling window. Must be > 0 and <= window_duration.
    sample_size : int, default=1000
        Maximum sample size N_i per stratum reservoir for each active window.
    stratum_key : StratumKeyExtractor[T]
        Key or function identifying the stratum S_i of arriving records.
    timestamp_key : Optional[Union[Callable[[T], float], str]], default=None
        Function or key to extract event timestamp from a record.
        If None, records must provide timestamp to `update(record, timestamp=...)`
        or default to virtual arrival clock.
    value_key : ValueExtractor, default=None
        Key or extractor for numerical values used in approximate sum and mean.
    confidence_level : float, default=0.95
        Confidence level for window estimation results (e.g. 0.95, 0.68, 0.997).
    z_score : Optional[float], default=None
        Explicit normal critical value overriding confidence_level.
    seed : Optional[int], default=None
        Master random seed for reproducible sampling across windows.
    base_time : Optional[float], default=None
        Anchor timestamp for window alignment. If None, aligns to the first seen timestamp.
    """

    def __init__(
        self,
        window_duration: float,
        slide_interval: Optional[float] = None,
        sample_size: int = 1000,
        stratum_key: StratumKeyExtractor[T] = lambda record: record["sensor_id"]
        if isinstance(record, dict)
        else getattr(record, "sensor_id"),
        timestamp_key: TimestampExtractor[T] = None,
        value_key: ValueExtractor = None,
        confidence_level: float = 0.95,
        z_score: Optional[float] = None,
        seed: Optional[int] = None,
        base_time: Optional[float] = None,
    ) -> None:
        if window_duration <= 0:
            raise ValueError(
                f"window_duration must be strictly positive, got {window_duration}"
            )

        if slide_interval is None:
            slide_interval = window_duration
        elif slide_interval <= 0:
            raise ValueError(
                f"slide_interval must be strictly positive, got {slide_interval}"
            )
        elif slide_interval > window_duration:
            raise ValueError(
                f"slide_interval ({slide_interval}) cannot exceed window_duration ({window_duration})"
            )

        self._window_duration: float = float(window_duration)
        self._slide_interval: float = float(slide_interval)
        self._sample_size: int = sample_size
        self._stratum_key = stratum_key
        self._timestamp_extractor = self._resolve_timestamp_extractor(timestamp_key)
        self._value_key = value_key
        self._confidence_level = confidence_level
        self._z_score = z_score
        self._seed = seed
        self._base_time: Optional[float] = (
            float(base_time) if base_time is not None else None
        )

        self._virtual_clock: float = 0.0
        self._current_watermark: float = -math.inf
        # Mapping of (start, end) -> active OASRS instance
        self._active_windows: Dict[Tuple[float, float], OASRS[T]] = {}

    @property
    def window_duration(self) -> float:
        """Window duration D."""
        return self._window_duration

    @property
    def slide_interval(self) -> float:
        """Slide step interval S."""
        return self._slide_interval

    @property
    def is_tumbling(self) -> bool:
        """True if windows are non-overlapping tumbling intervals (D == S)."""
        return math.isclose(self._window_duration, self._slide_interval)

    @property
    def active_window_count(self) -> int:
        """Number of windows currently active and accumulating samples."""
        return len(self._active_windows)

    @property
    def active_windows(self) -> List[Tuple[float, float]]:
        """List of active (start, end) window intervals sorted chronologically."""
        return sorted(self._active_windows.keys())

    @staticmethod
    def _resolve_timestamp_extractor(
        key_spec: TimestampExtractor[T],
    ) -> Optional[Callable[[T], float]]:
        """Convert string or callable timestamp extractor to uniform callable."""
        if key_spec is None:
            return None
        if callable(key_spec):
            return key_spec
        if isinstance(key_spec, str):
            attr = key_spec

            def _extract(record: T) -> float:
                if isinstance(record, dict):
                    if attr in record:
                        return float(record[attr])
                    raise KeyError(f"Record dict missing timestamp key '{attr}': {record}")
                if hasattr(record, attr):
                    return float(getattr(record, attr))
                raise AttributeError(f"Record object missing timestamp attribute '{attr}': {record}")

            return _extract
        raise TypeError(
            f"timestamp_key must be callable, str, or None; got {type(key_spec).__name__}"
        )

    def _extract_timestamp(self, record: T, explicit_ts: Optional[float]) -> float:
        """Determine timestamp from explicit parameter, extractor, or virtual clock."""
        if explicit_ts is not None:
            return float(explicit_ts)
        if self._timestamp_extractor is not None:
            return float(self._timestamp_extractor(record))
        if isinstance(record, dict) and "timestamp" in record:
            return float(record["timestamp"])
        if hasattr(record, "timestamp"):
            return float(getattr(record, "timestamp"))

        # Fallback to incremental virtual clock
        ts = self._virtual_clock
        self._virtual_clock += 1.0
        return float(ts)

    def _compute_covering_windows(self, ts: float) -> List[Tuple[float, float]]:
        """Compute all (start, end) intervals covering event timestamp ts."""
        if self._base_time is None:
            # Anchor base time to the first seen timestamp
            self._base_time = ts

        d = self._window_duration
        s = self._slide_interval
        base = self._base_time

        t_rel = ts - base
        if t_rel < 0:
            # Event precedes base time: start from k needed to cover t_rel
            k_max = int(math.floor(t_rel / s))
        else:
            k_max = int(math.floor(t_rel / s))

        # Earliest window index covering ts satisfies base + k*s + d > ts
        # => k*s > t_rel - d => k > (t_rel - d) / s
        k_min = int(math.floor((t_rel - d + 1e-9) / s)) + 1

        intervals: List[Tuple[float, float]] = []
        for k in range(k_min, k_max + 1):
            w_start = base + k * s
            w_end = w_start + d
            if w_start <= ts < w_end:
                intervals.append((w_start, w_end))

        return intervals

    def _close_window(
        self, interval: Tuple[float, float]
    ) -> WindowResult[T]:
        """Finalize a completed window and evaluate approximate queries."""
        sampler = self._active_windows.pop(interval)
        w_start, w_end = interval
        snap = sampler.snapshot()

        sum_res = estimate_sum(
            snap,
            value_key=self._value_key,
            confidence_level=self._confidence_level,
            z_score=self._z_score,
        )
        mean_res = estimate_mean(
            snap,
            value_key=self._value_key,
            confidence_level=self._confidence_level,
            z_score=self._z_score,
        )
        count_res = approximate_count(snap)

        return WindowResult(
            window_start=w_start,
            window_end=w_end,
            snapshot=snap,
            sum_result=sum_res,
            mean_result=mean_res,
            count_result=count_res,
        )

    def _create_window_sampler(self, interval: Tuple[float, float]) -> OASRS[T]:
        """Instantiate a new OASRS sampler dedicated to this window interval."""
        window_seed: Optional[int] = None
        if self._seed is not None:
            # Deterministic seed derivation based on window start time
            w_start, _ = interval
            window_seed = (self._seed + int(abs(w_start) * 1000) % 1000003) % (2**31 - 1)

        return OASRS(
            sample_size=self._sample_size,
            stratum_key=self._stratum_key,
            seed=window_seed,
        )

    def update(
        self, record: T, timestamp: Optional[float] = None
    ) -> List[WindowResult[T]]:
        """Process an incoming record into covering active windows.

        Emits any completed windows whose window_end <= event timestamp.

        Parameters
        ----------
        record : T
            Incoming record.
        timestamp : Optional[float], default=None
            Optional explicit timestamp.

        Returns
        -------
        List[WindowResult[T]]
            List of completed window results triggered by progress of time.
        """
        ts = self._extract_timestamp(record, timestamp)
        emitted: List[WindowResult[T]] = []

        # Advance watermark and retire completed windows
        if ts > self._current_watermark:
            self._current_watermark = ts
            completed_intervals = [
                iv for iv in sorted(self._active_windows.keys()) if iv[1] <= ts
            ]
            for iv in completed_intervals:
                emitted.append(self._close_window(iv))

        # Assign record to all covering windows
        covering = self._compute_covering_windows(ts)
        for iv in covering:
            if iv not in self._active_windows:
                self._active_windows[iv] = self._create_window_sampler(iv)
            self._active_windows[iv].update(record)

        return emitted

    def advance_time(self, current_time: float) -> List[WindowResult[T]]:
        """Advance the window clock to `current_time`, closing completed windows."""
        emitted: List[WindowResult[T]] = []
        completed_intervals = [
            iv for iv in sorted(self._active_windows.keys()) if iv[1] <= current_time
        ]
        for iv in completed_intervals:
            emitted.append(self._close_window(iv))
        return emitted

    def flush(self) -> List[WindowResult[T]]:
        """Close and emit all remaining open windows in chronological order."""
        emitted: List[WindowResult[T]] = []
        remaining_intervals = sorted(self._active_windows.keys())
        for iv in remaining_intervals:
            emitted.append(self._close_window(iv))
        return emitted

    def process_stream(
        self, records: Iterable[T]
    ) -> Iterator[WindowResult[T]]:
        """Stream generator processing an entire sequence of records and flushing windows."""
        for record in records:
            for result in self.update(record):
                yield result
        for result in self.flush():
            yield result


class CountWindowManager(Generic[T]):
    """Count-based tumbling window manager for micro-batch processing.

    Fulfills micro-batch analytics model (Apache Spark Streaming model §2.2, §4.1.1).

    Parameters
    ----------
    window_size : int
        Number of items per count window / micro-batch. Must be >= 1.
    sample_size : int, default=1000
        Sample size per stratum in each count window.
    stratum_key : StratumKeyExtractor[T]
        Stratum extractor function or key.
    value_key : ValueExtractor, default=None
        Value extractor for numeric analytics.
    seed : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        window_size: int,
        sample_size: int = 1000,
        stratum_key: StratumKeyExtractor[T] = lambda record: record["sensor_id"]
        if isinstance(record, dict)
        else getattr(record, "sensor_id"),
        value_key: ValueExtractor = None,
        confidence_level: float = 0.95,
        z_score: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self._window_size: int = window_size
        self._sample_size: int = sample_size
        self._stratum_key = stratum_key
        self._value_key = value_key
        self._confidence_level = confidence_level
        self._z_score = z_score
        self._seed = seed

        self._current_index: int = 0
        self._window_start_index: int = 0
        self._window_index: int = 0
        self._sampler: OASRS[T] = self._create_sampler()

    def _create_sampler(self) -> OASRS[T]:
        window_seed = (
            (self._seed + self._window_index * 10007) % (2**31 - 1)
            if self._seed is not None
            else None
        )
        return OASRS(
            sample_size=self._sample_size,
            stratum_key=self._stratum_key,
            seed=window_seed,
        )

    def update(self, record: T) -> List[WindowResult[T]]:
        """Add record; emits WindowResult whenever count threshold is reached."""
        emitted: List[WindowResult[T]] = []
        self._sampler.update(record)
        self._current_index += 1

        if (self._current_index - self._window_start_index) >= self._window_size:
            w_start = float(self._window_start_index)
            w_end = float(self._current_index)
            snap = self._sampler.snapshot()

            res = WindowResult(
                window_start=w_start,
                window_end=w_end,
                snapshot=snap,
                sum_result=estimate_sum(
                    snap,
                    value_key=self._value_key,
                    confidence_level=self._confidence_level,
                    z_score=self._z_score,
                ),
                mean_result=estimate_mean(
                    snap,
                    value_key=self._value_key,
                    confidence_level=self._confidence_level,
                    z_score=self._z_score,
                ),
                count_result=approximate_count(snap),
            )
            emitted.append(res)

            self._window_start_index = self._current_index
            self._window_index += 1
            self._sampler = self._create_sampler()

        return emitted

    def flush(self) -> List[WindowResult[T]]:
        """Flush remaining buffered items in current micro-batch if any."""
        if self._current_index > self._window_start_index:
            w_start = float(self._window_start_index)
            w_end = float(self._current_index)
            snap = self._sampler.snapshot()

            res = WindowResult(
                window_start=w_start,
                window_end=w_end,
                snapshot=snap,
                sum_result=estimate_sum(
                    snap,
                    value_key=self._value_key,
                    confidence_level=self._confidence_level,
                    z_score=self._z_score,
                ),
                mean_result=estimate_mean(
                    snap,
                    value_key=self._value_key,
                    confidence_level=self._confidence_level,
                    z_score=self._z_score,
                ),
                count_result=approximate_count(snap),
            )
            self._window_start_index = self._current_index
            self._window_index += 1
            self._sampler = self._create_sampler()
            return [res]
        return []

    def process_stream(self, records: Iterable[T]) -> Iterator[WindowResult[T]]:
        """Stream generator over records."""
        for record in records:
            for res in self.update(record):
                yield res
        for res in self.flush():
            yield res
