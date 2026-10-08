"""Approximate Linear Query Aggregators for StreamApprox.

Implements approximate linear queries from Section 3.2 of
"StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Formulations:
- Equation 2: SUM_i = (sum_{j=1}^{Y_i} I_{i,j}) * W_i
- Equation 3: SUM = sum_{i=1}^X SUM_i
- Equation 4: MEAN = SUM / sum_{i=1}^X C_i
- Filtered Count: COUNT(pred) = sum_{i=1}^X W_i * sum_{j=1}^{Y_i} 1_{pred}(I_{i,j})

Crucially, the aggregators preserve sampling weights W_i rather than
computing unweighted sample statistics.
"""

from typing import Any, Callable, Dict, Optional, Union

from streamapprox.models import OASRSSnapshot, StratumSnapshot

ValueExtractor = Optional[Union[Callable[[Any], float], str]]
Predicate = Optional[Callable[[Any], bool]]


def _extract_numeric_value(item: Any, value_key: ValueExtractor) -> float:
    """Extract float value from an item based on key, callable, or direct numeric type."""
    if value_key is not None:
        if callable(value_key):
            val = value_key(item)
        elif isinstance(value_key, str):
            if isinstance(item, dict):
                if value_key in item:
                    val = item[value_key]
                else:
                    raise KeyError(f"Item missing numeric field '{value_key}': {item}")
            elif hasattr(item, value_key):
                val = getattr(item, value_key)
            else:
                raise AttributeError(f"Item object missing numeric field '{value_key}': {item}")
        else:
            raise TypeError(f"value_key must be callable, str, or None; got {type(value_key).__name__}")
    else:
        if isinstance(item, (int, float)):
            val = item
        elif isinstance(item, dict):
            if "value" in item:
                val = item["value"]
            elif "val" in item:
                val = item["val"]
            else:
                raise ValueError(
                    f"Item is a dict without default 'value'/'val' key: {item}. "
                    f"Please provide an explicit value_key."
                )
        else:
            raise TypeError(
                f"Cannot infer numeric value from item of type {type(item).__name__}: {item}. "
                f"Please provide an explicit value_key."
            )

    try:
        return float(val)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert extracted value {val!r} to float: {e}") from e


def _resolve_snapshot(
    target: Any,
) -> Union[OASRSSnapshot, StratumSnapshot]:
    """Resolve an OASRS coordinator or snapshot into a valid snapshot."""
    if hasattr(target, "snapshot") and callable(target.snapshot):
        return target.snapshot()
    if isinstance(target, (OASRSSnapshot, StratumSnapshot)):
        return target
    raise TypeError(
        f"Expected OASRS sampler, OASRSSnapshot, or StratumSnapshot; got {type(target).__name__}"
    )


def approximate_stratum_sum(
    stratum: StratumSnapshot,
    value_key: ValueExtractor = None,
) -> float:
    """Calculate approximate sum for a single stratum S_i (Equation 2).

    SUM_i = (sum_{j=1}^{Y_i} I_{i,j}) * W_i
    """
    if stratum.sample_size == 0 or stratum.total_seen == 0:
        return 0.0

    raw_sample_sum = sum(
        _extract_numeric_value(item, value_key) for item in stratum.samples
    )
    return float(raw_sample_sum * stratum.weight)


def approximate_sum(
    snapshot: Any,
    value_key: ValueExtractor = None,
) -> float:
    """Calculate global approximate sum across all strata (Equation 3).

    SUM = sum_{i=1}^X SUM_i
    """
    snap = _resolve_snapshot(snapshot)

    if isinstance(snap, StratumSnapshot):
        return approximate_stratum_sum(snap, value_key=value_key)

    if snap.total_seen == 0 or snap.num_strata == 0:
        return 0.0

    return sum(
        approximate_stratum_sum(s_snap, value_key=value_key)
        for s_snap in snap.strata.values()
    )


def approximate_stratum_mean(
    stratum: StratumSnapshot,
    value_key: ValueExtractor = None,
) -> float:
    """Calculate approximate mean for a single stratum S_i.

    MEAN_i = (1 / Y_i) * sum_{j=1}^{Y_i} I_{i,j}
    """
    if stratum.sample_size == 0 or stratum.total_seen == 0:
        return 0.0

    raw_sample_sum = sum(
        _extract_numeric_value(item, value_key) for item in stratum.samples
    )
    return float(raw_sample_sum / stratum.sample_size)


def approximate_mean(
    snapshot: Any,
    value_key: ValueExtractor = None,
) -> float:
    """Calculate global approximate mean across all strata (Equation 4).

    MEAN = SUM / sum_{i=1}^X C_i
    """
    snap = _resolve_snapshot(snapshot)

    if isinstance(snap, StratumSnapshot):
        return approximate_stratum_mean(snap, value_key=value_key)

    total_c = snap.total_seen
    if total_c == 0:
        return 0.0

    total_sum = approximate_sum(snap, value_key=value_key)
    return float(total_sum / total_c)


def approximate_count(
    snapshot: Any,
    predicate: Predicate = None,
) -> float:
    """Calculate approximate count of records, optionally filtered by predicate.

    Without predicate:
        Returns exact total_seen (sum_i C_i).
    With predicate:
        Estimates matching records using stratum weights:
        COUNT_hat(pred) = sum_{i=1}^X W_i * sum_{j=1}^{Y_i} 1_{pred}(I_{i,j})
    """
    snap = _resolve_snapshot(snapshot)

    if predicate is None:
        return float(snap.total_seen)

    if isinstance(snap, StratumSnapshot):
        if snap.sample_size == 0 or snap.total_seen == 0:
            return 0.0
        matching = sum(1 for item in snap.samples if predicate(item))
        return float(matching * snap.weight)

    if snap.total_seen == 0 or snap.num_strata == 0:
        return 0.0

    total_est = 0.0
    for s_snap in snap.strata.values():
        if s_snap.sample_size > 0:
            matching = sum(1 for item in s_snap.samples if predicate(item))
            total_est += matching * s_snap.weight

    return float(total_est)


def approximate_stratum_sums(
    snapshot: Any,
    value_key: ValueExtractor = None,
) -> Dict[Any, float]:
    """Breakdown of approximate sum by stratum ID."""
    snap = _resolve_snapshot(snapshot)
    if isinstance(snap, StratumSnapshot):
        return {snap.stratum_id: approximate_stratum_sum(snap, value_key=value_key)}
    return {
        sid: approximate_stratum_sum(s_snap, value_key=value_key)
        for sid, s_snap in snap.strata.items()
    }


def approximate_stratum_means(
    snapshot: Any,
    value_key: ValueExtractor = None,
) -> Dict[Any, float]:
    """Breakdown of approximate mean by stratum ID."""
    snap = _resolve_snapshot(snapshot)
    if isinstance(snap, StratumSnapshot):
        return {snap.stratum_id: approximate_stratum_mean(snap, value_key=value_key)}
    return {
        sid: approximate_stratum_mean(s_snap, value_key=value_key)
        for sid, s_snap in snap.strata.items()
    }
