"""Variance and Error Estimation Module for StreamApprox.

Fulfills Section 3.3 and Section 5 of "StreamApprox: Approximate Computing for Stream Analytics"
(Do Le Quoc et al., ACM Middleware 2017).

Formulations:
- Equation 7 (Stratum sample variance):
    s_i^2 = (1 / (Y_i - 1)) * sum_{j=1}^{Y_i} (I_{i,j} - mean_i)^2
- Equation 6 (Estimated variance of approximate sum):
    Var_hat(SUM) = sum_{i=1}^X [ C_i * (C_i - Y_i) * (s_i^2 / Y_i) ]
- Equation 9 (Estimated variance of approximate mean):
    omega_i = C_i / sum_j C_j
    Var_hat(MEAN) = sum_{i=1}^X [ omega_i^2 * (s_i^2 / Y_i) * ((C_i - Y_i) / C_i) ]
- Standard Error:
    SE = sqrt(Var_hat)
- Confidence Interval / Error Bound:
    error_bound = z * SE
"""

import math
from typing import Any, Callable, Dict, Optional, Union

from streamapprox.aggregators import (
    ValueExtractor,
    _extract_numeric_value,
    _resolve_snapshot,
    approximate_mean,
    approximate_sum,
)
from streamapprox.models import EstimationResult, OASRSSnapshot, StratumSnapshot

# Standard critical values for common confidence levels
CONFIDENCE_LEVEL_TO_Z: Dict[float, float] = {
    0.68: 1.0,        # 1-sigma (68-95-99.7 rule)
    0.6827: 1.0,
    0.90: 1.645,
    0.95: 1.96,       # Standard 95% confidence
    0.9545: 2.0,      # 2-sigma
    0.99: 2.576,
    0.997: 3.0,       # 3-sigma (68-95-99.7 rule)
    0.9973: 3.0,
}


def _resolve_z_score(
    confidence_level: float = 0.95,
    z_score: Optional[float] = None,
) -> float:
    """Resolve z-score from explicit value or confidence level."""
    if z_score is not None:
        if z_score < 0:
            raise ValueError(f"z_score must be non-negative, got {z_score}")
        return float(z_score)

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )

    # Check exact lookup table
    for level, z in CONFIDENCE_LEVEL_TO_Z.items():
        if abs(confidence_level - level) < 1e-4:
            return z

    # High-precision Abramowitz & Stegun rational approximation for normal quantile
    p = 1.0 - (1.0 - confidence_level) / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return float(t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0))


def stratum_sample_variance(
    stratum: StratumSnapshot,
    value_key: ValueExtractor = None,
) -> float:
    """Calculate unbiased sample variance s_i^2 for stratum S_i (Equation 7).

    s_i^2 = (1 / (Y_i - 1)) * sum_{j=1}^{Y_i} (I_{i,j} - mean_i)^2

    Edge Cases:
    - Y_i <= 1: returns 0.0 (variance cannot be estimated from <= 1 samples).
    - Constant values: returns 0.0.
    """
    y_i = stratum.sample_size
    if y_i <= 1:
        return 0.0

    values = [
        _extract_numeric_value(item, value_key)
        for item in stratum.samples
    ]
    mean_i = sum(values) / y_i
    ss_diff = sum((x - mean_i) ** 2 for x in values)

    var = ss_diff / (y_i - 1)
    return max(0.0, float(var))


def estimate_sum(
    snapshot: Any,
    value_key: ValueExtractor = None,
    confidence_level: float = 0.95,
    z_score: Optional[float] = None,
) -> EstimationResult:
    """Estimate approximate sum and its rigorous variance / error bound.

    Implements Equation 6:
        Var_hat(SUM) = sum_{i=1}^X [ C_i * (C_i - Y_i) * (s_i^2 / Y_i) ]

    Edge cases handled:
    - Empty stream (total_seen == 0): estimate=0, variance=0, SE=0.
    - Full sample (C_i <= N_i): C_i - Y_i = 0, contributing exactly 0 variance.
    - Zero variance in samples: s_i^2 = 0, contributing 0 variance.
    - Y_i <= 1: handled cleanly without division by zero.
    """
    snap = _resolve_snapshot(snapshot)
    z = _resolve_z_score(confidence_level=confidence_level, z_score=z_score)

    approx_s = approximate_sum(snap, value_key=value_key)

    if isinstance(snap, StratumSnapshot):
        strata_iter = [snap]
    else:
        strata_iter = list(snap.strata.values())

    total_var = 0.0
    stratum_vars: Dict[Any, float] = {}

    for s_snap in strata_iter:
        c_i = s_snap.total_seen
        y_i = s_snap.sample_size

        if c_i <= s_snap.capacity or c_i <= y_i or y_i <= 1:
            # Finite population correction: C_i - Y_i == 0, or insufficient samples
            v_i = 0.0
        else:
            s_i_sq = stratum_sample_variance(s_snap, value_key=value_key)
            # Equation 6: C_i * (C_i - Y_i) * (s_i^2 / Y_i)
            v_i = float(c_i * (c_i - y_i) * (s_i_sq / y_i))

        v_i = max(0.0, v_i)
        stratum_vars[s_snap.stratum_id] = v_i
        total_var += v_i

    se = math.sqrt(total_var)
    bound = z * se

    return EstimationResult(
        estimate=approx_s,
        variance=total_var,
        standard_error=se,
        confidence_level=confidence_level,
        z_score=z,
        error_bound=bound,
        stratum_variances=stratum_vars,
    )


def estimate_mean(
    snapshot: Any,
    value_key: ValueExtractor = None,
    confidence_level: float = 0.95,
    z_score: Optional[float] = None,
) -> EstimationResult:
    """Estimate approximate mean and its rigorous variance / error bound.

    Implements Equation 9:
        omega_i = C_i / sum_j C_j
        Var_hat(MEAN) = sum_{i=1}^X [ omega_i^2 * (s_i^2 / Y_i) * ((C_i - Y_i) / C_i) ]

    Edge cases handled:
    - Empty stream (total_seen == 0): estimate=0, variance=0, SE=0.
    - Full sample (C_i <= N_i): C_i - Y_i = 0, contributing exactly 0 variance.
    - Zero variance in samples: s_i^2 = 0, contributing 0 variance.
    - Y_i <= 1: handled cleanly without division by zero.
    """
    snap = _resolve_snapshot(snapshot)
    z = _resolve_z_score(confidence_level=confidence_level, z_score=z_score)

    approx_m = approximate_mean(snap, value_key=value_key)

    if isinstance(snap, StratumSnapshot):
        total_c = snap.total_seen
        strata_iter = [snap]
    else:
        total_c = snap.total_seen
        strata_iter = list(snap.strata.values())

    if total_c == 0:
        return EstimationResult(
            estimate=0.0,
            variance=0.0,
            standard_error=0.0,
            confidence_level=confidence_level,
            z_score=z,
            error_bound=0.0,
            stratum_variances={},
        )

    total_var = 0.0
    stratum_vars: Dict[Any, float] = {}

    for s_snap in strata_iter:
        c_i = s_snap.total_seen
        y_i = s_snap.sample_size

        if c_i <= s_snap.capacity or c_i <= y_i or y_i <= 1 or c_i == 0:
            v_i = 0.0
        else:
            s_i_sq = stratum_sample_variance(s_snap, value_key=value_key)
            omega_i = c_i / total_c
            # Equation 9: omega_i^2 * (s_i^2 / Y_i) * ((C_i - Y_i) / C_i)
            v_i = float((omega_i ** 2) * (s_i_sq / y_i) * ((c_i - y_i) / c_i))

        v_i = max(0.0, v_i)
        stratum_vars[s_snap.stratum_id] = v_i
        total_var += v_i

    se = math.sqrt(total_var)
    bound = z * se

    return EstimationResult(
        estimate=approx_m,
        variance=total_var,
        standard_error=se,
        confidence_level=confidence_level,
        z_score=z,
        error_bound=bound,
        stratum_variances=stratum_vars,
    )


def estimate_sum_error(
    snapshot: Any,
    value_key: ValueExtractor = None,
    confidence_level: float = 0.95,
    z_score: Optional[float] = None,
) -> EstimationResult:
    """Convenience alias matching section 8 API design: estimate_sum_error(snapshot)."""
    return estimate_sum(
        snapshot,
        value_key=value_key,
        confidence_level=confidence_level,
        z_score=z_score,
    )


def estimate_mean_error(
    snapshot: Any,
    value_key: ValueExtractor = None,
    confidence_level: float = 0.95,
    z_score: Optional[float] = None,
) -> EstimationResult:
    """Convenience alias matching section 8 API design: estimate_mean_error(snapshot)."""
    return estimate_mean(
        snapshot,
        value_key=value_key,
        confidence_level=confidence_level,
        z_score=z_score,
    )


def calculate_relative_error(
    approx: float,
    exact: float,
    epsilon: float = 1e-12,
) -> float:
    """Calculate relative accuracy loss: |approx - exact| / |exact|.

    Guards against division by zero when exact == 0.0 without producing NaN or Inf.
    """
    diff = abs(approx - exact)
    abs_exact = abs(exact)

    if abs_exact < epsilon:
        if diff < epsilon:
            return 0.0
        # When exact is zero and approx is non-zero, return bounded ratio using epsilon
        return float(diff / epsilon)

    return float(diff / abs_exact)
