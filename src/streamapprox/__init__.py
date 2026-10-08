"""StreamApprox: Baseline Online Adaptive Stratified Reservoir Sampling (OASRS).

Reproducing the baseline algorithm from:
"StreamApprox: Approximate Computing for Stream Analytics"
Do Le Quoc et al., ACM Middleware 2017.
"""

from streamapprox.aggregators import (
    approximate_count,
    approximate_mean,
    approximate_stratum_means,
    approximate_stratum_sum,
    approximate_stratum_sums,
    approximate_sum,
)
from streamapprox.models import OASRSSnapshot, StratumSnapshot
from streamapprox.oasrs import OASRS
from streamapprox.reservoir import Reservoir
from streamapprox.strata import Stratum

__version__ = "0.1.0"

__all__ = [
    "OASRS",
    "OASRSSnapshot",
    "Reservoir",
    "Stratum",
    "StratumSnapshot",
    "approximate_sum",
    "approximate_mean",
    "approximate_count",
    "approximate_stratum_sum",
    "approximate_stratum_sums",
    "approximate_stratum_means",
]
