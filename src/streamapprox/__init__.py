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
from streamapprox.estimators import (
    EmpiricalError,
    calculate_relative_error,
    compare_with_exact,
    estimate_mean,
    estimate_mean_error,
    estimate_sum,
    estimate_sum_error,
    stratum_sample_variance,
)
from streamapprox.control import AdaptiveController, CostFunction, LinearCostFunction
from streamapprox.metrics import (
    BenchmarkResult,
    BenchmarkTimer,
    get_peak_rss_mb,
    save_benchmark_results,
)
from streamapprox.models import (
    EstimationResult,
    OASRSSnapshot,
    StratumSnapshot,
    WindowResult,
)
from streamapprox.oasrs import OASRS
from streamapprox.va_oasrs import VAOASRS
from streamapprox.dv_oasrs import DVOASRS
from streamapprox.reservoir import Reservoir
from streamapprox.srs import SRSSampler
from streamapprox.strata import Stratum
from streamapprox.synthetic import (
    StratumSpec,
    compute_ground_truth,
    create_paper_gaussian_stream,
    create_paper_poisson_stream,
    create_paper_skewed_gaussian_stream,
    create_paper_skewed_poisson_stream,
    generate_gaussian_stream,
    generate_multi_strata_stream,
    generate_poisson_stream,
    generate_uniform_stream,
)
from streamapprox.windows import CountWindowManager, TimeWindowManager

__version__ = "0.1.0"

__all__ = [
    "OASRS",
    "VAOASRS",
    "DVOASRS",
    "OASRSSnapshot",
    "Reservoir",
    "Stratum",
    "StratumSnapshot",
    "EstimationResult",
    "WindowResult",
    "TimeWindowManager",
    "CountWindowManager",
    "SRSSampler",
    "BenchmarkTimer",
    "BenchmarkResult",
    "get_peak_rss_mb",
    "save_benchmark_results",
    "approximate_sum",
    "approximate_mean",
    "approximate_count",
    "approximate_stratum_sum",
    "approximate_stratum_sums",
    "approximate_stratum_means",
    "estimate_sum",
    "estimate_mean",
    "estimate_sum_error",
    "estimate_mean_error",
    "stratum_sample_variance",
    "calculate_relative_error",
    "EmpiricalError",
    "compare_with_exact",
    "CostFunction",
    "LinearCostFunction",
    "AdaptiveController",
    "StratumSpec",
    "generate_uniform_stream",
    "generate_gaussian_stream",
    "generate_poisson_stream",
    "generate_multi_strata_stream",
    "create_paper_gaussian_stream",
    "create_paper_poisson_stream",
    "create_paper_skewed_gaussian_stream",
    "create_paper_skewed_poisson_stream",
    "compute_ground_truth",
]
