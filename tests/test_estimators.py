"""Tests for Variance and Error Estimation (Section 3.3, Equations 5, 6, 7, 8, 9)."""

import math
import numpy as np
import pytest

from streamapprox.estimators import (
    calculate_relative_error,
    estimate_mean,
    estimate_mean_error,
    estimate_sum,
    estimate_sum_error,
    stratum_sample_variance,
)
from streamapprox.models import StratumSnapshot
from streamapprox.oasrs import OASRS


class TestSampleVarianceFidelity:
    """Validate stratum sample variance s_i^2 against numpy ddof=1 (Equation 7)."""

    def test_sample_variance_matches_numpy(self):
        vals = [12.5, 18.2, 29.1, 14.7, 22.0, 31.4]
        snap = StratumSnapshot[dict](
            stratum_id="s1",
            samples=[{"v": x} for x in vals],
            total_seen=100,
            capacity=10,
            weight=10.0,
        )

        expected_var = float(np.var(vals, ddof=1))
        computed_var = stratum_sample_variance(snap, value_key="v")

        assert pytest.approx(computed_var, rel=1e-6) == expected_var

    def test_sample_variance_edge_cases(self):
        """Zero or single item must yield variance 0.0 without division by zero."""
        # Empty
        snap_empty = StratumSnapshot[float](stratum_id="e", samples=[], total_seen=0, capacity=10)
        assert stratum_sample_variance(snap_empty) == 0.0

        # Single item
        snap_single = StratumSnapshot[float](stratum_id="s", samples=[42.0], total_seen=1, capacity=10)
        assert stratum_sample_variance(snap_single) == 0.0

        # Identical items
        snap_const = StratumSnapshot[float](stratum_id="c", samples=[5.0, 5.0, 5.0], total_seen=3, capacity=10)
        assert stratum_sample_variance(snap_const) == 0.0


class TestVarianceAndErrorBounds:
    """Validate Equations 6 and 9 variance bounds, SE, and confidence intervals."""

    def test_exact_regime_zero_variance(self):
        """When C_i <= N_i for all strata, variance and standard error MUST be 0.0."""
        sampler = OASRS[dict](sample_size=20, stratum_key="sensor", seed=1)
        for i in range(10):
            sampler.update({"sensor": "s1", "temp": float(i)})
            sampler.update({"sensor": "s2", "temp": float(i * 2)})

        snap = sampler.snapshot()
        res_sum = estimate_sum(snap, value_key="temp")
        res_mean = estimate_mean(snap, value_key="temp")

        assert res_sum.variance == 0.0
        assert res_sum.standard_error == 0.0
        assert res_sum.error_bound == 0.0
        assert res_sum.lower_bound == res_sum.estimate
        assert res_sum.upper_bound == res_sum.estimate

        assert res_mean.variance == 0.0
        assert res_mean.standard_error == 0.0
        assert res_mean.error_bound == 0.0

    def test_empty_stream_graceful_handling(self):
        """Empty stream must produce zero estimate and zero error without NaN or Inf."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sid")
        snap = sampler.snapshot()

        res_sum = estimate_sum(snap)
        res_mean = estimate_mean(snap)

        assert res_sum.estimate == 0.0
        assert res_sum.variance == 0.0
        assert res_sum.standard_error == 0.0
        assert not math.isnan(res_sum.standard_error)
        assert not math.isinf(res_sum.standard_error)

        assert res_mean.estimate == 0.0
        assert res_mean.variance == 0.0
        assert res_mean.standard_error == 0.0

    def test_sum_and_mean_variance_mathematical_consistency(self):
        """Verify the exact relationship: Var_hat(MEAN) = Var_hat(SUM) / (sum C_j)^2."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=42)

        # Feed 100 items to s1, 50 items to s2 with Gaussian noise
        rng = np.random.RandomState(42)
        for x in rng.normal(loc=10.0, scale=2.0, size=100):
            sampler.update({"sensor": "s1", "v": float(x)})
        for x in rng.normal(loc=50.0, scale=5.0, size=50):
            sampler.update({"sensor": "s2", "v": float(x)})

        snap = sampler.snapshot()
        total_c = snap.total_seen  # 150

        res_sum = estimate_sum(snap, value_key="v")
        res_mean = estimate_mean(snap, value_key="v")

        assert res_sum.variance > 0.0
        assert res_mean.variance > 0.0
        assert res_sum.standard_error == math.sqrt(res_sum.variance)
        assert res_mean.standard_error == math.sqrt(res_mean.variance)

        expected_mean_var = res_sum.variance / (total_c ** 2)
        assert pytest.approx(res_mean.variance, rel=1e-6) == expected_mean_var

    def test_confidence_levels_and_z_scores(self):
        """Check z-scores for 68-95-99.7 rule and custom z-scores."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=7)
        for i in range(100):
            sampler.update({"sensor": "s1", "val": float(i)})

        snap = sampler.snapshot()

        res_68 = estimate_mean(snap, value_key="val", confidence_level=0.68)
        res_95 = estimate_mean(snap, value_key="val", confidence_level=0.95)
        res_997 = estimate_mean(snap, value_key="val", confidence_level=0.997)

        assert pytest.approx(res_68.z_score, abs=0.01) == 1.0
        assert pytest.approx(res_95.z_score, abs=0.01) == 1.96
        assert pytest.approx(res_997.z_score, abs=0.01) == 3.0

        assert res_68.error_bound < res_95.error_bound < res_997.error_bound

        # Explicit z-score override
        res_custom_z = estimate_mean(snap, value_key="val", z_score=2.5)
        assert res_custom_z.z_score == 2.5
        assert pytest.approx(res_custom_z.error_bound, rel=1e-6) == 2.5 * res_custom_z.standard_error

    def test_convenience_alias_api(self):
        """Check estimate_sum_error and estimate_mean_error aliases."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=1)
        for i in range(20):
            sampler.update({"sensor": "s1", "v": float(i)})

        res1 = estimate_mean_error(sampler, value_key="v")
        res2 = estimate_mean(sampler, value_key="v")
        assert res1.estimate == res2.estimate
        assert res1.variance == res2.variance

        res3 = estimate_sum_error(sampler, value_key="v")
        res4 = estimate_sum(sampler, value_key="v")
        assert res3.estimate == res4.estimate


class TestRelativeErrorCalculation:
    """Validate robust accuracy loss calculation under edge cases."""

    def test_nonzero_normal_values(self):
        # 10% error
        rel_err = calculate_relative_error(approx=90.0, exact=100.0)
        assert pytest.approx(rel_err, rel=1e-6) == 0.10

    def test_both_exact_and_approx_zero(self):
        """When both are zero, relative error is exactly 0.0."""
        assert calculate_relative_error(approx=0.0, exact=0.0) == 0.0

    def test_exact_zero_approx_nonzero(self):
        """When exact is zero and approx is nonzero, returns bounded non-NaN result."""
        rel_err = calculate_relative_error(approx=0.05, exact=0.0)
        assert not math.isnan(rel_err)
        assert not math.isinf(rel_err)
        assert rel_err > 0.0


class TestEstimatorEdgeCasesAndValidation:
    """Validate argument boundaries and models on single StratumSnapshots."""

    def test_invalid_z_score_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            estimate_sum(StratumSnapshot("s", [], 0, 10), z_score=-1.0)

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            estimate_sum(StratumSnapshot("s", [], 0, 10), confidence_level=1.5)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            estimate_sum(StratumSnapshot("s", [], 0, 10), confidence_level=0.0)

    def test_arbitrary_confidence_level_approximation(self):
        """Test continuous rational quantile approximation (e.g. 80% confidence -> z ~ 1.28)."""
        snap = StratumSnapshot("s", [1.0, 2.0, 3.0, 4.0, 5.0], 20, 5, weight=4.0)
        res = estimate_mean(snap, confidence_level=0.80)
        assert pytest.approx(res.z_score, abs=0.02) == 1.28155

    def test_single_stratum_snapshot_direct_estimation(self):
        """estimate_sum and estimate_mean work directly on an individual StratumSnapshot."""
        snap = StratumSnapshot("s1", [10.0, 20.0, 30.0], 10, 3, weight=10.0 / 3.0)
        res_sum = estimate_sum(snap)
        res_mean = estimate_mean(snap)

        assert res_sum.estimate == (10.0 + 20.0 + 30.0) * (10.0 / 3.0)  # 200.0
        assert res_mean.estimate == 20.0
        assert res_sum.variance > 0.0
        assert res_mean.variance > 0.0
        assert res_sum.confidence_interval == (res_sum.lower_bound, res_sum.upper_bound)
        assert res_sum.relative_error_bound() == res_sum.error_bound / res_sum.estimate

    def test_relative_error_bound_when_estimate_zero(self):
        snap_empty = StratumSnapshot("s", [], 0, 10)
        res = estimate_mean(snap_empty)
        assert res.estimate == 0.0
        assert res.relative_error_bound() == 0.0

