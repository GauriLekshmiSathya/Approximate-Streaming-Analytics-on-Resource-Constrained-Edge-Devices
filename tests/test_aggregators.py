"""Tests for approximate linear queries (Section 3.2, Equations 2, 3, 4)."""

import pytest

from streamapprox.aggregators import (
    approximate_count,
    approximate_mean,
    approximate_stratum_means,
    approximate_stratum_sum,
    approximate_stratum_sums,
    approximate_sum,
)
from streamapprox.models import StratumSnapshot
from streamapprox.oasrs import OASRS


class TestWeightedAggregationsExactComparison:
    """Compare exact vs approximate metrics under complete and sampled regimes."""

    def test_empty_stream_edge_case(self):
        """Empty stream must produce 0.0 without ZeroDivisionError or NaN."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sid")
        snap = sampler.snapshot()

        assert approximate_sum(snap) == 0.0
        assert approximate_mean(snap) == 0.0
        assert approximate_count(snap) == 0.0
        assert approximate_count(snap, predicate=lambda r: True) == 0.0
        assert approximate_stratum_sums(snap) == {}
        assert approximate_stratum_means(snap) == {}

    def test_exact_regime_when_all_strata_below_capacity(self):
        """When C_i <= N_i for all strata, approximate sum/mean MUST equal exact sum/mean."""
        sampler = OASRS[dict](sample_size=50, stratum_key="sensor", seed=42)

        stream = [
            {"sensor": "s1", "val": 10.0},
            {"sensor": "s1", "val": 20.0},
            {"sensor": "s2", "val": 30.0},
            {"sensor": "s2", "val": 40.0},
            {"sensor": "s2", "val": 50.0},
        ]
        sampler.update_batch(stream)
        snap = sampler.snapshot()

        exact_sum = sum(r["val"] for r in stream)  # 150.0
        exact_mean = exact_sum / len(stream)       # 30.0

        approx_sum = approximate_sum(snap, value_key="val")
        approx_mean = approximate_mean(snap, value_key="val")
        approx_count = approximate_count(snap)

        assert approx_sum == exact_sum
        assert approx_mean == exact_mean
        assert approx_count == 5.0

    def test_known_synthetic_stream_large_sample_fidelity(self):
        """Under large sampling budget, approximate mean converges closely to exact mean."""
        sampler = OASRS[dict](sample_size=1000, stratum_key="source", seed=123)

        # Stratum A: 10,000 items with value 50.0
        # Stratum B: 5,000 items with value 100.0
        count_a = 10000
        count_b = 5000
        val_a = 50.0
        val_b = 100.0

        for _ in range(count_a):
            sampler.update({"source": "A", "val": val_a})
        for _ in range(count_b):
            sampler.update({"source": "B", "val": val_b})

        snap = sampler.snapshot()

        exact_sum = count_a * val_a + count_b * val_b  # 500,000 + 500,000 = 1,000,000
        exact_mean = exact_sum / (count_a + count_b)   # 1,000,000 / 15,000 = 66.6667

        approx_s = approximate_sum(snap, value_key="val")
        approx_m = approximate_mean(snap, value_key="val")

        # Because values in each stratum are constant here, approximate sum and mean are EXACT
        assert pytest.approx(approx_s, rel=1e-6) == exact_sum
        assert pytest.approx(approx_m, rel=1e-6) == exact_mean


class TestWeightPreservationAgainstUnweightedBias:
    """CRITICAL: Demonstrate that weighted mean preserves correctness while unweighted fails.

    Scenario:
    Stratum Heavy: 10,000 items, all value 10.0
    Stratum Light: 100 items, all value 1,000.0
    Total items = 10,100
    Exact Total Sum = 10,000 * 10 + 100 * 1,000 = 100,000 + 100,000 = 200,000.0
    Exact Mean = 200,000 / 10,100 = 19.80198...

    Both reservoirs have capacity N = 100:
    Heavy reservoir: 100 items of value 10.0 -> W_heavy = 10000 / 100 = 100.0
    Light reservoir: 100 items of value 1000.0 -> W_light = 100 / 100 = 1.0

    If someone mistakenly took an unweighted average of the combined reservoirs:
      (100 * 10 + 100 * 1000) / 200 = 101,000 / 200 = 505.0  (2450% error!)
    Our weighted approximate_mean:
      SUM_heavy = (100 * 10) * 100 = 100,000
      SUM_light = (100 * 1000) * 1 = 100,000
      SUM = 200,000
      MEAN = 200,000 / 10,100 = 19.80198... (0% error!)
    """

    def test_weighted_vs_unweighted_mean_under_skew(self):
        sampler = OASRS[dict](sample_size=100, stratum_key="stratum", seed=42)

        for _ in range(10000):
            sampler.update({"stratum": "heavy", "val": 10.0})
        for _ in range(100):
            sampler.update({"stratum": "light", "val": 1000.0})

        snap = sampler.snapshot()

        exact_sum = 10000 * 10.0 + 100 * 1000.0
        exact_mean = exact_sum / (10000 + 100)

        # Unweighted average of pooled reservoir items:
        all_sampled_vals = [
            r["val"]
            for s_snap in snap.strata.values()
            for r in s_snap.samples
        ]
        unweighted_mean = sum(all_sampled_vals) / len(all_sampled_vals)

        # OASRS weighted approximate mean:
        weighted_mean = approximate_mean(snap, value_key="val")
        weighted_sum = approximate_sum(snap, value_key="val")

        assert pytest.approx(weighted_sum, rel=1e-6) == exact_sum
        assert pytest.approx(weighted_mean, rel=1e-6) == exact_mean

        # Prove unweighted mean is drastically biased
        unweighted_error = abs(unweighted_mean - exact_mean) / exact_mean
        weighted_error = abs(weighted_mean - exact_mean) / exact_mean

        assert unweighted_error > 20.0       # > 2000% error
        assert weighted_error < 1e-5         # ~ 0% error


class TestValueExtractionFlexibility:
    """Validate numeric value extraction across diverse input formats."""

    def test_numeric_scalar_stream(self):
        """Stream of bare floats or ints."""
        sampler = OASRS[float](sample_size=10, stratum_key=lambda x: "default", seed=1)
        sampler.update_batch([1.5, 2.5, 3.5, 4.5])
        snap = sampler.snapshot()

        assert approximate_sum(snap) == 12.0
        assert approximate_mean(snap) == 3.0

    def test_callable_value_key(self):
        sampler = OASRS[tuple](sample_size=10, stratum_key=lambda x: x[0], seed=1)
        sampler.update_batch([("A", 10), ("A", 20), ("B", 30)])
        snap = sampler.snapshot()

        assert approximate_sum(snap, value_key=lambda t: t[1]) == 60.0
        assert approximate_mean(snap, value_key=lambda t: t[1]) == 20.0

    def test_direct_sampler_pass_through(self):
        """User can pass either OASRS instance or snapshot directly."""
        sampler = OASRS[dict](sample_size=10, stratum_key="s")
        sampler.update({"s": "1", "val": 42.0})

        assert approximate_sum(sampler, value_key="val") == 42.0
        assert approximate_mean(sampler, value_key="val") == 42.0

    def test_filtered_approximate_count(self):
        """Approximate count with predicate."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=1)
        # s1: 20 items (values 0..19), N=10 -> W = 2.0
        for i in range(20):
            sampler.update({"sensor": "s1", "temp": float(i)})

        snap = sampler.snapshot()
        # All items >= 10 in stratum s1
        filtered_cnt = approximate_count(snap, predicate=lambda r: r["temp"] >= 10.0)
        # In sample of 10 items, each item represents W=2.0
        # Since reservoir sampled 10 items from 20, count should approximate 10 * 2 = 20 or matching * W
        snap_s1 = snap.get_stratum("s1")
        assert snap_s1 is not None
        matching_in_sample = sum(1 for r in snap_s1.samples if r["temp"] >= 10.0)
        assert filtered_cnt == matching_in_sample * snap_s1.weight
