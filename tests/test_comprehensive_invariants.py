"""Comprehensive Invariant, Statistical Coverage, and Skew Stress Tests for StreamApprox.

Fulfills Phase 6 and Section 9 (Testing) of project requirements:
- Empirical confidence interval coverage rate (~95% coverage under Equation 9)
- Extreme skew streams (80% : 19.99% : 0.01% as in Paper §5.7)
- Memory boundedness invariant under 200,000+ streaming items
- Numerical stability with large magnitudes (10^9 - 10^14)
- Dynamic arrival rate shift (8K:2K:100 -> 100:2K:8K as in Paper §5.4)
"""

import math
import random
import numpy as np
import pytest

from streamapprox.aggregators import approximate_mean, approximate_sum
from streamapprox.estimators import estimate_mean, estimate_sum
from streamapprox.oasrs import OASRS


class TestEmpiricalConfidenceIntervalCoverage:
    """Validate that Equation 9 produces rigorous 95% confidence intervals covering the ground truth.

    We simulate 200 independent stream runs from a heterogeneous 3-strata Gaussian mixture.
    For each run, we calculate the exact ground truth mean and the 95% confidence interval.
    The empirical coverage proportion must be close to 0.95 (e.g. >= 90%).
    """

    def test_95_percent_confidence_interval_coverage(self):
        num_runs = 200
        coverage_count = 0
        target_confidence = 0.95
        z_95 = 1.96

        for run_idx in range(num_runs):
            run_rng = np.random.RandomState(1000 + run_idx)
            py_rng = random.Random(2000 + run_idx)

            # 3 strata with different distributions
            # S1: N(10, 2), 500 items
            # S2: N(50, 5), 300 items
            # S3: N(100, 10), 200 items
            n1, n2, n3 = 500, 300, 200
            s1_items = run_rng.normal(10.0, 2.0, size=n1).tolist()
            s2_items = run_rng.normal(50.0, 5.0, size=n2).tolist()
            s3_items = run_rng.normal(100.0, 10.0, size=n3).tolist()

            all_records = (
                [{"s": "S1", "v": float(x)} for x in s1_items]
                + [{"s": "S2", "v": float(x)} for x in s2_items]
                + [{"s": "S3", "v": float(x)} for x in s3_items]
            )
            py_rng.shuffle(all_records)

            exact_mean = sum(r["v"] for r in all_records) / len(all_records)

            # Sample size 50 per stratum
            sampler = OASRS[dict](
                sample_size=50,
                stratum_key="s",
                seed=3000 + run_idx,
            )
            sampler.update_batch(all_records)
            snap = sampler.snapshot()

            est = estimate_mean(
                snap, value_key="v", confidence_level=target_confidence, z_score=z_95
            )

            # Check if ground truth falls within [lower_bound, upper_bound]
            if est.lower_bound <= exact_mean <= est.upper_bound:
                coverage_count += 1

        empirical_coverage = coverage_count / num_runs
        # With 200 runs, binomial SE = sqrt(0.95 * 0.05 / 200) = 0.0154.
        # Acceptable empirical range: >= 0.89
        assert empirical_coverage >= 0.89, (
            f"Empirical 95% CI coverage was {empirical_coverage:.3f}, expected >= 0.89"
        )


class TestExtremeSkewLongTailFidelity:
    """Validate OASRS under extreme paper skew §5.7 (Poisson 80% : 19.99% : 0.01%).

    Total records: 100,000
    A = 80,000 records (80%)
    B = 19,990 records (19.99%)
    C = 10 records (0.01%)
    """

    def test_extreme_skew_minority_representation_and_accuracy(self):
        sample_size = 50
        sampler = OASRS[dict](
            sample_size=sample_size,
            stratum_key="stratum",
            seed=42,
        )

        count_a = 80000
        count_b = 19990
        count_c = 10

        # Sub-stream A (common low value, e.g. 10)
        # Sub-stream B (medium value, e.g. 100)
        # Sub-stream C (rare extreme high value, e.g. 100,000)
        val_a = 10.0
        val_b = 100.0
        val_c = 100000.0

        for _ in range(count_a):
            sampler.update({"stratum": "A", "val": val_a})
        for _ in range(count_b):
            sampler.update({"stratum": "B", "val": val_b})
        for _ in range(count_c):
            sampler.update({"stratum": "C", "val": val_c})

        snap = sampler.snapshot()

        # Invariant: Total seen matches
        assert snap.total_seen == 100000

        # Minority stratum C must have ALL 10 records retained (not starved)
        snap_c = snap.get_stratum("C")
        assert snap_c is not None
        assert snap_c.total_seen == 10
        assert snap_c.sample_size == 10
        assert snap_c.weight == 1.0

        # Majority stratum A sampled down to sample_size
        snap_a = snap.get_stratum("A")
        assert snap_a is not None
        assert snap_a.total_seen == count_a
        assert snap_a.sample_size == sample_size
        assert pytest.approx(snap_a.weight, rel=1e-6) == count_a / sample_size

        # Exact values:
        exact_sum = count_a * val_a + count_b * val_b + count_c * val_c
        exact_mean = exact_sum / snap.total_seen

        approx_s = approximate_sum(snap, value_key="val")
        approx_m = approximate_mean(snap, value_key="val")

        # Because values in each stratum were constant here, approximate sum and mean are EXACT
        assert pytest.approx(approx_s, rel=1e-6) == exact_sum
        assert pytest.approx(approx_m, rel=1e-6) == exact_mean


class TestMemoryBoundednessInvariant:
    """Verify memory boundedness: total sampled items NEVER exceed sum_i N_i regardless of stream scale."""

    def test_memory_strictly_bounded_over_large_stream(self):
        sample_size = 100
        num_strata = 5
        sampler = OASRS[dict](
            sample_size=sample_size,
            stratum_key="sid",
            seed=101,
        )

        total_stream_items = 200000
        strata = [f"sensor_{i}" for i in range(num_strata)]

        for i in range(total_stream_items):
            sid = strata[i % num_strata]
            sampler.update({"sid": sid, "metric": float(i)})

        # Total seen is 200,000
        assert sampler.total_seen == total_stream_items
        # Active strata count is exactly 5
        assert sampler.num_strata == num_strata
        # Memory / Sample Size invariant: exactly num_strata * sample_size = 500 items!
        assert sampler.sample_size == num_strata * sample_size
        assert len(sampler) == 500

        snap = sampler.snapshot()
        assert snap.sample_size == 500
        for s in snap:
            assert s.sample_size == sample_size
            assert s.total_seen == total_stream_items // num_strata


class TestNumericalStabilityAndMagnitudes:
    """Verify numerical stability with large numbers (10^9 to 10^14) and tiny numbers (10^-6)."""

    def test_large_magnitudes_network_flow_sizes(self):
        """Simulate NetFlow byte counts (10^9 bytes) as in CAIDA case study (§6.2)."""
        sampler = OASRS[dict](sample_size=50, stratum_key="proto", seed=77)

        # 10,000 flows of 1 GB each
        gb = 10**9
        for _ in range(10000):
            sampler.update({"proto": "TCP", "bytes": float(gb)})

        snap = sampler.snapshot()
        res_sum = estimate_sum(snap, value_key="bytes")
        res_mean = estimate_mean(snap, value_key="bytes")

        expected_total_bytes = 10000.0 * gb  # 10^13
        assert pytest.approx(res_sum.estimate, rel=1e-6) == expected_total_bytes
        assert pytest.approx(res_mean.estimate, rel=1e-6) == float(gb)
        assert not math.isnan(res_sum.variance)
        assert not math.isinf(res_sum.variance)

    def test_tiny_fractional_magnitudes(self):
        """Simulate small error rates (10^-6)."""
        sampler = OASRS[dict](sample_size=50, stratum_key="s", seed=88)
        tiny = 1e-6
        for i in range(5000):
            sampler.update({"s": "A", "val": tiny * (1.0 + (i % 5))})

        snap = sampler.snapshot()
        res_mean = estimate_mean(snap, value_key="val")
        assert res_mean.estimate > 0.0
        assert res_mean.estimate < 1e-4
        assert not math.isnan(res_mean.standard_error)


class TestDynamicArrivalRateShift:
    """Reproduce Section 5.4 dynamic arrival rate shift: 8K:2K:100 -> 100:2K:8K."""

    def test_dynamic_shift_adapts_without_reconfiguration(self):
        sampler = OASRS[dict](sample_size=100, stratum_key="s", seed=99)

        # Epoch 1: A=8000, B=2000, C=100
        for _ in range(8000):
            sampler.update({"s": "A", "val": 10.0})
        for _ in range(2000):
            sampler.update({"s": "B", "val": 50.0})
        for _ in range(100):
            sampler.update({"s": "C", "val": 100.0})

        snap_epoch1 = sampler.snapshot()
        assert snap_epoch1["A"].total_seen == 8000
        assert snap_epoch1["B"].total_seen == 2000
        assert snap_epoch1["C"].total_seen == 100
        assert pytest.approx(snap_epoch1["A"].weight, rel=1e-6) == 80.0
        assert pytest.approx(snap_epoch1["B"].weight, rel=1e-6) == 20.0
        assert snap_epoch1["C"].weight == 1.0  # C <= N

        # Reset for epoch 2 to simulate next sliding window
        sampler.reset()

        # Epoch 2: Shifted rates A=100, B=2000, C=8000
        for _ in range(100):
            sampler.update({"s": "A", "val": 10.0})
        for _ in range(2000):
            sampler.update({"s": "B", "val": 50.0})
        for _ in range(8000):
            sampler.update({"s": "C", "val": 100.0})

        snap_epoch2 = sampler.snapshot()
        assert snap_epoch2["A"].total_seen == 100
        assert snap_epoch2["B"].total_seen == 2000
        assert snap_epoch2["C"].total_seen == 8000
        assert snap_epoch2["A"].weight == 1.0  # A <= N
        assert pytest.approx(snap_epoch2["B"].weight, rel=1e-6) == 20.0
        assert pytest.approx(snap_epoch2["C"].weight, rel=1e-6) == 80.0
