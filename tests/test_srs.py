"""Tests for Simple Random Sampling (SRS) Baseline Sampler."""

import pytest

from streamapprox.srs import SRSSampler


class TestSRSSamplerBasics:
    """Validate core SRS reservoir and weight mechanics."""

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError, match="sample_size must be >= 1"):
            SRSSampler(sample_size=0)

    def test_srs_reservoir_size_invariants(self):
        sampler = SRSSampler[dict](sample_size=10, seed=42)
        assert sampler.total_seen == 0
        assert sampler.sample_size == 0
        assert sampler.weight == 1.0

        for i in range(5):
            sampler.update({"val": i})
        assert sampler.total_seen == 5
        assert sampler.sample_size == 5
        assert sampler.weight == 1.0

        for i in range(5, 50):
            sampler.update({"val": i})
        assert sampler.total_seen == 50
        assert sampler.sample_size == 10
        assert sampler.weight == 5.0  # 50 / 10

    def test_srs_approximate_sum_and_mean(self):
        sampler = SRSSampler[float](sample_size=10, seed=42)
        # 10 items of value 20.0
        for _ in range(10):
            sampler.update(20.0)
        assert sampler.approximate_sum() == 200.0
        assert sampler.approximate_mean() == 20.0

        # Feed 10 more
        for _ in range(10):
            sampler.update(20.0)
        assert sampler.approximate_sum() == 400.0
        assert sampler.approximate_mean() == 20.0

    def test_srs_stratum_representation_and_starvation(self):
        """Under extreme skew, SRS starves minority strata."""
        sampler = SRSSampler[dict](sample_size=50, seed=42)

        # 999 items of stratum A, 1 item of stratum B
        for _ in range(999):
            sampler.update({"sensor_id": "A", "val": 1.0})
        sampler.update({"sensor_id": "B", "val": 1000.0})

        rep = sampler.stratum_representation("sensor_id")
        assert rep.get("A", 0) > 0
        # With probability 50/1000 = 5%, minority item B is overwhelmingly likely to be absent (starved)
        assert rep.get("B", 0) <= 1

    def test_srs_reset_and_repr(self):
        sampler = SRSSampler[int](sample_size=5)
        sampler.update_batch([1, 2, 3, 4, 5, 6])
        assert sampler.total_seen == 6
        assert "SRSSampler" in repr(sampler)

        sampler.reset()
        assert sampler.total_seen == 0
        assert sampler.sample_size == 0
        assert sampler.approximate_sum() == 0.0
        assert sampler.approximate_mean() == 0.0
