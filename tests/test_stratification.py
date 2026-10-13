"""Tests for OASRS stratification under heavy skew and arrival-rate imbalances."""

import pytest
from streamapprox.oasrs import OASRS


class TestStratificationSkew:
    """Validate OASRS behavior under highly imbalanced stream arrival rates.

    Specification from project guidelines:
    Generate highly imbalanced streams:
      A = 100,000 records
      B = 10,000 records
      C = 100 records
    Verify every stratum receives its own reservoir.
    """

    def test_imbalanced_stream_preserves_minority_strata(self):
        """Under severe skew (100,000 vs 10,000 vs 100), verify OASRS preserves minority strata."""
        sample_size_per_stratum = 100
        sampler = OASRS[dict](
            sample_size=sample_size_per_stratum,
            stratum_key="stratum",
            seed=42,
        )

        count_a = 100000
        count_b = 10000
        count_c = 100

        # Feed stream in interleaved chunks to emulate streaming arrival
        chunk_size = 100
        for _ in range(count_c // chunk_size):
            for _ in range(chunk_size):
                sampler.update({"stratum": "C", "val": 3.0})

        for _ in range(count_b // chunk_size):
            for _ in range(chunk_size):
                sampler.update({"stratum": "B", "val": 2.0})

        for _ in range(count_a // chunk_size):
            for _ in range(chunk_size):
                sampler.update({"stratum": "A", "val": 1.0})

        snap = sampler.snapshot()

        # Invariant 1: Exactly 3 strata discovered
        assert snap.num_strata == 3
        assert set(snap.strata.keys()) == {"A", "B", "C"}

        # Invariant 2: Total seen matches exact stream length
        expected_total = count_a + count_b + count_c
        assert snap.total_seen == expected_total

        # Invariant 3: Check stratum A (majority)
        snap_a = snap.get_stratum("A")
        assert snap_a is not None
        assert snap_a.total_seen == count_a
        assert snap_a.sample_size == sample_size_per_stratum
        assert pytest.approx(snap_a.weight, rel=1e-6) == count_a / sample_size_per_stratum
        assert snap_a.is_exact is False

        # Invariant 4: Check stratum B (medium)
        snap_b = snap.get_stratum("B")
        assert snap_b is not None
        assert snap_b.total_seen == count_b
        assert snap_b.sample_size == sample_size_per_stratum
        assert pytest.approx(snap_b.weight, rel=1e-6) == count_b / sample_size_per_stratum
        assert snap_b.is_exact is False

        # Invariant 5: Check stratum C (minority)
        # Since count_c == 100 <= N_i=100, ALL 100 records are retained (exact sample)
        snap_c = snap.get_stratum("C")
        assert snap_c is not None
        assert snap_c.total_seen == count_c
        assert snap_c.sample_size == count_c
        assert snap_c.weight == 1.0
        assert snap_c.is_exact is True
        assert snap_c.sampling_fraction == 1.0

        # Invariant 6: Total sampled records across all reservoirs
        assert snap.sample_size == sample_size_per_stratum * 2 + count_c

    def test_dynamic_stratum_emergence(self):
        """New strata arriving late in the stream are dynamically discovered without pre-configuration."""
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=101)

        # Only sensor 1 active in first batch
        for i in range(50):
            sampler.update({"sensor": "s1", "temp": 20.0 + i})

        assert sampler.num_strata == 1
        assert "s1" in sampler.strata_ids

        # Sensor 2 appears later
        for i in range(30):
            sampler.update({"sensor": "s2", "temp": 25.0 + i})

        assert sampler.num_strata == 2
        assert "s2" in sampler.strata_ids

        # Sensor 3 appears even later
        for i in range(5):
            sampler.update({"sensor": "s3", "temp": 18.0 + i})

        assert sampler.num_strata == 3
        assert "s3" in sampler.strata_ids

        snap = sampler.snapshot()
        assert snap["s1"].total_seen == 50
        assert snap["s2"].total_seen == 30
        assert snap["s3"].total_seen == 5
        assert snap["s3"].sample_size == 5
        assert snap["s3"].weight == 1.0
