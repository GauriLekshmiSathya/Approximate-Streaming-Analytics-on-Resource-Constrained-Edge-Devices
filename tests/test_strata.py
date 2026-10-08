"""Tests for Stratum and weight calculation (Section 3.2, Equation 1)."""

import pytest
from streamapprox.strata import Stratum


class TestStratumWeights:
    """Validate weight calculation formula:

    W_i = C_i / N_i  if C_i > N_i
    W_i = 1.0        if C_i <= N_i
    """

    def test_weight_when_stream_is_empty(self):
        """When C_i == 0, C_i <= N_i, weight must be 1.0."""
        s = Stratum[int](stratum_id="sensor_1", capacity=10)
        assert s.total_seen == 0
        assert s.sample_size == 0
        assert s.weight == 1.0

    def test_weight_when_total_seen_less_than_capacity(self):
        """When C_i < N_i, weight must be 1.0."""
        s = Stratum[int](stratum_id="sensor_1", capacity=10)
        for i in range(5):
            s.update(i)
        assert s.total_seen == 5
        assert s.sample_size == 5
        assert s.weight == 1.0

    def test_weight_when_total_seen_equals_capacity(self):
        """When C_i == N_i, weight must be 1.0."""
        s = Stratum[int](stratum_id="sensor_1", capacity=10)
        for i in range(10):
            s.update(i)
        assert s.total_seen == 10
        assert s.sample_size == 10
        assert s.weight == 1.0

    def test_weight_when_total_seen_exceeds_capacity(self):
        """When C_i > N_i, weight must be exactly C_i / N_i."""
        s = Stratum[int](stratum_id="sensor_1", capacity=10, seed=42)
        for i in range(25):
            s.update(i)
        assert s.total_seen == 25
        assert s.sample_size == 10
        assert s.weight == 2.5

    def test_figure_2_weights_from_paper(self):
        """Reproduce Figure 2 exact weights from Middleware '17 paper:

        Figure 2:
        S1: C1 = 6, N1 = 3 -> W1 = 6 / 3 = 2.0
        S2: C2 = 4, N2 = 3 -> W2 = 4 / 3 = 1.3333...
        S3: C3 = 2, N3 = 3 -> W3 = 1.0
        """
        s1 = Stratum[int](stratum_id="S1", capacity=3, seed=1)
        s2 = Stratum[int](stratum_id="S2", capacity=3, seed=2)
        s3 = Stratum[int](stratum_id="S3", capacity=3, seed=3)

        for i in range(6):
            s1.update(i)
        for i in range(4):
            s2.update(i)
        for i in range(2):
            s3.update(i)

        assert s1.total_seen == 6
        assert s1.sample_size == 3
        assert pytest.approx(s1.weight, rel=1e-6) == 2.0

        assert s2.total_seen == 4
        assert s2.sample_size == 3
        assert pytest.approx(s2.weight, rel=1e-6) == 4.0 / 3.0

        assert s3.total_seen == 2
        assert s3.sample_size == 2
        assert s3.weight == 1.0


class TestStratumSnapshotAndReset:
    """Validate snapshot creation, isolation, and reset functionality."""

    def test_snapshot_preserves_state(self):
        s = Stratum[dict](stratum_id="edge_node_A", capacity=5, seed=10)
        records = [{"val": i} for i in range(12)]
        for r in records:
            s.update(r)

        snap = s.snapshot()
        assert snap.stratum_id == "edge_node_A"
        assert snap.total_seen == 12
        assert snap.capacity == 5
        assert snap.sample_size == 5
        assert snap.weight == 2.4
        assert snap.is_exact is False
        assert pytest.approx(snap.sampling_fraction, rel=1e-6) == 5.0 / 12.0

        # Verify snapshot is decoupled: modifying s or snapshot doesn't bleed
        s.update({"val": 999})
        assert s.total_seen == 13
        assert snap.total_seen == 12

    def test_reset_clears_stratum(self):
        s = Stratum[int](stratum_id="sensor_B", capacity=4)
        for i in range(10):
            s.update(i)

        assert s.total_seen == 10
        assert s.sample_size == 4

        s.reset()
        assert s.total_seen == 0
        assert s.sample_size == 0
        assert s.weight == 1.0
        assert s.samples == []
