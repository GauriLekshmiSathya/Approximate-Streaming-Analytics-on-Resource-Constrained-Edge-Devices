"""Tests for sampling weight invariants and propagation (Equations 1, 2, 3, 4)."""

import pytest

from streamapprox.aggregators import approximate_mean, approximate_sum
from streamapprox.oasrs import OASRS
from streamapprox.strata import Stratum


class TestSamplingWeightInvariants:
    """Validate strict adherence to Equation 1:

    W_i = C_i / N_i  if C_i > N_i
    W_i = 1.0        if C_i <= N_i
    """

    def test_weight_transition_boundary(self):
        """Verify the exact boundary transition at C_i == N_i and C_i == N_i + 1."""
        capacity = 5
        stratum = Stratum[float](stratum_id="test", capacity=capacity, seed=42)

        # Before capacity: C_i = 1..4 -> W_i = 1.0
        for i in range(1, capacity):
            stratum.update(float(i))
            assert stratum.total_seen == i
            assert stratum.sample_size == i
            assert stratum.weight == 1.0

        # Exact boundary: C_i == N_i (5) -> W_i = 1.0
        stratum.update(5.0)
        assert stratum.total_seen == capacity
        assert stratum.sample_size == capacity
        assert stratum.weight == 1.0

        # First record over capacity: C_i == N_i + 1 (6) -> W_i = 6 / 5 = 1.2
        stratum.update(6.0)
        assert stratum.total_seen == capacity + 1
        assert stratum.sample_size == capacity
        assert pytest.approx(stratum.weight, rel=1e-6) == 1.2

        # Subsequent arrivals scale proportionally
        for i in range(7, 21):
            stratum.update(float(i))
        assert stratum.total_seen == 20
        assert stratum.sample_size == capacity
        assert pytest.approx(stratum.weight, rel=1e-6) == 20.0 / 5.0  # 4.0

    def test_weights_in_multi_strata_aggregation(self):
        """Verify weights correctly scale each stratum's contribution to global sum and mean."""
        # 3 strata with different arrival counts:
        # S1: C1 = 100, N1 = 10 -> W1 = 10.0 (all items value 1.0)
        # S2: C2 = 50,  N2 = 10 -> W2 = 5.0  (all items value 2.0)
        # S3: C3 = 5,   N3 = 10 -> W3 = 1.0  (all items value 3.0)
        sampler = OASRS[dict](sample_size=10, stratum_key="sensor", seed=99)

        for _ in range(100):
            sampler.update({"sensor": "S1", "v": 1.0})
        for _ in range(50):
            sampler.update({"sensor": "S2", "v": 2.0})
        for _ in range(5):
            sampler.update({"sensor": "S3", "v": 3.0})

        snap = sampler.snapshot()

        assert pytest.approx(snap["S1"].weight, rel=1e-6) == 10.0
        assert pytest.approx(snap["S2"].weight, rel=1e-6) == 5.0
        assert snap["S3"].weight == 1.0

        # Stratum S1 sample sum = 10 * 1.0 = 10.0 -> SUM_1 = 10.0 * 10.0 = 100.0
        # Stratum S2 sample sum = 10 * 2.0 = 20.0 -> SUM_2 = 20.0 * 5.0 = 100.0
        # Stratum S3 sample sum = 5 * 3.0 = 15.0  -> SUM_3 = 15.0 * 1.0 = 15.0
        # Total approximate SUM = 100 + 100 + 15 = 215.0
        # Total arrivals = 100 + 50 + 5 = 155
        # Total approximate MEAN = 215.0 / 155 = 1.387096...
        approx_s = approximate_sum(snap, value_key="v")
        approx_m = approximate_mean(snap, value_key="v")

        assert pytest.approx(approx_s, rel=1e-6) == 215.0
        assert pytest.approx(approx_m, rel=1e-6) == 215.0 / 155.0
