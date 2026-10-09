"""Tests for Reservoir Sampling (Algorithm 1)."""

import math
import random
import pytest

from streamapprox.reservoir import Reservoir


class TestReservoirInvariants:
    """Validate core invariants specified in Algorithm 1 of StreamApprox."""

    def test_invalid_capacity_raises(self):
        """Capacity must be an integer >= 1."""
        with pytest.raises(ValueError, match="positive integer"):
            Reservoir(capacity=0)

        with pytest.raises(ValueError, match="positive integer"):
            Reservoir(capacity=-5)

        with pytest.raises(ValueError, match="positive integer"):
            Reservoir(capacity="10")  # type: ignore

    def test_empty_reservoir_initial_state(self):
        """Newly created reservoir must be empty."""
        r = Reservoir[int](capacity=10)
        assert r.capacity == 10
        assert r.total_seen == 0
        assert r.sample_size == 0
        assert len(r) == 0
        assert r.items == []
        assert not r.is_full

    def test_stream_smaller_than_capacity(self):
        """When stream size C < N, all items must be stored (|reservoir| = C)."""
        capacity = 10
        r = Reservoir[int](capacity=capacity)
        items = list(range(6))
        for item in items:
            accepted = r.update(item)
            assert accepted is True

        assert r.total_seen == 6
        assert r.sample_size == 6
        assert r.items == [0, 1, 2, 3, 4, 5]
        assert not r.is_full

    def test_stream_equal_to_capacity(self):
        """When stream size C == N, exactly N items must be stored."""
        capacity = 5
        r = Reservoir[int](capacity=capacity)
        items = [10, 20, 30, 40, 50]
        for item in items:
            accepted = r.update(item)
            assert accepted is True

        assert r.total_seen == 5
        assert r.sample_size == 5
        assert r.is_full
        assert r.items == [10, 20, 30, 40, 50]

    def test_stream_larger_than_capacity_size_bound(self):
        """When stream size C > N, reservoir size must NEVER exceed capacity N."""
        capacity = 20
        r = Reservoir[int](capacity=capacity, seed=42)
        stream_size = 1000

        for i in range(stream_size):
            r.update(i)
            # Invariant check at every single step
            assert len(r) <= capacity
            assert r.sample_size <= capacity

        assert r.total_seen == stream_size
        assert r.sample_size == capacity
        assert r.is_full
        assert len(r.items) == capacity

    def test_reservoir_reset(self):
        """Reset clears all stored items and counter."""
        r = Reservoir[str](capacity=5)
        for ch in ["a", "b", "c", "d", "e", "f", "g"]:
            r.update(ch)

        assert r.total_seen == 7
        assert r.sample_size == 5

        r.reset()
        assert r.total_seen == 0
        assert r.sample_size == 0
        assert r.items == []
        assert not r.is_full

    def test_items_property_returns_copy(self):
        """Mutating items returned by .items must not alter the reservoir."""
        r = Reservoir[int](capacity=5)
        for i in range(3):
            r.update(i)

        snapshot = r.items
        snapshot.append(999)
        assert r.items == [0, 1, 2]
        assert r.sample_size == 3

    def test_reservoir_repr(self):
        r = Reservoir[int](capacity=5)
        assert "Reservoir" in repr(r)



class TestReservoirStatisticalProperties:
    """Validate that Algorithm 1 yields a uniform random sample."""

    def test_uniform_inclusion_probability(self):
        """Each stream item should have equal probability N / C of being in the reservoir.

        We simulate stream of length C=100 with capacity N=10 over 10,000 trials.
        Expected inclusion probability per item: 10 / 100 = 0.10.
        By binomial standard error: SE = sqrt(0.10 * 0.90 / 10000) = 0.003.
        A 4-sigma margin is 0.012 -> acceptable range [0.088, 0.112].
        """
        capacity = 10
        stream_size = 100
        num_trials = 10000
        counts = [0] * stream_size

        rng = random.Random(12345)
        for _ in range(num_trials):
            r = Reservoir[int](capacity=capacity, rng=rng)
            for item in range(stream_size):
                r.update(item)
            for selected in r.items:
                counts[selected] += 1

        expected_prob = capacity / stream_size  # 0.10
        margin = 0.015

        for item_idx, count in enumerate(counts):
            emp_prob = count / num_trials
            assert abs(emp_prob - expected_prob) < margin, (
                f"Item {item_idx} empirical inclusion prob {emp_prob:.4f} "
                f"deviated from expected {expected_prob:.4f}"
            )

    def test_deterministic_reproducibility(self):
        """With identical seed, reservoirs produce identical samples."""
        r1 = Reservoir[int](capacity=15, seed=999)
        r2 = Reservoir[int](capacity=15, seed=999)

        stream = list(range(200))
        for x in stream:
            r1.update(x)
            r2.update(x)

        assert r1.items == r2.items
