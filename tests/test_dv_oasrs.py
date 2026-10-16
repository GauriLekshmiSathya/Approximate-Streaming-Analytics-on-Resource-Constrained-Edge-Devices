from collections import Counter
import random

import pytest

from streamapprox.aggregators import approximate_mean
from streamapprox.dv_oasrs import DVOASRS


def test_dynamic_sampler_stays_within_budget_and_allocates_late_stratum():
    sampler = DVOASRS(
        20, "sensor_id", "value", seed=4, reallocation_interval=100,
        variance_alpha=0.1, max_change_fraction=0.25,
    )
    sampler.update_batch([{"sensor_id": "early", "value": float(i % 2)} for i in range(500)])
    assert sampler.sample_size <= 20
    before = sampler.total_seen
    sampler.update({"sensor_id": "late", "value": 100.0})
    snap = sampler.snapshot()
    assert sampler.total_seen == before + 1
    assert sampler.strata["late"].capacity >= 1
    assert snap["late"].sample_size == 1
    assert sampler.sample_size <= 20
    assert sampler.allocation_history[-1]["records_seen"] == before + 1


def test_priority_sample_is_uniform_for_a_fixed_quota():
    appearances = Counter()
    runs = 500
    for seed in range(runs):
        sampler = DVOASRS(3, "sensor_id", "value", seed=seed,
                          reallocation_interval=10_000, max_change_fraction=1.0)
        sampler.update_batch([{"sensor_id": "A", "value": float(i)} for i in range(10)])
        appearances.update(item["value"] for item in sampler.finalize()["A"].samples)
    # Each of 10 records has inclusion probability 3/10; tolerate Monte Carlo noise.
    assert all(100 <= appearances[i] <= 200 for i in range(10))


def test_budget_smaller_than_active_strata_fails_explicitly():
    sampler = DVOASRS(1, "sensor_id", "value", seed=1)
    sampler.update({"sensor_id": "A", "value": 1.0})
    with pytest.raises(ValueError, match="budget"):
        sampler.update({"sensor_id": "B", "value": 2.0})


def test_weighted_mean_is_finite_after_dynamic_allocations():
    sampler = DVOASRS(30, "sensor_id", "value", seed=3,
                      reallocation_interval=20, variance_alpha=0.2)
    sampler.update_batch([
        {"sensor_id": "a", "value": float(i % 3)} for i in range(100)
    ] + [
        {"sensor_id": "b", "value": float(i % 5) * 10} for i in range(100)
    ])
    result = approximate_mean(sampler.finalize(), "value")
    assert 0 <= result <= 40


def test_ema_allocation_responds_to_variance_shift():
    rng = random.Random(28)
    sampler = DVOASRS(100, "sensor_id", "value", seed=8,
                      reallocation_interval=100, variance_alpha=0.05,
                      max_change_fraction=0.25)
    before = {}
    for i in range(2000):
        sid = "A" if i % 2 == 0 else "B"
        scale = 1.0 if sid == "A" or i < 1000 else 20.0
        sampler.update({"sensor_id": sid, "value": rng.gauss(0, scale)})
        if i == 999:
            before = {k: s.capacity for k, s in sampler.strata.items()}
    after = {k: s.capacity for k, s in sampler.strata.items()}
    assert after["B"] > before["B"]
    assert sampler.sample_size <= 100
