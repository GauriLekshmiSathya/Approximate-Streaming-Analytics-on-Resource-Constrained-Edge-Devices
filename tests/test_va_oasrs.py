from streamapprox.aggregators import approximate_mean
from streamapprox.va_oasrs import VAOASRS


def test_va_oasrs_allocates_fixed_budget_after_online_pilot():
    records = [
        {"sensor_id": "low", "value": float(i % 2)}
        if i % 2 == 0 else {"sensor_id": "high", "value": float(i % 2) * 20.0}
        for i in range(400)
    ]
    sampler = VAOASRS(
        40, "sensor_id", "value", seed=3,
        reallocation_interval=100, burn_in_records=100,
    )
    sampler.update_batch(records)
    snapshot = sampler.finalize()
    assert snapshot.total_seen == len(records)
    assert snapshot.sample_size == 40
    assert sum(s.capacity for s in snapshot.strata.values()) == 40
    assert all(s.sample_size > 0 for s in snapshot.strata.values())
    assert 0 <= approximate_mean(snapshot, "value") <= 20


def test_va_oasrs_uses_only_seen_values_to_allocate():
    sampler = VAOASRS(
        12, "sensor_id", "value", seed=1,
        reallocation_interval=4, burn_in_records=4,
    )
    sampler.update_batch([
        {"sensor_id": "A", "value": 1.0},
        {"sensor_id": "B", "value": 1.0},
        {"sensor_id": "A", "value": 1.0},
        {"sensor_id": "B", "value": 100.0},
    ])
    scores = sampler.allocation_history[0]["scores"]
    assert set(scores) == {"A", "B"}
    assert scores["A"] != scores["B"]

