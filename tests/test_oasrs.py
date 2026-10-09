"""Tests for OASRS (Online Adaptive Stratified Reservoir Sampling - Algorithm 3)."""

from dataclasses import dataclass
import pytest

from streamapprox.oasrs import OASRS


class TestOASRSBasics:
    """Validate core OASRS functionality and configuration."""

    def test_invalid_sample_size_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            OASRS(sample_size=0)
        with pytest.raises(ValueError, match="positive integer"):
            OASRS(sample_size=-10)

    def test_callable_stratum_key(self):
        sampler = OASRS[dict](
            sample_size=5,
            stratum_key=lambda r: r["sensor_id"],
        )
        stream = [
            {"sensor_id": "temp_1", "value": 20.5},
            {"sensor_id": "temp_2", "value": 15.0},
            {"sensor_id": "temp_1", "value": 21.0},
        ]
        sampler.update_batch(stream)

        assert sampler.num_strata == 2
        assert set(sampler.strata_ids) == {"temp_1", "temp_2"}
        assert sampler.total_seen == 3
        assert sampler.sample_size == 3

    def test_string_stratum_key_dict(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="sensor_id")
        sampler.update({"sensor_id": "s1", "v": 10})
        sampler.update({"sensor_id": "s2", "v": 20})
        assert sampler.num_strata == 2

    def test_string_stratum_key_missing_key_raises(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="sensor_id")
        with pytest.raises(KeyError, match="missing stratum key"):
            sampler.update({"other_key": "s1"})

    def test_object_attribute_stratum_key(self):
        @dataclass
        class Reading:
            device: str
            val: float

        sampler = OASRS[Reading](sample_size=5, stratum_key="device")
        sampler.update(Reading(device="devA", val=1.0))
        sampler.update(Reading(device="devB", val=2.0))
        assert sampler.num_strata == 2

    def test_invalid_stratum_key_type_raises(self):
        with pytest.raises(TypeError, match="must be callable or str"):
            OASRS(sample_size=5, stratum_key=123)  # type: ignore


class TestOASRSStateAndWeights:
    """Validate multi-strata state tracking and weight derivation."""

    def test_figure_2_multi_strata_weights(self):
        """Reproduce Figure 2 from the Middleware '17 paper in full OASRS:
        S1: 6 arrivals, N1 = 3 -> W1 = 6 / 3 = 2.0
        S2: 4 arrivals, N2 = 3 -> W2 = 4 / 3 = 1.3333...
        S3: 2 arrivals, N3 = 3 -> W3 = 1.0
        """
        sampler = OASRS[dict](sample_size=3, stratum_key="sensor", seed=42)

        # Interleave stream items across strata
        stream = (
            [{"sensor": "S1", "v": i} for i in range(6)]
            + [{"sensor": "S2", "v": i} for i in range(4)]
            + [{"sensor": "S3", "v": i} for i in range(2)]
        )
        sampler.update_batch(stream)

        assert sampler.num_strata == 3
        assert sampler.total_seen == 12
        assert sampler.sample_size == 8  # 3 + 3 + 2

        weights = sampler.weights
        assert pytest.approx(weights["S1"], rel=1e-6) == 2.0
        assert pytest.approx(weights["S2"], rel=1e-6) == 4.0 / 3.0
        assert weights["S3"] == 1.0

    def test_custom_stratum_capacity_callable(self):
        """Strata can have varying sample capacities N_i based on domain rules."""
        capacity_rule = lambda s_id: 10 if s_id == "high_prio" else 2
        sampler = OASRS[dict](
            sample_size=5,
            stratum_key="type",
            stratum_capacity=capacity_rule,
        )

        for i in range(20):
            sampler.update({"type": "high_prio", "v": i})
            sampler.update({"type": "low_prio", "v": i})

        high_stratum = sampler.get_stratum("high_prio")
        low_stratum = sampler.get_stratum("low_prio")
        assert high_stratum is not None
        assert low_stratum is not None

        assert high_stratum.capacity == 10
        assert high_stratum.sample_size == 10
        assert low_stratum.capacity == 2
        assert low_stratum.sample_size == 2

    def test_snapshot_isolation(self):
        """A snapshot must not be mutated by subsequent updates to the sampler."""
        sampler = OASRS[dict](sample_size=5, stratum_key="sid")
        sampler.update_batch([{"sid": "A", "val": 1}, {"sid": "B", "val": 2}])

        snap = sampler.snapshot()
        assert snap.total_seen == 2
        assert snap.num_strata == 2

        # Ingest more records
        sampler.update_batch([{"sid": "A", "val": 3}, {"sid": "C", "val": 4}])
        assert sampler.total_seen == 4
        assert sampler.num_strata == 3

        # Snapshot remains unchanged
        assert snap.total_seen == 2
        assert snap.num_strata == 2
        assert snap.get_stratum("C") is None

    def test_reset_functionality(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="sid")
        sampler.update_batch([{"sid": "A", "val": 1}, {"sid": "B", "val": 2}])

        # Soft reset: keeps registered strata, resets counts
        sampler.reset(clear_strata=False)
        assert sampler.total_seen == 0
        assert sampler.sample_size == 0
        assert sampler.num_strata == 2

        # Hard reset: clears strata registry
        sampler.reset(clear_strata=True)
        assert sampler.total_seen == 0
        assert sampler.sample_size == 0
        assert sampler.num_strata == 0

    def test_object_missing_attribute_raises_attributeerror(self):
        class Obj:
            pass

        sampler = OASRS[Obj](sample_size=5, stratum_key="sensor_id")
        with pytest.raises(AttributeError, match="missing stratum attribute"):
            sampler.update(Obj())

    def test_stratum_capacity_invalid_callable_raises(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="s", stratum_capacity=lambda sid: -1)
        with pytest.raises(ValueError, match="invalid capacity"):
            sampler.update({"s": "1"})

    def test_stratum_capacity_as_int(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="s", stratum_capacity=8)
        sampler.update({"s": "1"})
        assert sampler.get_stratum("1").capacity == 8

    def test_sampler_len_and_repr(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="s")
        sampler.update({"s": "1"})
        assert len(sampler) == 1
        assert "OASRS" in repr(sampler)

    def test_snapshot_iteration_len_getitem_and_weights(self):
        sampler = OASRS[dict](sample_size=5, stratum_key="s")
        sampler.update({"s": "A"})
        sampler.update({"s": "B"})
        snap = sampler.snapshot()

        assert len(snap) == 2
        assert snap["A"].stratum_id == "A"
        assert list(snap.weights.keys()) == ["A", "B"]
        iter_strata = [s.stratum_id for s in snap]
        assert iter_strata == ["A", "B"]

