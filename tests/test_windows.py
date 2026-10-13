"""Tests for sliding and tumbling window abstractions (Section 2.2, 3.1, and Section 6)."""

import pytest

from streamapprox.windows import CountWindowManager, TimeWindowManager


class TestTimeWindowManagerTumbling:
    """Validate tumbling time-window semantics (duration == slide)."""

    def test_tumbling_window_emission_on_boundary(self):
        manager = TimeWindowManager[dict](
            window_duration=10.0,
            sample_size=10,
            stratum_key="sensor",
            value_key="val",
            base_time=0.0,
        )

        assert manager.is_tumbling is True

        # Ingest records within first window [0.0, 10.0)
        emitted_1 = manager.update({"sensor": "s1", "val": 10.0}, timestamp=1.0)
        emitted_2 = manager.update({"sensor": "s1", "val": 20.0}, timestamp=5.0)
        emitted_3 = manager.update({"sensor": "s2", "val": 30.0}, timestamp=9.5)
        assert len(emitted_1) == 0
        assert len(emitted_2) == 0
        assert len(emitted_3) == 0

        # Ingest record crossing window boundary (t = 10.1 >= 10.0)
        emitted_4 = manager.update({"sensor": "s1", "val": 40.0}, timestamp=10.1)
        assert len(emitted_4) == 1

        w1 = emitted_4[0]
        assert w1.window_start == 0.0
        assert w1.window_end == 10.0
        assert w1.total_seen == 3
        assert w1.num_strata == 2
        assert w1.sum_result.estimate == 60.0
        assert w1.mean_result.estimate == 20.0

        # Ingest another record in second window [10.0, 20.0)
        manager.update({"sensor": "s2", "val": 50.0}, timestamp=15.0)

        # Flush remaining window
        flushed = manager.flush()
        assert len(flushed) == 1
        w2 = flushed[0]
        assert w2.window_start == 10.0
        assert w2.window_end == 20.0
        assert w2.total_seen == 2  # 40.0 and 50.0
        assert w2.sum_result.estimate == 90.0
        assert w2.mean_result.estimate == 45.0


class TestTimeWindowManagerSliding:
    """Validate sliding time-window semantics (e.g. w=10s, slide=5s as in paper §5.7)."""

    def test_paper_sliding_window_10s_duration_5s_slide(self):
        """Reproduce paper evaluation sliding setup: duration = 10s, slide = 5s."""
        manager = TimeWindowManager[dict](
            window_duration=10.0,
            slide_interval=5.0,
            sample_size=10,
            stratum_key="sensor",
            value_key="val",
            base_time=0.0,
        )

        assert manager.is_tumbling is False
        assert manager.window_duration == 10.0
        assert manager.slide_interval == 5.0

        # Item at t = 2.0 covers [0, 10)
        manager.update({"sensor": "s1", "val": 10.0}, timestamp=2.0)

        # Item at t = 7.0 covers [0, 10) AND [5, 15)
        manager.update({"sensor": "s1", "val": 20.0}, timestamp=7.0)

        # Item at t = 12.0 covers [5, 15) AND [10, 20).
        # Since t = 12.0 >= 10.0, window [0, 10) must be completed and emitted!
        emitted = manager.update({"sensor": "s1", "val": 30.0}, timestamp=12.0)
        assert len(emitted) == 1

        w1 = emitted[0]
        assert w1.window_start == 0.0
        assert w1.window_end == 10.0
        # Records in [0, 10): t=2.0 (val 10) and t=7.0 (val 20)
        assert w1.total_seen == 2
        assert w1.sum_result.estimate == 30.0
        assert w1.mean_result.estimate == 15.0

        # Advance time to 18.0: window [5, 15) completes (since 15.0 <= 18.0)
        emitted_advance = manager.advance_time(18.0)
        assert len(emitted_advance) == 1

        w2 = emitted_advance[0]
        assert w2.window_start == 5.0
        assert w2.window_end == 15.0
        # Records in [5, 15): t=7.0 (val 20) and t=12.0 (val 30)
        assert w2.total_seen == 2
        assert w2.sum_result.estimate == 50.0
        assert w2.mean_result.estimate == 25.0

        # Flush remaining window [10, 20)
        flushed = manager.flush()
        assert len(flushed) == 1
        w3 = flushed[0]
        assert w3.window_start == 10.0
        assert w3.window_end == 20.0
        # Records in [10, 20): t=12.0 (val 30)
        assert w3.total_seen == 1
        assert w3.sum_result.estimate == 30.0


class TestCountWindowManager:
    """Validate micro-batch count windowing (§2.2 Spark Streaming model)."""

    def test_count_window_tumbling_micro_batches(self):
        manager = CountWindowManager[dict](
            window_size=5,
            sample_size=10,
            stratum_key="sensor",
            value_key="val",
        )

        stream = [{"sensor": "s1", "val": float(i)} for i in range(12)]
        results = list(manager.process_stream(stream))

        # 12 items with window size 5 -> 2 full windows + 1 flushed remainder
        assert len(results) == 3

        # Window 1: indices 0..5
        assert results[0].window_start == 0.0
        assert results[0].window_end == 5.0
        assert results[0].total_seen == 5
        assert results[0].sum_result.estimate == sum(range(5))  # 10.0

        # Window 2: indices 5..10
        assert results[1].window_start == 5.0
        assert results[1].window_end == 10.0
        assert results[1].total_seen == 5
        assert results[1].sum_result.estimate == sum(range(5, 10))  # 35.0

        # Window 3 (remainder flushed): indices 10..12
        assert results[2].window_start == 10.0
        assert results[2].window_end == 12.0
        assert results[2].total_seen == 2
        assert results[2].sum_result.estimate == 10.0 + 11.0  # 21.0


class TestWindowValidationAndEdgeCases:
    """Validate parameters, exceptions, and automatic timestamp extraction."""

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TimeWindowManager(window_duration=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            TimeWindowManager(window_duration=10.0, slide_interval=-1.0)

        with pytest.raises(ValueError, match="cannot exceed window_duration"):
            TimeWindowManager(window_duration=10.0, slide_interval=15.0)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            CountWindowManager(window_size=0)

    def test_timestamp_key_extraction(self):
        # Extract from dict key
        manager = TimeWindowManager[dict](
            window_duration=5.0,
            timestamp_key="ts",
            stratum_key="s",
            value_key="v",
        )
        res = list(manager.process_stream([
            {"s": "A", "v": 1.0, "ts": 1.0},
            {"s": "A", "v": 2.0, "ts": 6.0},
        ]))
        assert len(res) == 2
        assert res[0].total_seen == 1
        assert res[1].total_seen == 1

    def test_virtual_arrival_clock_when_no_timestamp(self):
        manager = TimeWindowManager[dict](
            window_duration=3.0,
            stratum_key="s",
            value_key="v",
        )
        # Without timestamps, virtual clock advances 1 unit per item:
        # Item 0 (t=0), Item 1 (t=1), Item 2 (t=2) -> [0, 3)
        # Item 3 (t=3) -> triggers [0, 3)
        results = list(manager.process_stream([
            {"s": "A", "v": 1.0},
            {"s": "A", "v": 1.0},
            {"s": "A", "v": 1.0},
            {"s": "A", "v": 2.0},
        ]))
        assert len(results) == 2
        assert results[0].total_seen == 3
        assert results[1].total_seen == 1
