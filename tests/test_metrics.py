"""Tests for performance and resource instrumentation."""

import os
import tempfile
import pytest

from streamapprox.metrics import (
    BenchmarkResult,
    BenchmarkTimer,
    get_peak_rss_mb,
    save_benchmark_results,
)


class TestMetricsAndTimer:
    """Validate resource measurements, timers, and serialization."""

    def test_peak_rss_measurement(self):
        rss = get_peak_rss_mb()
        assert rss > 0.0  # Python process uses > 0 MB of RAM

    def test_benchmark_timer_computations(self):
        with BenchmarkTimer(record_count=1000) as timer:
            # Short workload
            total = sum(i for i in range(10000))

        assert timer.elapsed_sec > 0.0
        assert timer.throughput > 0.0
        assert timer.latency_us > 0.0
        assert timer.cpu_utilization_pct >= 0.0
        assert timer.peak_rss_mb > 0.0

    def test_benchmark_result_serialization(self):
        res = BenchmarkResult(
            method="OASRS",
            dataset="Gaussian_Paper",
            total_records=10000,
            sample_size=500,
            sampling_fraction_pct=5.0,
            exact_sum=1000.0,
            approx_sum=998.5,
            exact_mean=10.0,
            approx_mean=9.985,
            sum_accuracy_loss_pct=0.15,
            mean_accuracy_loss_pct=0.15,
            elapsed_time_sec=0.015,
            cpu_time_sec=0.014,
            throughput_items_per_sec=666666.67,
            latency_us_per_item=1.5,
            peak_rss_mb=45.2,
            stratum_representation={"A": 200, "B": 200, "C": 100},
        )

        d = res.to_dict()
        assert d["method"] == "OASRS"
        assert d["total_records"] == 10000

        csv_line = res.to_csv_row()
        assert "OASRS" in csv_line
        assert "Gaussian_Paper" in csv_line

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file, csv_file = save_benchmark_results([res], tmpdir, "test_res")
            assert os.path.exists(json_file)
            assert os.path.exists(csv_file)
            with open(csv_file) as f:
                lines = f.readlines()
                assert len(lines) == 2  # header + row
