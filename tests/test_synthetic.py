"""Tests for synthetic stream generators and ground truth calculation."""

import numpy as np
import pytest

from streamapprox.synthetic import (
    StratumSpec,
    compute_ground_truth,
    create_paper_gaussian_stream,
    create_paper_poisson_stream,
    create_paper_skewed_gaussian_stream,
    create_paper_skewed_poisson_stream,
    generate_gaussian_stream,
    generate_multi_strata_stream,
    generate_poisson_stream,
    generate_uniform_stream,
)


class TestSyntheticStreamGenerators:
    """Validate generators yield expected counts, columns, and distributions."""

    def test_uniform_stream(self):
        records = list(generate_uniform_stream(n_records=500, low=10.0, high=20.0, seed=42))
        assert len(records) == 500
        vals = [r["value"] for r in records]
        assert all(10.0 <= v <= 20.0 for v in vals)
        assert pytest.approx(np.mean(vals), abs=0.5) == 15.0

    def test_gaussian_stream(self):
        records = list(generate_gaussian_stream(n_records=1000, mean=50.0, std=5.0, seed=42))
        assert len(records) == 1000
        vals = [r["value"] for r in records]
        assert pytest.approx(np.mean(vals), abs=0.5) == 50.0
        assert pytest.approx(np.std(vals), abs=0.5) == 5.0

    def test_poisson_stream(self):
        records = list(generate_poisson_stream(n_records=1000, lam=25.0, seed=42))
        assert len(records) == 1000
        vals = [r["value"] for r in records]
        assert pytest.approx(np.mean(vals), abs=0.8) == 25.0

    def test_multi_strata_stream_generation(self):
        specs = [
            StratumSpec("A", "gaussian", {"mean": 10.0, "std": 1.0}, weight=3.0),
            StratumSpec("B", "uniform", {"low": 0.0, "high": 10.0}, weight=1.0),
        ]
        records = list(generate_multi_strata_stream(specs, total_records=1000, seed=42))
        assert len(records) == 1000
        gt = compute_ground_truth(records)
        # Expected ratio A:B is roughly 3:1 (750:250)
        assert gt["stratum_counts"]["A"] > 680
        assert gt["stratum_counts"]["B"] < 320

    def test_paper_gaussian_stream_preset(self):
        records = list(create_paper_gaussian_stream(total_records=2000, seed=42))
        assert len(records) == 2000
        gt = compute_ground_truth(records)
        assert set(gt["stratum_counts"].keys()) == {"A", "B", "C"}
        # A should dominate (~80%), B (~19%), C (~1%)
        assert gt["stratum_counts"]["A"] > gt["stratum_counts"]["B"] > gt["stratum_counts"]["C"]

    def test_paper_poisson_stream_preset(self):
        records = list(create_paper_poisson_stream(total_records=1000, seed=42))
        assert len(records) == 1000
        gt = compute_ground_truth(records)
        assert set(gt["stratum_counts"].keys()) == {"A", "B", "C"}

    def test_paper_skewed_presets(self):
        records_g = list(create_paper_skewed_gaussian_stream(total_records=1000, seed=42))
        records_p = list(create_paper_skewed_poisson_stream(total_records=1000, seed=42))
        assert len(records_g) == 1000
        assert len(records_p) == 1000

    def test_compute_ground_truth_empty_stream(self):
        gt = compute_ground_truth([])
        assert gt["count"] == 0
        assert gt["sum"] == 0.0
        assert gt["mean"] == 0.0
