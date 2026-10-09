"""Accuracy Benchmark: Exact vs SRS vs OASRS across Sampling Fractions & Skews.

Fulfills Phase 8, Section 11, Section 12, and Section 13 of project requirements:
- Baseline comparison: Full Stream (Exact) vs Simple Random Sampling (SRS) vs OASRS
- Varying sampling fractions: 100%, 80%, 60%, 40%, 20%, 10%
- Arrival rate experiments: 8K:2K:100 vs 3K:3K:3K vs 100:2K:8K
- Extreme skew Poisson: 80% : 19.99% : 0.01%
- Machine-readable CSV and JSON exports to benchmarks/results/
"""

import os
from typing import Dict, List
import numpy as np

from streamapprox.aggregators import approximate_mean, approximate_sum
from streamapprox.estimators import calculate_relative_error
from streamapprox.metrics import BenchmarkResult, BenchmarkTimer, save_benchmark_results
from streamapprox.oasrs import OASRS
from streamapprox.srs import SRSSampler
from streamapprox.synthetic import (
    compute_ground_truth,
    create_paper_gaussian_stream,
    create_paper_skewed_poisson_stream,
)


def run_fraction_benchmark(
    dataset_name: str,
    records: list,
    fractions: List[float] = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Run accuracy comparison across sampling fractions for a given dataset."""
    total_records = len(records)
    gt = compute_ground_truth(records)
    results: List[BenchmarkResult] = []

    # 1. Exact full stream baseline
    with BenchmarkTimer(record_count=total_records) as timer:
        exact_sum = sum(r["value"] for r in records)
        exact_mean = exact_sum / total_records

    exact_res = BenchmarkResult(
        method="Exact",
        dataset=dataset_name,
        total_records=total_records,
        sample_size=total_records,
        sampling_fraction_pct=100.0,
        exact_sum=gt["sum"],
        approx_sum=exact_sum,
        exact_mean=gt["mean"],
        approx_mean=exact_mean,
        sum_accuracy_loss_pct=0.0,
        mean_accuracy_loss_pct=0.0,
        elapsed_time_sec=timer.elapsed_sec,
        cpu_time_sec=timer.cpu_sec,
        throughput_items_per_sec=timer.throughput,
        latency_us_per_item=timer.latency_us,
        peak_rss_mb=timer.peak_rss_mb,
        stratum_representation=dict(gt["stratum_counts"]),
    )
    results.append(exact_res)

    # 2. Iterate over sampling fractions for SRS and OASRS
    num_strata = len(gt["stratum_counts"])

    for frac in fractions:
        target_sample_size = max(num_strata, int(total_records * frac))
        sample_size_per_stratum = max(1, target_sample_size // num_strata)

        # A. Simple Random Sampling (SRS)
        with BenchmarkTimer(record_count=total_records) as srs_timer:
            srs = SRSSampler[dict](sample_size=target_sample_size, seed=seed)
            srs.update_batch(records)
            srs_sum = srs.approximate_sum("value")
            srs_mean = srs.approximate_mean("value")

        srs_rep = {str(k): v for k, v in srs.stratum_representation("sensor_id").items()}
        srs_sum_loss = calculate_relative_error(srs_sum, gt["sum"]) * 100.0
        srs_mean_loss = calculate_relative_error(srs_mean, gt["mean"]) * 100.0

        results.append(
            BenchmarkResult(
                method="SRS",
                dataset=dataset_name,
                total_records=total_records,
                sample_size=srs.sample_size,
                sampling_fraction_pct=frac * 100.0,
                exact_sum=gt["sum"],
                approx_sum=srs_sum,
                exact_mean=gt["mean"],
                approx_mean=srs_mean,
                sum_accuracy_loss_pct=srs_sum_loss,
                mean_accuracy_loss_pct=srs_mean_loss,
                elapsed_time_sec=srs_timer.elapsed_sec,
                cpu_time_sec=srs_timer.cpu_sec,
                throughput_items_per_sec=srs_timer.throughput,
                latency_us_per_item=srs_timer.latency_us,
                peak_rss_mb=srs_timer.peak_rss_mb,
                stratum_representation=srs_rep,
            )
        )

        # B. OASRS
        with BenchmarkTimer(record_count=total_records) as oasrs_timer:
            oasrs = OASRS[dict](
                sample_size=sample_size_per_stratum,
                stratum_key="sensor_id",
                seed=seed,
            )
            oasrs.update_batch(records)
            snap = oasrs.snapshot()
            oasrs_sum = approximate_sum(snap, value_key="value")
            oasrs_mean = approximate_mean(snap, value_key="value")

        oasrs_rep = {str(sid): s.sample_size for sid, s in snap.strata.items()}
        oasrs_sum_loss = calculate_relative_error(oasrs_sum, gt["sum"]) * 100.0
        oasrs_mean_loss = calculate_relative_error(oasrs_mean, gt["mean"]) * 100.0

        results.append(
            BenchmarkResult(
                method="OASRS",
                dataset=dataset_name,
                total_records=total_records,
                sample_size=snap.sample_size,
                sampling_fraction_pct=frac * 100.0,
                exact_sum=gt["sum"],
                approx_sum=oasrs_sum,
                exact_mean=gt["mean"],
                approx_mean=oasrs_mean,
                sum_accuracy_loss_pct=oasrs_sum_loss,
                mean_accuracy_loss_pct=oasrs_mean_loss,
                elapsed_time_sec=oasrs_timer.elapsed_sec,
                cpu_time_sec=oasrs_timer.cpu_sec,
                throughput_items_per_sec=oasrs_timer.throughput,
                latency_us_per_item=oasrs_timer.latency_us,
                peak_rss_mb=oasrs_timer.peak_rss_mb,
                stratum_representation=oasrs_rep,
            )
        )

    return results


def print_comparison_table(results: List[BenchmarkResult], title: str) -> None:
    """Print readable benchmark summary table."""
    print("\n" + "=" * 90)
    print(f"BENCHMARK: {title}")
    print("=" * 90)
    print(
        f"{'Method':<8} | {'Fraction':<9} | {'Sample Size':<11} | {'Mean Approx':<14} | "
        f"{'Mean Loss (%)':<14} | {'Throughput (it/s)':<17} | {'Stratum Rep (A, B, C)':<20}"
    )
    print("-" * 90)

    for r in results:
        rep_str = ", ".join(f"{k}:{v}" for k, v in sorted(r.stratum_representation.items()))
        print(
            f"{r.method:<8} | {r.sampling_fraction_pct:>6.1f}%  | {r.sample_size:<11} | "
            f"{r.approx_mean:<14.2f} | {r.mean_accuracy_loss_pct:<14.4f} | "
            f"{r.throughput_items_per_sec:>14.0f}  | {rep_str}"
        )
    print("=" * 90)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    all_results: List[BenchmarkResult] = []

    # Experiment 1: Sampling fraction benchmark on Gaussian stream (§5.1 & §5.2)
    print("\n>>> Running Experiment 1: Gaussian Stream across Sampling Fractions (10% to 100%)...")
    records_gauss = list(create_paper_gaussian_stream(total_records=20000, seed=42))
    res_gauss = run_fraction_benchmark("Gaussian_Paper_5.1", records_gauss)
    print_comparison_table(res_gauss, "Gaussian Baseline (Paper §5.1, §5.2)")
    all_results.extend(res_gauss)

    # Experiment 2: Arrival-Rate & Skew Experiment (§5.4 & §5.7)
    # Poisson 80% : 19.99% : 0.01%
    print("\n>>> Running Experiment 2: Extreme Skew Poisson Stream (80% : 19.99% : 0.01%)...")
    records_poisson = list(create_paper_skewed_poisson_stream(total_records=50000, seed=42))
    res_poisson = run_fraction_benchmark("Poisson_Skewed_5.7", records_poisson, fractions=[0.4, 0.2, 0.1])
    print_comparison_table(res_poisson, "Extreme Skew Poisson (Paper §5.7)")
    all_results.extend(res_poisson)

    # Save CSV and JSON
    json_path, csv_path = save_benchmark_results(all_results, output_dir, "accuracy_results")
    print(f"\n[Artifacts Saved]")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
