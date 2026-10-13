"""Sampling Throughput and Latency Benchmark.

Fulfills Phase 9, Section 11, and Section 14 of project requirements:
- Measures throughput (items / second) and latency (microseconds / item)
- Evaluates full stream ingestion vs SRS vs OASRS
- Scales stream volume up to 200,000 records
- Saves JSON and CSV to benchmarks/results/
"""

import os
from typing import List

from streamapprox.metrics import BenchmarkResult, BenchmarkTimer, save_benchmark_results
from streamapprox.oasrs import OASRS
from streamapprox.srs import SRSSampler
from streamapprox.synthetic import create_paper_gaussian_stream


def run_throughput_benchmark(
    stream_sizes: List[int] = [10000, 50000, 100000, 200000],
    fractions: List[float] = [0.8, 0.4, 0.1],
    seed: int = 42,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for n in stream_sizes:
        print(f"Generating stream of {n} records...")
        records = list(create_paper_gaussian_stream(total_records=n, seed=seed))
        gt_sum = sum(r["value"] for r in records)
        gt_mean = gt_sum / n

        # 1. Native Full Processing (no sampling)
        with BenchmarkTimer(record_count=n) as native_timer:
            full_buffer = []
            for r in records:
                full_buffer.append(r)
            full_sum = sum(x["value"] for x in full_buffer)
            full_mean = full_sum / len(full_buffer)

        results.append(
            BenchmarkResult(
                method="Native_Full",
                dataset=f"Gaussian_{n}",
                total_records=n,
                sample_size=n,
                sampling_fraction_pct=100.0,
                exact_sum=gt_sum,
                approx_sum=full_sum,
                exact_mean=gt_mean,
                approx_mean=full_mean,
                sum_accuracy_loss_pct=0.0,
                mean_accuracy_loss_pct=0.0,
                elapsed_time_sec=native_timer.elapsed_sec,
                cpu_time_sec=native_timer.cpu_sec,
                throughput_items_per_sec=native_timer.throughput,
                latency_us_per_item=native_timer.latency_us,
                peak_rss_mb=native_timer.peak_rss_mb,
            )
        )

        # 2. SRS & OASRS across fractions
        for frac in fractions:
            sample_size = max(3, int(n * frac))
            per_stratum_cap = max(1, sample_size // 3)

            # SRS
            with BenchmarkTimer(record_count=n) as srs_timer:
                srs = SRSSampler[dict](sample_size=sample_size, seed=seed)
                srs.update_batch(records)
                srs_s = srs.approximate_sum("value")
                srs_m = srs.approximate_mean("value")

            results.append(
                BenchmarkResult(
                    method=f"SRS_{int(frac*100)}%",
                    dataset=f"Gaussian_{n}",
                    total_records=n,
                    sample_size=srs.sample_size,
                    sampling_fraction_pct=frac * 100.0,
                    exact_sum=gt_sum,
                    approx_sum=srs_s,
                    exact_mean=gt_mean,
                    approx_mean=srs_m,
                    sum_accuracy_loss_pct=abs(srs_s - gt_sum) / gt_sum * 100.0,
                    mean_accuracy_loss_pct=abs(srs_m - gt_mean) / gt_mean * 100.0,
                    elapsed_time_sec=srs_timer.elapsed_sec,
                    cpu_time_sec=srs_timer.cpu_sec,
                    throughput_items_per_sec=srs_timer.throughput,
                    latency_us_per_item=srs_timer.latency_us,
                    peak_rss_mb=srs_timer.peak_rss_mb,
                )
            )

            # OASRS
            with BenchmarkTimer(record_count=n) as oasrs_timer:
                oasrs = OASRS[dict](
                    sample_size=per_stratum_cap,
                    stratum_key="sensor_id",
                    seed=seed,
                )
                oasrs.update_batch(records)
                snap = oasrs.snapshot()
                o_s = sum(s.total_seen * (sum(x["value"] for x in s.samples) / s.sample_size if s.sample_size > 0 else 0) for s in snap)
                o_m = o_s / snap.total_seen if snap.total_seen > 0 else 0

            results.append(
                BenchmarkResult(
                    method=f"OASRS_{int(frac*100)}%",
                    dataset=f"Gaussian_{n}",
                    total_records=n,
                    sample_size=snap.sample_size,
                    sampling_fraction_pct=frac * 100.0,
                    exact_sum=gt_sum,
                    approx_sum=o_s,
                    exact_mean=gt_mean,
                    approx_mean=o_m,
                    sum_accuracy_loss_pct=abs(o_s - gt_sum) / gt_sum * 100.0,
                    mean_accuracy_loss_pct=abs(o_m - gt_mean) / gt_mean * 100.0,
                    elapsed_time_sec=oasrs_timer.elapsed_sec,
                    cpu_time_sec=oasrs_timer.cpu_sec,
                    throughput_items_per_sec=oasrs_timer.throughput,
                    latency_us_per_item=oasrs_timer.latency_us,
                    peak_rss_mb=oasrs_timer.peak_rss_mb,
                )
            )

    return results


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    print("=" * 80)
    print("StreamApprox: Sampling Throughput & Latency Benchmark")
    print("=" * 80)

    results = run_throughput_benchmark(stream_sizes=[10000, 50000, 100000])

    print("\n" + "=" * 80)
    print(f"{'Method':<14} | {'Stream Size':<12} | {'Throughput (it/s)':<18} | {'Latency (us/it)':<16} | {'Peak RSS (MB)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r.method:<14} | {r.total_records:<12} | {r.throughput_items_per_sec:>16.0f}  | {r.latency_us_per_item:>14.2f}  | {r.peak_rss_mb:>10.2f}")
    print("=" * 80)

    json_path, csv_path = save_benchmark_results(results, output_dir, "sampling_throughput")
    print(f"\n[Artifacts Saved]")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
