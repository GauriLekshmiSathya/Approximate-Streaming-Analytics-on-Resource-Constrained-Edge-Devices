"""Resource Measurement Benchmark: CPU Utilization, Peak RSS, and Memory Scaling.

Fulfills Phase 9 and Section 14 of project requirements:
- Measures memory scaling: Unbounded Full Stream O(C) vs OASRS O(sum N_i)
- Measures process CPU utilization and execution latency
- Saves JSON and CSV to benchmarks/results/
"""

import os
from typing import List

from streamapprox.metrics import BenchmarkResult, BenchmarkTimer, save_benchmark_results
from streamapprox.oasrs import OASRS
from streamapprox.synthetic import create_paper_gaussian_stream


def run_resource_benchmark(
    stream_sizes: List[int] = [10000, 50000, 100000, 200000],
    sample_size_per_stratum: int = 200,
    seed: int = 42,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for n in stream_sizes:
        print(f"Profiling resources on stream of {n} records...")
        records = list(create_paper_gaussian_stream(total_records=n, seed=seed))
        gt_sum = sum(r["value"] for r in records)
        gt_mean = gt_sum / n

        # A. Full Stream Storage (Unbounded buffer)
        with BenchmarkTimer(record_count=n) as full_timer:
            full_buffer = [r for r in records]
            s_full = sum(r["value"] for r in full_buffer)
            m_full = s_full / len(full_buffer)

        results.append(
            BenchmarkResult(
                method="Full_Buffer",
                dataset=f"Scale_{n}",
                total_records=n,
                sample_size=len(full_buffer),
                sampling_fraction_pct=100.0,
                exact_sum=gt_sum,
                approx_sum=s_full,
                exact_mean=gt_mean,
                approx_mean=m_full,
                sum_accuracy_loss_pct=0.0,
                mean_accuracy_loss_pct=0.0,
                elapsed_time_sec=full_timer.elapsed_sec,
                cpu_time_sec=full_timer.cpu_sec,
                throughput_items_per_sec=full_timer.throughput,
                latency_us_per_item=full_timer.latency_us,
                peak_rss_mb=full_timer.peak_rss_mb,
            )
        )

        # B. OASRS Bounded Reservoir
        with BenchmarkTimer(record_count=n) as oasrs_timer:
            sampler = OASRS[dict](
                sample_size=sample_size_per_stratum,
                stratum_key="sensor_id",
                seed=seed,
            )
            sampler.update_batch(records)
            snap = sampler.snapshot()
            o_s = sum(s.total_seen * (sum(x["value"] for x in s.samples) / s.sample_size if s.sample_size > 0 else 0) for s in snap)
            o_m = o_s / snap.total_seen if snap.total_seen > 0 else 0

        results.append(
            BenchmarkResult(
                method="OASRS_Bounded",
                dataset=f"Scale_{n}",
                total_records=n,
                sample_size=snap.sample_size,
                sampling_fraction_pct=(snap.sample_size / n) * 100.0,
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
    print("StreamApprox: Resource & Memory Profiling Benchmark")
    print("=" * 80)

    results = run_resource_benchmark(stream_sizes=[10000, 50000, 100000, 200000])

    print("\n" + "=" * 90)
    print(
        f"{'Method':<15} | {'Stream Size':<12} | {'Retained Items':<15} | "
        f"{'Peak RSS (MB)':<14} | {'Throughput (it/s)':<18} | {'Latency (us)':<12}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.method:<15} | {r.total_records:<12} | {r.sample_size:<15} | "
            f"{r.peak_rss_mb:>12.2f}  | {r.throughput_items_per_sec:>16.0f}  | {r.latency_us_per_item:>10.2f}"
        )
    print("=" * 90)

    json_path, csv_path = save_benchmark_results(results, output_dir, "resource_results")
    print(f"\n[Artifacts Saved]")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
