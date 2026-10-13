"""End-to-end example demonstrating the StreamApprox baseline OASRS on synthetic streams."""

from streamapprox.aggregators import (
    approximate_count,
    approximate_mean,
    approximate_sum,
)
from streamapprox.estimators import (
    calculate_relative_error,
    estimate_mean,
    estimate_sum,
)
from streamapprox.oasrs import OASRS
from streamapprox.synthetic import (
    compute_ground_truth,
    create_paper_gaussian_stream,
)


def main():
    print("=" * 80)
    print("StreamApprox Baseline Reproduction: End-to-End Execution")
    print("Base Paper: Middleware '17 (Do Le Quoc et al.)")
    print("=" * 80)

    # 1. Generate Synthetic Stream
    total_records = 20000
    sample_size_per_stratum = 200
    print(f"\n[1] Generating synthetic stream of {total_records} records...")
    print("    Configuration: Heterogeneous Gaussian stream (Paper §5.1, §5.4)")
    print("    Strata: A (mu=10, sigma=5, rate=8000), B (mu=1000, sigma=50, rate=2000), C (mu=10000, sigma=500, rate=100)")

    records = list(create_paper_gaussian_stream(total_records=total_records, seed=42))
    gt = compute_ground_truth(records)

    print(f"    -> Generated {len(records)} records.")
    for sid, cnt in sorted(gt["stratum_counts"].items()):
        print(f"       Stratum {sid}: {cnt} items ({cnt/total_records*100:.1f}%), true mean = {gt['stratum_means'][sid]:.2f}")

    # 2. Ingest stream online via OASRS
    print(f"\n[2] Ingesting stream online via OASRS (reservoir capacity N = {sample_size_per_stratum})...")
    sampler = OASRS[dict](
        sample_size=sample_size_per_stratum,
        stratum_key="sensor_id",
        seed=100,
    )
    sampler.update_batch(records)

    # 3. Take snapshot
    snap = sampler.snapshot()
    print(f"    -> Processed {snap.total_seen} items across {snap.num_strata} strata.")
    print(f"    -> Retained sample size: {snap.sample_size} / {snap.total_seen} ({snap.sample_size / snap.total_seen * 100:.2f}%)")
    print("    Stratum states & sampling weights (Equation 1):")
    for sid, s in sorted(snap.strata.items()):
        print(f"       Stratum {sid}: C_{sid}={s.total_seen}, Y_{sid}={s.sample_size}, N_{sid}={s.capacity}, W_{sid}={s.weight:.4f}")

    # 4. Approximate Aggregation & Error Bounds
    print("\n[3] Computing Approximate Queries and Rigorous Error Bounds...")
    est_sum = estimate_sum(snap, value_key="value", confidence_level=0.95)
    est_mean = estimate_mean(snap, value_key="value", confidence_level=0.95)

    sum_rel_err = calculate_relative_error(est_sum.estimate, gt["sum"])
    mean_rel_err = calculate_relative_error(est_mean.estimate, gt["mean"])

    print("\n" + "-" * 80)
    print(f"{'Metric':<15} | {'Exact Ground Truth':<18} | {'OASRS Estimate':<18} | {'95% Error Bound':<16} | {'Accuracy Loss':<10}")
    print("-" * 80)
    print(f"{'SUM':<15} | {gt['sum']:<18.2f} | {est_sum.estimate:<18.2f} | +/- {est_sum.error_bound:<12.2f} | {sum_rel_err * 100:.2f}%")
    print(f"{'MEAN':<15} | {gt['mean']:<18.2f} | {est_mean.estimate:<18.2f} | +/- {est_mean.error_bound:<12.2f} | {mean_rel_err * 100:.2f}%")
    print(f"{'COUNT':<15} | {gt['count']:<18} | {approximate_count(snap):<18.0f} | {'N/A':<16} | 0.00%")
    print("-" * 80)

    # 5. Confidence Interval Check
    ci_sum = est_sum.confidence_interval
    ci_mean = est_mean.confidence_interval
    sum_in_ci = ci_sum[0] <= gt["sum"] <= ci_sum[1]
    mean_in_ci = ci_mean[0] <= gt["mean"] <= ci_mean[1]

    print(f"\n[4] Statistical Error Bound Verification:")
    print(f"    Sum 95% CI:  [{ci_sum[0]:.2f}, {ci_sum[1]:.2f}] -> True value inside: {sum_in_ci}")
    print(f"    Mean 95% CI: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}] -> True value inside: {mean_in_ci}")
    print("=" * 80)


if __name__ == "__main__":
    main()
