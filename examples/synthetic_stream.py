"""Example script: Demonstrating synthetic stream generation for StreamApprox."""

from streamapprox.synthetic import (
    compute_ground_truth,
    create_paper_gaussian_stream,
    create_paper_poisson_stream,
    create_paper_skewed_gaussian_stream,
    create_paper_skewed_poisson_stream,
)


def main():
    print("=" * 70)
    print("StreamApprox: Synthetic Data Stream Generator Demonstrations")
    print("=" * 70)

    # 1. Paper Gaussian baseline stream (§5.1)
    print("\n1. Paper Gaussian Stream (A: 8000/s, B: 2000/s, C: 100/s)")
    records_gaussian = list(create_paper_gaussian_stream(total_records=10000, seed=42))
    gt_gauss = compute_ground_truth(records_gaussian)
    print(f"   Total records generated: {gt_gauss['count']}")
    print(f"   Exact Mean: {gt_gauss['mean']:.4f}")
    print("   Stratum record counts:")
    for sid, cnt in gt_gauss["stratum_counts"].items():
        print(f"     Stratum {sid}: {cnt} items ({cnt / gt_gauss['count'] * 100:.1f}%), Mean: {gt_gauss['stratum_means'][sid]:.2f}")

    # 2. Paper Poisson baseline stream (§5.1)
    print("\n2. Paper Poisson Stream (A: lam=10, B: lam=1000, C: lam=10^8)")
    records_poisson = list(create_paper_poisson_stream(total_records=10000, seed=42))
    gt_poiss = compute_ground_truth(records_poisson)
    print(f"   Total records generated: {gt_poiss['count']}")
    print(f"   Exact Mean: {gt_poiss['mean']:.4f}")
    for sid, cnt in gt_poiss["stratum_counts"].items():
        print(f"     Stratum {sid}: {cnt} items ({cnt / gt_poiss['count'] * 100:.1f}%), Mean: {gt_poiss['stratum_means'][sid]:.2f}")

    # 3. Paper Skewed Poisson stream (§5.7: 80% : 19.99% : 0.01%)
    print("\n3. Paper Extreme Skew Poisson Stream (80% : 19.99% : 0.01%)")
    records_skewed = list(create_paper_skewed_poisson_stream(total_records=50000, seed=42))
    gt_skew = compute_ground_truth(records_skewed)
    print(f"   Total records generated: {gt_skew['count']}")
    for sid, cnt in sorted(gt_skew["stratum_counts"].items()):
        print(f"     Stratum {sid}: {cnt} items ({cnt / gt_skew['count'] * 100:.2f}%), Mean: {gt_skew['stratum_means'][sid]:.2f}")

    print("\nDemonstration complete.")


if __name__ == "__main__":
    main()
