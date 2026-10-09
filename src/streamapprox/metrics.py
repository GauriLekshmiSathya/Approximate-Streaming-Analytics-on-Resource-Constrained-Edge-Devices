"""Performance and Resource Measurement Layer for StreamApprox.

Fulfills Phase 9, Section 11, and Section 14 of project requirements:
- Elapsed time and CPU utilization
- Throughput (records / sec) and per-record latency (microseconds)
- Peak RSS memory tracking via resource module
- Accuracy loss: |approx - exact| / |exact| (safely handled when exact == 0)
- Machine-readable serialization to Dict, CSV, and JSON
"""

from dataclasses import asdict, dataclass, field
import json
import os
import resource
import time
from typing import Any, Callable, Dict, List, Optional


def get_peak_rss_mb() -> float:
    """Return peak Resident Set Size (RSS) memory in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux, ru_maxrss is reported in kilobytes
    return float(usage.ru_maxrss / 1024.0)


@dataclass
class BenchmarkResult:
    """Structured benchmark run outcome."""

    method: str
    dataset: str
    total_records: int
    sample_size: int
    sampling_fraction_pct: float
    exact_sum: float
    approx_sum: float
    exact_mean: float
    approx_mean: float
    sum_accuracy_loss_pct: float
    mean_accuracy_loss_pct: float
    elapsed_time_sec: float
    cpu_time_sec: float
    throughput_items_per_sec: float
    latency_us_per_item: float
    peak_rss_mb: float
    stratum_representation: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return asdict(self)

    @classmethod
    def csv_header(cls) -> str:
        """Header line for CSV export."""
        return (
            "method,dataset,total_records,sample_size,sampling_fraction_pct,"
            "exact_sum,approx_sum,exact_mean,approx_mean,"
            "sum_accuracy_loss_pct,mean_accuracy_loss_pct,"
            "elapsed_time_sec,cpu_time_sec,throughput_items_per_sec,"
            "latency_us_per_item,peak_rss_mb"
        )

    def to_csv_row(self) -> str:
        """Serialize as a single CSV row."""
        return (
            f"{self.method},{self.dataset},{self.total_records},{self.sample_size},"
            f"{self.sampling_fraction_pct:.2f},{self.exact_sum:.4f},{self.approx_sum:.4f},"
            f"{self.exact_mean:.4f},{self.approx_mean:.4f},"
            f"{self.sum_accuracy_loss_pct:.4f},{self.mean_accuracy_loss_pct:.4f},"
            f"{self.elapsed_time_sec:.6f},{self.cpu_time_sec:.6f},"
            f"{self.throughput_items_per_sec:.2f},{self.latency_us_per_item:.4f},"
            f"{self.peak_rss_mb:.2f}"
        )


class BenchmarkTimer:
    """High-precision context manager measuring wall time, CPU time, and peak memory."""

    def __init__(self, record_count: int = 0) -> None:
        self.record_count = record_count
        self.start_wall: float = 0.0
        self.start_cpu: float = 0.0
        self.elapsed_sec: float = 0.0
        self.cpu_sec: float = 0.0
        self.peak_rss_mb: float = 0.0

    def __enter__(self) -> "BenchmarkTimer":
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_sec = max(1e-9, time.perf_counter() - self.start_wall)
        self.cpu_sec = max(0.0, time.process_time() - self.start_cpu)
        self.peak_rss_mb = get_peak_rss_mb()

    @property
    def throughput(self) -> float:
        """Items processed per second."""
        if self.elapsed_sec <= 0:
            return 0.0
        return self.record_count / self.elapsed_sec

    @property
    def latency_us(self) -> float:
        """Microseconds per record."""
        if self.record_count <= 0:
            return 0.0
        return (self.elapsed_sec * 1e6) / self.record_count

    @property
    def cpu_utilization_pct(self) -> float:
        """CPU utilization percentage."""
        if self.elapsed_sec <= 0:
            return 0.0
        return (self.cpu_sec / self.elapsed_sec) * 100.0


def save_benchmark_results(
    results: List[BenchmarkResult], output_dir: str, prefix: str
) -> Tuple[str, str]:
    """Save benchmark results to both JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}.json")
    csv_path = os.path.join(output_dir, f"{prefix}.csv")

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Write CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(BenchmarkResult.csv_header() + "\n")
        for r in results:
            f.write(r.to_csv_row() + "\n")

    return json_path, csv_path
