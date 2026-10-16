"""Evaluate natural Intel mote_id strata; ground truth is offline only."""

import argparse
import csv
import gzip
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from streamapprox.allocation import allocate_capacities
from streamapprox.aggregators import approximate_mean
from streamapprox.estimators import estimate_mean
from streamapprox.oasrs import OASRS
from streamapprox.srs import SRSSampler
from streamapprox.va_oasrs import VAOASRS
from streamapprox.dv_oasrs import DVOASRS

FIELDS = ("temperature", "humidity", "light", "voltage")


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        position = 0
        for line in stream:
            parts = line.split()
            if len(parts) != 8:
                continue
            try:
                mote_id = int(parts[3])
                if not 1 <= mote_id <= 54:
                    continue
                yield {"mote_id": mote_id, "temperature": float(parts[4]),
                       "humidity": float(parts[5]), "light": float(parts[6]),
                       "voltage": float(parts[7]), "timestamp": f"{parts[0]} {parts[1]}",
                       "position": position}
                position += 1
            except ValueError:
                continue


def load_and_truth(path):
    rows = list(records(path))
    counts = defaultdict(int)
    sums = {field: 0.0 for field in FIELDS}
    for row in rows:
        counts[row["mote_id"]] += 1
        for field in FIELDS:
            sums[field] += row[field]
    return rows, counts, {field: sums[field] / len(rows) for field in FIELDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=ROOT / "data/data.txt")
    parser.add_argument("--budgets", type=int, nargs="+", default=[270, 540, 1080, 2160])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--interval", type=int, default=100000)
    parser.add_argument("--dynamic-interval", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/va_oasrs/intel")
    args = parser.parse_args()
    rows, counts, truth = load_and_truth(args.file)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = []
    allocation_rows = []
    sample_count_rows = []
    for budget in args.budgets:
        if budget < len(counts):
            continue
        equal = allocate_capacities(budget, {sid: 1.0 for sid in counts}, minimum=1)
        for run in range(args.runs):
            seed = 100 + run
            methods = []
            sampler = OASRS(1, stratum_key="mote_id", seed=seed,
                            stratum_capacity=lambda sid: equal[sid])
            start = time.perf_counter(); sampler.update_batch(rows); elapsed = time.perf_counter()-start
            methods.append(("OASRS-equal", sampler.snapshot(), elapsed, sampler.sample_size))
            va = VAOASRS(budget, "mote_id", "temperature", seed=seed,
                         reallocation_interval=args.interval, burn_in_records=args.interval)
            start = time.perf_counter(); va.update_batch(rows); snap = va.finalize(); elapsed = time.perf_counter()-start
            methods.append(("VA-OASRS", snap, elapsed, va.sample_size))
            dv = DVOASRS(budget, "mote_id", "temperature", seed=seed,
                         reallocation_interval=args.dynamic_interval, variance_alpha=0.01)
            start = time.perf_counter(); dv.update_batch(rows); snap = dv.finalize(); elapsed = time.perf_counter()-start
            methods.append(("DV-OASRS", snap, elapsed, dv.sample_size))
            for sid, state in dv.strata.items():
                sample_count_rows.append({"seed": seed, "budget": budget, "mote_id": sid,
                    "population": state.count, "capacity": state.capacity,
                    "sample_count": min(state.count, state.capacity, len(state.priority_heap))})
            for event_id, event in enumerate(dv.allocation_history):
                for sid, capacity in event["capacities"].items():
                    allocation_rows.append({"seed":seed,"budget":budget,"event_id":event_id,
                        "records_seen":event["records_seen"],"mote_id":sid,"capacity":capacity,
                        "score":event["scores"].get(sid, ""),"ema_variance":event["ema_variances"].get(sid, "")})
            srs = SRSSampler(budget, seed=seed)
            start = time.perf_counter(); srs.update_batch(rows); elapsed = time.perf_counter()-start
            methods.append(("Uniform Reservoir", srs, elapsed, srs.sample_size))
            for method, snap, elapsed, sampled in methods:
                for field in FIELDS:
                    if method == "Uniform Reservoir":
                        estimate = sum(row[field] for row in snap.samples) / snap.sample_size
                        from streamapprox.estimators import _resolve_z_score
                        values = [float(row[field]) for row in snap.samples]
                        sample_variance = statistics.variance(values) if len(values) > 1 else 0.0
                        se = (sample_variance / len(values) * (snap.total_seen-len(values)) / max(1,snap.total_seen)) ** 0.5
                    else:
                        estimate = approximate_mean(snap, field)
                        se = estimate_mean(snap, field).standard_error
                    error = abs(estimate - truth[field])
                    raw.append({"seed":seed,"budget":budget,"method":method,"query":field+"_mean",
                                "exact":truth[field],"estimate":estimate,"absolute_error":error,
                                "relative_error":error/abs(truth[field]) if truth[field] else error,
                                "estimated_se":se,"ci95_contains_exact":error<=1.96*se,
                                "sampled":sampled,"records":len(rows),"elapsed_seconds":elapsed,
                                "throughput_records_per_second":len(rows)/elapsed,
                                "allocation_events":len(dv.allocation_history) if method=="DV-OASRS" else "",
                                "allocation_seconds":dv.allocation_seconds if method=="DV-OASRS" else "",
                                "priority_seconds":dv.priority_seconds if method=="DV-OASRS" else "",
                                "variance_seconds":dv.variance_seconds if method=="DV-OASRS" else "",
                                "new_stratum_first_sample_positions":json.dumps(dv.late_stratum_first_sample) if method=="DV-OASRS" else ""})
    with (args.output/"raw_results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    groups=defaultdict(list)
    for row in raw: groups[(row["budget"],row["method"],row["query"])].append(row)
    summary=[]
    for (budget,method,query),values in sorted(groups.items()):
        errs=[x["relative_error"] for x in values]
        summary.append({"budget":budget,"method":method,"query":query,"runs":len(errs),
                        "mean_relative_error":statistics.fmean(errs),
                        "sd_relative_error":statistics.stdev(errs) if len(errs)>1 else 0,
                        "ci95_coverage":statistics.fmean(float(x["ci95_contains_exact"]) for x in values),
                        "mean_throughput_records_per_second":statistics.fmean(x["throughput_records_per_second"] for x in values),
                        "mean_sampled":statistics.fmean(x["sampled"] for x in values)})
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (args.output/"allocation_over_time.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for row in allocation_rows for k in row}));writer.writeheader();writer.writerows(allocation_rows)
    counts_path=args.output/"strata.csv"
    with counts_path.open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["mote_id","population"]);writer.writerows(sorted(counts.items()))
    with (args.output/"first_appearance.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["mote_id","first_position"])
        first={}
        for row in rows:first.setdefault(row["mote_id"],row["position"])
        writer.writerows(sorted(first.items(),key=lambda x:x[1]))
    with (args.output/"sample_counts.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(sample_count_rows[0]));writer.writeheader();writer.writerows(sample_count_rows)
    (args.output/"config.json").write_text(json.dumps({"file":str(args.file),"budgets":args.budgets,"runs":args.runs,"static_interval":args.interval,"dynamic_interval":args.dynamic_interval,"records":len(rows),"strata":len(counts)},indent=2))
    print(f"Intel evaluation: {len(rows)} records, {len(counts)} strata; wrote {args.output}")


if __name__ == "__main__": main()
