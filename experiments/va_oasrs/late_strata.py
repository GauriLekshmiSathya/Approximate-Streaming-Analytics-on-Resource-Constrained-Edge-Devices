"""Late-source stress test with records and allocations saved as CSV."""

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from streamapprox.allocation import allocate_capacities
from streamapprox.aggregators import approximate_mean
from streamapprox.dv_oasrs import DVOASRS
from streamapprox.oasrs import OASRS
from streamapprox.va_oasrs import VAOASRS


def stream(total: int, seed: int):
    rng = random.Random(seed)
    milestones = ((0, "S1"), (0, "S2"), (0, "S3"), (0, "S4"),
                  (total // 4, "S5"), (total // 2, "S6"))
    for position in range(total):
        active = [sid for onset, sid in milestones if position >= onset]
        sid = rng.choice(active)
        scale = {"S1": 1.0, "S2": 2.0, "S3": 4.0, "S4": 1.0, "S5": 15.0, "S6": 25.0}[sid]
        yield {"sensor_id": sid, "value": 10.0 + scale * rng.gauss(0, 1), "position": position}


def exact_rows(rows):
    n = len(rows)
    return sum(x["value"] for x in rows) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=400_000)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23, 37, 53])
    parser.add_argument("--interval", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/va_oasrs/late_strata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = []
    allocation_rows = []
    first = {}
    for seed in args.seeds:
        records = list(stream(args.records, seed))
        truth = exact_rows(records)
        keys = sorted({r["sensor_id"] for r in records})
        for row in records:
            first.setdefault(row["sensor_id"], row["position"])
        equal = allocate_capacities(args.budget, {k: 1.0 for k in keys}, minimum=1)
        oasrs = OASRS(1, stratum_key="sensor_id", seed=seed,
                      stratum_capacity=lambda sid: equal[sid])
        static = VAOASRS(args.budget, "sensor_id", "value", seed=seed,
                         reallocation_interval=args.interval, burn_in_records=args.interval)
        dynamic = DVOASRS(args.budget, "sensor_id", "value", seed=seed,
                          reallocation_interval=args.interval, variance_alpha=0.02)
        outputs = {}
        timings = {}
        for name, sampler in (("OASRS", oasrs), ("Static VA-OASRS", static), ("DV-OASRS", dynamic)):
            started = time.perf_counter()
            sampler.update_batch(records)
            snapshot = sampler.finalize() if hasattr(sampler, "finalize") else sampler.snapshot()
            timings[name] = time.perf_counter() - started
            outputs[name] = snapshot
        for event_sampler_name, sampler in (("Static VA-OASRS", static), ("DV-OASRS", dynamic)):
            for event in sampler.allocation_history:
                for sid in sorted(set(event.get("capacities", {})) | set(event.get("scores", {})), key=str):
                    allocation_rows.append({"seed":seed,"method":event_sampler_name,
                        "records_seen":event["records_seen"],"stratum":sid,
                        "capacity":event.get("capacities",{}).get(sid,0),
                        "population":event.get("counts",{}).get(sid,""),
                        "ema_variance":event.get("ema_variances",{}).get(sid,""),
                        "score":event.get("scores",{}).get(sid,""),
                        "sampled":event.get("sampled","")})
        for name, sampler in (("OASRS", oasrs), ("Static VA-OASRS", static), ("DV-OASRS", dynamic)):
            snap = outputs[name]
            estimate = approximate_mean(snap, "value")
            first_sample = getattr(sampler, "late_stratum_first_sample", {})
            cap5 = snap.strata["S5"].capacity if "S5" in snap.strata else 0
            cap6 = snap.strata["S6"].capacity if "S6" in snap.strata else 0
            raw.append({"seed":seed,"method":name,"exact_mean":truth,"estimated_mean":estimate,
                "absolute_error":abs(estimate-truth),"relative_error":abs(estimate-truth)/abs(truth),
                "budget":args.budget,"sampled":snap.sample_size,
                "throughput_records_per_second":len(records)/timings[name],
                "allocation_events":len(getattr(sampler,"allocation_history",[])),
                "late_S5_capacity":cap5,"late_S5_first_sample_position":first_sample.get("S5",""),
                "late_S6_capacity":cap6,"late_S6_first_sample_position":first_sample.get("S6","")})

    with (args.output / "first_appearance.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["stratum", "first_position"])
        writer.writerows(sorted(first.items(), key=lambda x: x[1]))
    with (args.output / "allocation_over_time.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(allocation_rows[0])); writer.writeheader(); writer.writerows(allocation_rows)
    with (args.output / "raw_results.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    with (args.output / "summary.csv").open("w", newline="") as f:
        grouped = {}
        for row in raw: grouped.setdefault(row["method"], []).append(row)
        fields = ["method","runs","mean_relative_error","sd_relative_error","ci95_low","ci95_high",
                  "mean_sampled","mean_allocation_events","mean_late_S5_capacity","mean_late_S5_first_sample_position",
                  "mean_late_S6_capacity","mean_late_S6_first_sample_position"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,values in sorted(grouped.items()):
            errs=[float(x["relative_error"]) for x in values]; mean=statistics.fmean(errs)
            sd=statistics.stdev(errs) if len(errs)>1 else 0.0
            half=2.776*sd/(len(errs)**0.5) if len(errs)==5 else 0.0
            numeric=lambda key: [float(x[key]) for x in values if x[key] != ""]
            writer.writerow({"method":method,"runs":len(values),"mean_relative_error":mean,
                "sd_relative_error":sd,"ci95_low":max(0,mean-half),"ci95_high":mean+half,
                "mean_sampled":statistics.fmean(float(x["sampled"]) for x in values),
                "mean_allocation_events":statistics.fmean(float(x["allocation_events"]) for x in values),
                "mean_late_S5_capacity":statistics.fmean(numeric("late_S5_capacity")),
                "mean_late_S5_first_sample_position":statistics.fmean(numeric("late_S5_first_sample_position")) if numeric("late_S5_first_sample_position") else "",
                "mean_late_S6_capacity":statistics.fmean(numeric("late_S6_capacity")),
                "mean_late_S6_first_sample_position":statistics.fmean(numeric("late_S6_first_sample_position")) if numeric("late_S6_first_sample_position") else ""})
    (args.output / "config.json").write_text(json.dumps(vars(args) | {"strata": ["S1","S2","S3","S4","S5","S6"]}, default=str, indent=2))
    print(f"saved five-seed late-strata experiment to {args.output}")


if __name__ == "__main__":
    main()
