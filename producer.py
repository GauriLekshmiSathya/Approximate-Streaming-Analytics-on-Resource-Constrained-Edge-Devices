"""Replay the Intel Berkeley sensor dataset into Kafka as a live stream.

Examples
--------
  # 2000 events/s, whole dataset
  python producer.py --file data/data.txt.gz --schedule 2000

  # burst pattern for adaptation experiments: 2000 ev/s for 30 s, 10000 ev/s for 10 s, repeat
  python producer.py --file data/data.txt.gz --schedule "2000:30,10000:10"

  # as fast as possible, first 200k records (quick test)
  python producer.py --file data/data.txt.gz --schedule 0 --limit 200000

Each message:  key = mote_id (the stratum),  value = JSON record with a global
`seq` (0..N-1 over valid records) so the archive can be checked for completeness,
and `sent_at` (unix time) so ingest latency can be measured.
"""
import argparse
import gzip
import json
import time

import common


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="Intel Berkeley data.txt or data.txt.gz")
    ap.add_argument("--schedule", default="1000", help="events/s, or 'rate:sec,rate:sec,...' (0 = unthrottled)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N valid records (0 = all)")
    ap.add_argument("--topic", default=common.TOPIC_RAW)
    ap.add_argument("--bootstrap", default=common.BOOTSTRAP)
    ap.add_argument("--metrics-port", type=int, default=8000)
    args = ap.parse_args()

    from confluent_kafka import Producer
    from prometheus_client import Counter, Gauge, start_http_server

    sent = Counter("producer_messages_total", "Messages delivered to Kafka")
    sent_bytes = Counter("producer_bytes_total", "Bytes delivered to Kafka")
    errors = Counter("producer_errors_total", "Failed deliveries")
    dropped = Counter("producer_dropped_records_total", "Malformed source lines skipped")
    target_rate = Gauge("producer_target_rate", "Configured replay rate (events/s)")
    start_http_server(args.metrics_port)

    common.ensure_topics(args.bootstrap)
    producer = Producer({
        "bootstrap.servers": args.bootstrap,
        "linger.ms": 5,
        "batch.num.messages": 1000,
        "queue.buffering.max.messages": 200000,
        "acks": "all",
    })

    def on_delivery(err, msg):
        if err is not None:
            errors.inc()
        else:
            sent.inc()
            sent_bytes.inc(len(msg.value()))

    sched = common.RateSchedule(args.schedule)
    t0 = next_t = time.perf_counter()
    last_report, last_report_seq, seq, n_dropped = t0, 0, 0, 0

    with open_text(args.file) as f:
        for line in f:
            rec = common.parse_intel_line(line)
            if rec is None:
                n_dropped += 1
                dropped.inc()
                continue

            # ---- pacing (sleep only when we are >2 ms ahead, so high rates stay cheap)
            now = time.perf_counter()
            rate = sched.rate_at(now - t0)
            if rate > 0:
                if next_t < now - 0.5:      # fell behind (or rate just changed): don't burst to catch up
                    next_t = now
                next_t += 1.0 / rate
                delay = next_t - now
                if delay > 0.002:
                    time.sleep(delay)
            else:
                next_t = now

            rec["seq"] = seq
            rec["sent_at"] = time.time()
            payload = json.dumps(rec, separators=(",", ":")).encode()
            while True:
                try:
                    producer.produce(args.topic, key=str(rec["mote_id"]).encode(),
                                     value=payload, on_delivery=on_delivery)
                    break
                except BufferError:
                    producer.poll(0.1)
            producer.poll(0)
            seq += 1

            if now - last_report >= 2.0:
                target_rate.set(rate)
                achieved = (seq - last_report_seq) / (now - last_report)
                print(f"[producer] seq={seq:,}  target={rate:.0f}/s  achieved={achieved:.0f}/s  "
                      f"skipped={n_dropped:,}")
                last_report, last_report_seq = now, seq

            if args.limit and seq >= args.limit:
                break

    producer.flush()
    secs = time.perf_counter() - t0
    print(json.dumps({"sent": seq, "skipped_malformed": n_dropped, "seconds": round(secs, 1),
                      "avg_rate": round(seq / secs, 1)}))


if __name__ == "__main__":
    main()
