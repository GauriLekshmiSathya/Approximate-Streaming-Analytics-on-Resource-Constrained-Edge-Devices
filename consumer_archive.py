"""Archive the raw Kafka stream to date-partitioned Parquet (the ground truth).

  archive/raw/date=2004-02-28/part-00000000-00012345-ab12cd.parquet
  archive/raw/date=2004-02-29/...

Why: relative error / accuracy loss = |approx - exact| / exact needs the exact
answer over ALL items, so the unsampled stream is persisted here and the exact
value for any window can be recomputed offline.

Delivery: at-least-once. Offsets are committed only after a batch is safely on
disk, so a crash can cause duplicate rows, never missing ones. Duplicates are
removed on read by `seq` (verify_archive.py shows how).

Metrics (Prometheus, default :8001): consumer_messages_total, consumer_bytes_total,
archive_rows_total, archive_files_total, archive_flush_seconds, ingest_latency_seconds,
kafka_consumer_lag{partition}.
"""
import argparse
import json
import os
import signal
import time
import uuid

import pandas as pd

import common

COLUMNS = ["seq", "event_ts", "epoch", "mote_id", "temperature", "humidity", "light",
           "voltage", "sent_at", "ingested_at", "kafka_partition", "kafka_offset"]


def _schema():
    import pyarrow as pa
    return pa.schema([
        ("seq", pa.int64()),
        ("event_ts", pa.timestamp("us")),
        ("epoch", pa.int64()),
        ("mote_id", pa.int16()),
        ("temperature", pa.float64()),
        ("humidity", pa.float64()),
        ("light", pa.float64()),
        ("voltage", pa.float64()),
        ("sent_at", pa.float64()),
        ("ingested_at", pa.float64()),
        ("kafka_partition", pa.int32()),
        ("kafka_offset", pa.int64()),
    ])


def flush_rows(rows: list, out_dir: str) -> int:
    """Write buffered rows to Parquet, one file per event-date. Returns #files written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    df["_date"] = df["event_ts"].dt.strftime("%Y-%m-%d")
    schema = _schema()
    files = 0
    for date, g in df.groupby("_date"):
        part_dir = os.path.join(out_dir, f"date={date}")
        os.makedirs(part_dir, exist_ok=True)
        name = f"part-{int(g.seq.min()):08d}-{int(g.seq.max()):08d}-{uuid.uuid4().hex[:6]}.parquet"
        tmp = os.path.join(part_dir, "." + name + ".tmp")   # dot-prefixed: readers ignore it
        table = pa.Table.from_pandas(g.drop(columns="_date"), schema=schema, preserve_index=False)
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, os.path.join(part_dir, name))        # atomic publish
        files += 1
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default=common.TOPIC_RAW)
    ap.add_argument("--bootstrap", default=common.BOOTSTRAP)
    ap.add_argument("--group", default="raw-archiver")
    ap.add_argument("--out", default="archive/raw")
    ap.add_argument("--batch-size", type=int, default=50_000, help="rows per flush")
    ap.add_argument("--flush-interval", type=float, default=5.0, help="max seconds between flushes")
    ap.add_argument("--stop-after-idle", type=float, default=0,
                    help="exit after N seconds with no messages (0 = run forever)")
    ap.add_argument("--metrics-port", type=int, default=8001)
    args = ap.parse_args()

    from confluent_kafka import Consumer, KafkaError
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    c_msgs = Counter("consumer_messages_total", "Messages consumed")
    c_bytes = Counter("consumer_bytes_total", "Bytes consumed")
    c_rows = Counter("archive_rows_total", "Rows written to Parquet")
    c_files = Counter("archive_files_total", "Parquet files written")
    h_flush = Histogram("archive_flush_seconds", "Time to write one batch to disk")
    h_lat = Histogram("ingest_latency_seconds", "producer sent_at -> consumer receive",
                      buckets=(.001, .005, .01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30))
    g_lag = Gauge("kafka_consumer_lag", "Messages behind the log end", ["partition"])
    start_http_server(args.metrics_port)

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,      # we commit manually after the data is on disk
    })
    consumer.subscribe([args.topic])

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    buf, total = [], 0
    last_flush = last_msg = last_lag = time.time()
    print(f"[archiver] consuming '{args.topic}' -> {args.out}  (Ctrl+C to stop)")

    def flush():
        nonlocal buf, total, last_flush
        if not buf:
            return
        t = time.time()
        files = flush_rows(buf, args.out)
        consumer.commit(asynchronous=False)       # safe: everything consumed so far is on disk
        h_flush.observe(time.time() - t)
        c_rows.inc(len(buf))
        c_files.inc(files)
        total += len(buf)
        print(f"[archiver] wrote {len(buf):,} rows in {files} file(s)  total={total:,}")
        buf, last_flush = [], time.time()

    while running:
        msgs = consumer.consume(num_messages=1000, timeout=1.0)
        now = time.time()
        for m in msgs:
            if m.error():
                if m.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[archiver] kafka error: {m.error()}")
                continue
            raw = m.value()
            d = json.loads(raw)
            d["ingested_at"] = now
            d["kafka_partition"] = m.partition()
            d["kafka_offset"] = m.offset()
            buf.append(d)
            c_msgs.inc()
            c_bytes.inc(len(raw))
            h_lat.observe(max(0.0, now - d["sent_at"]))
        if msgs:
            last_msg = now

        if len(buf) >= args.batch_size or (buf and now - last_flush >= args.flush_interval):
            flush()

        if now - last_lag >= 5:                    # consumer lag per partition
            last_lag = now
            try:
                for tp in consumer.position(consumer.assignment()):
                    lo, hi = consumer.get_watermark_offsets(tp, timeout=2)
                    pos = tp.offset if tp.offset >= 0 else lo
                    g_lag.labels(partition=str(tp.partition)).set(max(0, hi - pos))
            except Exception:
                pass

        if args.stop_after_idle and now - last_msg > args.stop_after_idle:
            print("[archiver] idle, stopping")
            break

    flush()
    consumer.close()
    print(f"[archiver] done. total rows archived: {total:,}")


if __name__ == "__main__":
    main()
