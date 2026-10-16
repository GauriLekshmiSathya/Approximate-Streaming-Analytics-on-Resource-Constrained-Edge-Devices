"""Run OASRS over ``sensor.raw`` and publish per-window sample summaries.

This is a Kafka integration extension around the standalone OASRS library. It
uses event-time windows anchored at the first event unless ``--base-time`` is
provided. Kafka partitions are merged by sorting bounded batches, and the
watermark allows configurable lateness. The raw archive remains the durable
source for offline exact ground truth.
"""

import argparse
import json
import logging
import signal
import time
from datetime import datetime, timezone

import common
from streamapprox.control import AdaptiveController, LinearCostFunction
from streamapprox.estimators import compare_with_exact
from streamapprox.windows import TimeWindowManager

LOG = logging.getLogger("streamapprox.sampler")


def _event_seconds(record):
    value = datetime.fromisoformat(record["event_ts"])
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _window_payload(result, next_sample_size=None):
    snap = result.snapshot
    strata = {}
    for sid, state in snap.strata.items():
        strata[str(sid)] = {
            "count": state.total_seen,
            "sample_count": state.sample_size,
            "capacity": state.capacity,
            "weight": state.weight,
            "samples": state.samples,
        }
    payload = {
        "type": "window_result",
        "window_start": result.window_start,
        "window_end": result.window_end,
        "input_items": result.total_seen,
        "sampled_items": result.sample_size,
        "strata_count": result.num_strata,
        "query": "temperature_mean",
        "approximate_sum": result.sum_result.estimate,
        "sum_standard_error": result.sum_result.standard_error,
        "sum_sigma_ranges": result.sum_result.sigma_ranges,
        "approximate_mean": result.mean_result.estimate,
        "mean_standard_error": result.mean_result.standard_error,
        "mean_sigma_ranges": result.mean_result.sigma_ranges,
        "approximate_count": result.count_result,
        "strata": strata,
    }
    if result.exact_sum is not None:
        mean_error = compare_with_exact(result.mean_result.estimate, result.exact_mean)
        sum_error = compare_with_exact(result.sum_result.estimate, result.exact_sum)
        payload.update({
            "exact_sum": result.exact_sum,
            "exact_mean": result.exact_mean,
            "exact_count": result.exact_count,
            "sum_absolute_error": sum_error.absolute_error,
            "sum_accuracy_loss": sum_error.accuracy_loss,
            "mean_absolute_error": mean_error.absolute_error,
            "mean_accuracy_loss": mean_error.accuracy_loss,
        })
    if next_sample_size is not None:
        payload["next_sample_size_per_stratum"] = next_sample_size
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default=common.TOPIC_RAW)
    ap.add_argument("--output-topic", default=common.TOPIC_SAMPLED)
    ap.add_argument("--bootstrap", default=common.BOOTSTRAP)
    ap.add_argument("--group", default="oasrs-sampler")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--slide", type=float, default=5.0)
    ap.add_argument("--allowed-lateness", type=float, default=3600.0,
                    help="event-time watermark grace period in seconds")
    ap.add_argument("--time-mode", choices=("event", "processing"), default="event",
                    help="window on source event timestamps or Kafka processing time")
    ap.add_argument("--sample-size", type=int, default=1000,
                    help="reservoir capacity per stratum")
    ap.add_argument("--query-budget", type=float, default=None,
                    help="optional abstract budget mapped to per-stratum capacity")
    ap.add_argument("--items-per-budget-unit", type=float, default=1000.0,
                    help="engineering cost-function scale used with --query-budget")
    ap.add_argument("--expected-strata", type=int, default=54)
    ap.add_argument("--target-relative-error", type=float, default=0.05,
                    help="adaptive target; set 0 to disable sample-size feedback")
    ap.add_argument("--max-sample-size", type=int, default=10000)
    ap.add_argument("--evaluation", action="store_true",
                    help="track exact sum/mean/count per window in constant memory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-time", type=float, default=None,
                    help="event-time epoch boundary; default anchors at first event")
    ap.add_argument("--metrics-port", type=int, default=8002)
    ap.add_argument("--stop-after-idle", type=float, default=0,
                    help="stop after N idle seconds (0 means run forever)")
    args = ap.parse_args()
    if (args.sample_size < 1 or args.window <= 0 or args.slide <= 0 or args.slide > args.window
            or args.allowed_lateness < 0
            or args.max_sample_size < args.sample_size or args.expected_strata < 1):
        ap.error("invalid sample size, window, slide, or lateness configuration")
    if args.query_budget is not None:
        args.sample_size = LinearCostFunction(
            items_per_budget_unit=args.items_per_budget_unit,
            minimum=1,
            maximum=args.max_sample_size,
        ).sample_size(args.query_budget, strata=args.expected_strata)

    from confluent_kafka import Consumer, KafkaError, Producer
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    consumed = Counter("sampler_input_messages_total", "Raw messages read by OASRS sampler")
    malformed = Counter("sampler_malformed_messages_total", "Malformed input records")
    publish_errors = Counter("sampler_publish_errors_total", "Window summary publish failures")
    windows = Counter("sampler_windows_total", "Completed approximate query windows")
    sampled = Counter("sampler_sampled_items_total", "Items retained in completed windows")
    current_sampled = Gauge("sampler_sampled_items_current", "Items retained in the most recently completed window")
    mean_loss = Gauge("sampler_mean_accuracy_loss", "Per-window empirical mean accuracy loss")
    estimate_error = Gauge("sampler_mean_estimated_standard_error", "Per-window estimated mean standard error")
    capacity_gauge = Gauge("sampler_capacity_per_stratum", "Capacity for newly created windows")
    latency = Histogram("sampler_window_processing_seconds", "Window finalization and publish time")
    input_rate = Gauge("sampler_input_rate", "Input records processed per second")
    start_http_server(args.metrics_port)

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    producer = Producer({"bootstrap.servers": args.bootstrap})
    manager = TimeWindowManager(
        window_duration=args.window,
        slide_interval=args.slide,
        sample_size=args.sample_size,
        stratum_key="mote_id",
        timestamp_key=_event_seconds if args.time_mode == "event" else None,
        value_key="temperature",
        seed=args.seed,
        base_time=args.base_time,
        evaluation_mode=args.evaluation,
        allowed_lateness=(args.allowed_lateness if args.time_mode == "event" else 0.0),
    )
    consumer.subscribe([args.topic])
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    last_msg = time.monotonic()
    rate_start = last_msg
    rate_count = 0
    effective_lateness = args.allowed_lateness if args.time_mode == "event" else 0.0
    LOG.info("sampling %s -> %s (mode=%s window=%ss slide=%ss lateness=%ss N_i=%s)",
             args.topic, args.output_topic, args.time_mode, args.window,
             args.slide, effective_lateness, args.sample_size)

    controller = (
        AdaptiveController(
            target_relative_error=args.target_relative_error,
            minimum_sample_size=1,
            maximum_sample_size=args.max_sample_size,
        ) if args.target_relative_error > 0 else None
    )

    def publish_result(result):
        started = time.perf_counter()
        next_size = None
        if controller is not None:
            next_size = controller.next_sample_size(
                manager.sample_size,
                result.mean_result.estimate,
                result.mean_result.standard_error,
            )
            manager.set_sample_size(next_size)
            capacity_gauge.set(next_size)
        payload = _window_payload(result, next_size)
        if result.exact_mean is not None:
            mean_loss.set(compare_with_exact(result.mean_result.estimate, result.exact_mean).accuracy_loss)
        estimate_error.set(result.mean_result.standard_error)

        def on_delivery(error, _message):
            if error is not None:
                publish_errors.inc()
                LOG.error("Kafka summary delivery failed: %s", error)

        while True:
            try:
                producer.produce(
                    args.output_topic,
                    key=f"{result.window_start:.6f}:{result.window_end:.6f}".encode(),
                    value=json.dumps(payload, separators=(",", ":")).encode(),
                    on_delivery=on_delivery,
                )
                break
            except BufferError:
                producer.poll(0.1)
        producer.poll(0)
        windows.inc()
        sampled.inc(result.sample_size)
        current_sampled.set(result.sample_size)
        latency.observe(time.perf_counter() - started)
        LOG.info("window [%s,%s): count=%d sample=%d mean=%.6g mean_se=%.6g next_N=%s exact_mean=%s",
                 result.window_start, result.window_end, result.total_seen,
                 result.sample_size, result.mean_result.estimate,
                 result.mean_result.standard_error, next_size, result.exact_mean)

    manager.set_result_callback(publish_result)

    try:
        while running:
            messages = consumer.consume(num_messages=500, timeout=1.0)
            now = time.monotonic()
            ordered_records = []
            for message in messages:
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        LOG.warning("Kafka consumer error: %s", message.error())
                    continue
                try:
                    record = json.loads(message.value())
                    event_seconds = _event_seconds(record)
                    if "mote_id" not in record or "temperature" not in record:
                        raise ValueError("missing mote_id or temperature")
                    ordered_records.append((event_seconds, record))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    malformed.inc()
                    LOG.warning("skip malformed Kafka record: %s", exc)
            records_to_process = (
                sorted(ordered_records, key=lambda pair: pair[0])
                if args.time_mode == "event"
                else [(now, record) for _, record in ordered_records]
            )
            for timestamp, record in records_to_process:
                manager.update(record, timestamp=timestamp)
                consumed.inc()
                rate_count += 1
                last_msg = now
            elapsed = now - rate_start
            if elapsed >= 2.0:
                input_rate.set(rate_count / elapsed)
                rate_count, rate_start = 0, now
            if messages:
                # Do not commit while window state is only in memory. On a
                # crash, Kafka replays from the previous committed offset.
                if producer.flush(30) != 0:
                    raise RuntimeError("timed out publishing completed window summaries")
            if args.stop_after_idle and now - last_msg >= args.stop_after_idle:
                break
        manager.flush()
        if producer.flush(30) != 0:
            raise RuntimeError("timed out publishing final window summaries")
    finally:
        consumer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
