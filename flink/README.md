# Flink DV-OASRS reference job

`DVOASRSJob` is a single-parallelism reference implementation that reads JSON from Kafka, discovers the `mote_id` or `sensor_id` key online, maintains per-stratum Welford/EMA and bounded priority state in Flink managed `MapState`, and emits a weighted mean summary periodically.

## Configuration

Environment variables:

| Name | Default | Meaning |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | Kafka brokers |
| `KAFKA_INPUT_TOPIC` | `sensor.raw` | Input JSON topic |
| `KAFKA_OUTPUT_TOPIC` | `sensor.dv_oasrs` | Summary topic |
| `KAFKA_GROUP_ID` | `dv-oasrs-flink` | Consumer group |
| `SAMPLE_BUDGET` | `500` | Global sample budget |
| `REALLOCATION_INTERVAL` | `10000` | Input records between periodic allocations; a new source triggers immediately |
| `REPORT_EVERY` | `10000` | Input records between result emissions |
| `CHECKPOINT_URI` | `file:///tmp/flink-checkpoints/dv-oasrs` | Flink checkpoint storage |

Input records must be JSON objects with `mote_id` or `sensor_id`, plus `temperature` or `value` as a number. The selected numeric field drives online variance and the emitted mean.

## State and delivery semantics

The global allocation is held under a constant keyed-stream key, so the job deliberately runs with parallelism one. Per-stratum counts, Welford/EMA values, capacities, and at most B priority candidates are stored in `MapState`; global counters are `ValueState`. Checkpoints are configured every 10 seconds with Flink `CheckpointingMode.EXACTLY_ONCE`. The Kafka sink has no transactional ID or exactly-once delivery guarantee configured; do not interpret checkpoint mode as end-to-end exactly-once output. The default checkpoint URI is local filesystem storage and is not suitable for a multi-node production deployment without changing configuration.

The priority candidates retain only numeric values and random priorities. Worst-case state is O(XB), where X is active strata. Priority candidate replacement in this Java reference scans the per-stratum candidate list, so the implementation is not optimized for high throughput.

## Build and run status

The job has not been compiled or run in the current environment: Maven and the Flink CLI were unavailable. There are no matched Flink exact, uniform reservoir, OASRS, or static VA-OASRS jobs yet, and no Kafka throughput/lag, latency, CPU, memory, checkpoint-size, recovery, or parallelism measurements. Treat this as an unverified implementation scaffold, not a validated Flink result. Build once Maven is available with `mvn -f flink/pom.xml package`; deploy the shaded JAR using the target Flink installation and a reachable Kafka cluster.
