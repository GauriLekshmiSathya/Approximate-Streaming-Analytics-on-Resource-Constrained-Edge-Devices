"""Shared config and helpers for the ingestion / storage layer.

Design notes (see README):
  * Kafka message KEY = sensor (mote) id  -> each sensor is one stratum, as in
    ApproxECIoT (the paper splits the stream by key k_i; Kafka keys replace Flume).
  * Everything here is importable without Kafka installed, so it can be unit-tested.
"""
import os
from datetime import datetime, timedelta

# ----------------------------------------------------------------- config
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_RAW = os.getenv("TOPIC_RAW", "sensor.raw")          # full, unsampled stream
TOPIC_SAMPLED = os.getenv("TOPIC_SAMPLED", "sensor.sampled")  # written by the sampler stage
NUM_PARTITIONS = int(os.getenv("TOPIC_PARTITIONS", "3"))

N_MOTES = 54  # Intel Berkeley lab: sensors 1..54 (ids outside this range are junk)


# ----------------------------------------------------------------- parsing
def _parse_ts(date_s: str, time_s: str) -> str:
    """'2004-02-28' + '00:59:16.02785' -> '2004-02-28 00:59:16.027850'.

    The raw file has a variable number of fractional digits, so we normalise.
    Raises ValueError on a bad timestamp.
    """
    base, _, frac = time_s.partition(".")
    dt = datetime.strptime(f"{date_s} {base}", "%Y-%m-%d %H:%M:%S")
    if frac:
        if not frac.isdigit():
            raise ValueError("bad fraction")
        dt += timedelta(microseconds=int(frac.ljust(6, "0")[:6]))
    return dt.isoformat(sep=" ", timespec="microseconds")


def parse_intel_line(line: str):
    """Parse one line of the Intel Berkeley data.txt.

    Columns: date time epoch moteid temperature humidity light voltage
    Returns a dict, or None if the line is malformed / has an invalid sensor id.
    (We do NOT clean outliers here: the archive must hold the raw stream.)
    """
    parts = line.split()
    if len(parts) != 8:
        return None
    try:
        mote = int(parts[3])
        if not 1 <= mote <= N_MOTES:
            return None
        return {
            "event_ts": _parse_ts(parts[0], parts[1]),
            "epoch": int(parts[2]),
            "mote_id": mote,
            "temperature": float(parts[4]),
            "humidity": float(parts[5]),
            "light": float(parts[6]),
            "voltage": float(parts[7]),
        }
    except ValueError:
        return None


# ----------------------------------------------------------------- replay rate
class RateSchedule:
    """Replay-rate profile, e.g. "2000:30,10000:10" = 2000 ev/s for 30 s, then
    10000 ev/s for 10 s, repeating. A single number ("1000") is constant.
    Rate 0 means unthrottled (as fast as possible).
    Useful later for stressing the resource-aware controller with bursts.
    """

    def __init__(self, spec: str):
        self.segs = []
        for part in spec.split(","):
            rate, _, dur = part.strip().partition(":")
            self.segs.append((float(rate), float(dur) if dur else None))
        durations = [d for _, d in self.segs]
        self.cycle = sum(durations) if all(durations) else None

    def rate_at(self, t: float) -> float:
        if self.cycle:
            t %= self.cycle
        for rate, dur in self.segs:
            if dur is None or t < dur:
                return rate
            t -= dur
        return self.segs[-1][0]


# ----------------------------------------------------------------- kafka admin
def ensure_topics(bootstrap=BOOTSTRAP, topics=None, partitions=NUM_PARTITIONS):
    """Create the topics if they do not exist yet (idempotent)."""
    from confluent_kafka.admin import AdminClient, NewTopic

    topics = topics or [TOPIC_RAW, TOPIC_SAMPLED]
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=15).topics)
    new = [NewTopic(t, num_partitions=partitions, replication_factor=1)
           for t in topics if t not in existing]
    if not new:
        return
    for topic, fut in admin.create_topics(new).items():
        try:
            fut.result()
            print(f"[topics] created {topic} ({partitions} partitions)")
        except Exception as e:  # e.g. created concurrently
            print(f"[topics] {topic}: {e}")
