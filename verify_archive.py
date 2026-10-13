"""Check the Parquet archive for completeness and show how 'exact' values are obtained.

  python verify_archive.py                       # inspect archive/raw
  python verify_archive.py --expected 2313682    # also compare with the producer's sent count

Checks: row count, duplicates (at-least-once delivery), missing seq numbers,
sensors / date range. Then prints exact per-attribute means: the denominators
for accuracy loss = |approx - exact| / exact.
"""
import argparse

import pandas as pd


def load(path: str) -> pd.DataFrame:
    import pyarrow.dataset as ds
    return ds.dataset(path, format="parquet", partitioning="hive").to_table().to_pandas()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="archive/raw")
    ap.add_argument("--expected", type=int, default=0, help="records the producer reported sending")
    args = ap.parse_args()

    df = load(args.dir)
    if df.empty:
        print("archive is empty")
        return

    rows = len(df)
    df = df.drop_duplicates("seq").sort_values("seq")
    uniq = len(df)
    span = int(df.seq.max()) + 1
    print(f"rows on disk        : {rows:,}")
    print(f"unique records      : {uniq:,}   (duplicates removed: {rows - uniq:,})")
    print(f"missing seq numbers : {span - uniq:,}   (seq range 0..{span - 1})")
    if args.expected:
        status = "OK" if uniq == args.expected else "MISMATCH"
        print(f"expected (producer) : {args.expected:,}   -> {status}")
    print(f"sensors             : {df.mote_id.nunique()}   events/sensor min/max = "
          f"{df.mote_id.value_counts().min():,}/{df.mote_id.value_counts().max():,}")
    print(f"event time          : {df.event_ts.min()}  ->  {df.event_ts.max()}")

    print("\nexact means over ALL archived records (ground truth):")
    print(df[["temperature", "humidity", "light", "voltage"]].mean().round(4).to_string())
    print("\nexact mean temperature per sensor (first 5 strata):")
    print(df.groupby("mote_id").temperature.mean().round(3).head().to_string())


if __name__ == "__main__":
    main()
