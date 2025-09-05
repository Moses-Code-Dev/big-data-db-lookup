#!/usr/bin/env python3
"""
build_and_benchmark_index.py

Build a 4-level in-memory index from a SQLite DB and benchmark:
  - in-memory lookup
  - sqlite warm lookup (single persistent connection)
  - sqlite cold lookup (open/close per lookup)

Usage:
  pip install tqdm
  python build_and_benchmark_index.py --db cells_10m.db --sample 200000 --bench-samples 1000 --repeat 3

Notes:
  * sample = how many rows to load into memory index (default 200k). If you set sample >= total DB rows, it will load all.
  * bench-samples = how many random lookups to perform (default 1000).
  * For large sample sizes use machines with enough RAM.
"""
import sqlite3
import random
import time
import argparse
import os
from tqdm import tqdm

def build_index(db_path, sample_limit=None, batch=10000):
    """Build nested dict index: lvl1 -> lvl2 -> lvl3 -> lvl4 -> payload
       Also returns a list of keys (lvl1,lvl2,lvl3,lvl4,cellId) useful for benchmarking.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_rows = cursor.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
    print(f"DB rows: {total_rows}")

    # Decide how many rows we'll load
    load_all = sample_limit is None or sample_limit >= total_rows
    to_load = total_rows if load_all else sample_limit
    print(f"Loading {to_load} rows into memory index (sample_limit={sample_limit})")

    query = "SELECT cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName, avatarId, lastUpdate, status FROM cells"
    cursor.execute(query)

    index = {}
    keys_list = []
    loaded = 0
    pbar = tqdm(total=to_load, desc="Building in-memory index", unit="rows")

    try:
        while loaded < to_load:
            rows = cursor.fetchmany(batch)
            if not rows:
                break
            for row in rows:
                cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName, avatarId, lastUpdate, status = row
                # nested dicts
                lvl1_map = index.setdefault(lvl1, {})
                lvl2_map = lvl1_map.setdefault(lvl2, {})
                lvl3_map = lvl2_map.setdefault(lvl3, {})
                # leaf store tuple (keep minimal data to save RAM)
                lvl3_map[lvl4] = (cellId, storeId, storeName, avatarId, lastUpdate, status)
                keys_list.append((lvl1, lvl2, lvl3, lvl4, cellId))
                loaded += 1
                pbar.update(1)
                if loaded >= to_load:
                    break
    finally:
        pbar.close()
        conn.close()

    print(f"Loaded {loaded} rows into memory index (keys_list length: {len(keys_list)})")
    return index, keys_list

def pick_random_keys_from_list(keys_list, n):
    """Choose n random keys from keys_list (or return all if n >= len(keys_list))."""
    if not keys_list:
        return []
    if n >= len(keys_list):
        # shuffle to avoid bias
        random.shuffle(keys_list)
        return keys_list.copy()
    return random.sample(keys_list, n)

def bench_in_memory(index, keys, repeat=1):
    """Benchmark in-memory lookups for given keys. Returns avg time per lookup in seconds."""
    if not keys:
        return float('nan')
    # warm-up
    for k in keys[:min(10, len(keys))]:
        l1,l2,l3,l4,_ = k
        _ = index[l1][l2][l3][l4]
    # timed loop
    t0 = time.time()
    for _ in range(repeat):
        for k in keys:
            l1,l2,l3,l4,_ = k
            _ = index[l1][l2][l3][l4]
    total = time.time() - t0
    avg = total / (len(keys) * repeat)
    return avg

def bench_sqlite_warm(db_path, keys, repeat=1):
    """Benchmark sqlite lookups with a persistent connection (warm cache)."""
    if not keys:
        return float('nan')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    t0 = time.time()
    for _ in range(repeat):
        for k in keys:
            cellId = k[4]
            cursor.execute("SELECT cellId FROM cells WHERE cellId = ?", (cellId,))
            cursor.fetchone()
    total = time.time() - t0
    conn.close()
    avg = total / (len(keys) * repeat)
    return avg

def bench_sqlite_cold(db_path, keys, repeat=1):
    """Benchmark sqlite lookups by opening/closing connection per lookup (cold reads)."""
    if not keys:
        return float('nan')
    t0 = time.time()
    for _ in range(repeat):
        for k in keys:
            cellId = k[4]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT cellId FROM cells WHERE cellId = ?", (cellId,))
            cursor.fetchone()
            conn.close()
    total = time.time() - t0
    avg = total / (len(keys) * repeat)
    return avg

def main():
    parser = argparse.ArgumentParser(description="Build in-memory index and benchmark lookups compared to SQLite.")
    parser.add_argument("--db", type=str, default="cells_10m.db", help="SQLite DB file (default cells_10m.db)")
    parser.add_argument("--sample", type=int, default=200000, help="Rows to load into memory index (default 200k). Use smaller on limited RAM.")
    parser.add_argument("--bench-samples", type=int, default=1000, help="Number of random lookups to benchmark (default 1000).")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat cycles during bench (default 3).")
    parser.add_argument("--batch", type=int, default=10000, help="fetchmany batch size when streaming DB (default 10000).")
    args = parser.parse_args()

    print("=== config ===")
    print(f"DB: {args.db}")
    print(f"sample (rows to load): {args.sample}")
    print(f"bench-samples: {args.bench_samples}")
    print(f"repeat: {args.repeat}")
    print("===============")

    index, keys_list = build_index(args.db, sample_limit=args.sample, batch=args.batch)
    # If keys_list is shorter than requested bench-samples, use what's available
    desired = args.bench_samples
    if len(keys_list) < desired:
        print(f"Warning: only {len(keys_list)} keys available, reducing bench-samples to that number.")
        desired = len(keys_list)
    bench_keys = pick_random_keys_from_list(keys_list, desired)
    print(f"Prepared {len(bench_keys)} keys for benchmark.")

    print("\nRunning benchmarks (this may take a while depending on bench-samples and repeat)...")
    in_mem_avg = bench_in_memory(index, bench_keys, repeat=args.repeat)
    print(f"In-memory avg lookup: {in_mem_avg*1000:.6f} ms")

    sqlite_warm_avg = bench_sqlite_warm(args.db, bench_keys, repeat=args.repeat)
    print(f"SQLite (warm connection) avg lookup: {sqlite_warm_avg*1000:.6f} ms")

    sqlite_cold_avg = bench_sqlite_cold(args.db, bench_keys, repeat=args.repeat)
    print(f"SQLite (cold open/close) avg lookup: {sqlite_cold_avg*1000:.6f} ms")

    print("\nSummary (avg per lookup):")
    print(f"  In-memory : {in_mem_avg*1000:.6f} ms")
    print(f"  SQLite(warm): {sqlite_warm_avg*1000:.6f} ms")
    print(f"  SQLite(cold): {sqlite_cold_avg*1000:.6f} ms")

if __name__ == "__main__":
    main()
