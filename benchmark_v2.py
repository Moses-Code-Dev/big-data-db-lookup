import sqlite3
import random
import time

DB_NAME = "cells_10m.db"  # adjust as needed

def benchmark_cold(query, params=None, repeat=5):
    total_time = 0.0
    for _ in range(repeat):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        start = time.time()
        cursor.execute(query, params or ())
        cursor.fetchall()
        total_time += (time.time() - start)

        conn.close()  # force close (flush cache)
    return total_time / repeat

if __name__ == "__main__":
    # Pick a random row to test consistently
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cells")
    total_rows = cursor.fetchone()[0]
    print(f"📊 Cold Benchmarking on {total_rows} rows")

    cursor.execute("SELECT cellId, lvl1, lvl2, lvl3, lvl4 FROM cells ORDER BY RANDOM() LIMIT 1")
    known_cellId, lvl1, lvl2, lvl3, lvl4 = cursor.fetchone()
    conn.close()

    print(f"🔹 Testing for cellId: {known_cellId} (lvl path: {lvl1}-{lvl2}-{lvl3}-{lvl4})")

    # Structured lookup cold
    avg_time = benchmark_cold(
        "SELECT * FROM cells WHERE lvl1 = ? AND lvl2 = ? AND lvl3 = ? AND lvl4 = ?",
        (lvl1, lvl2, lvl3, lvl4)
    )
    print(f"🌐 Structured lookup (cold): {avg_time*1000:.3f} ms")

    # Direct lookup cold
    avg_time = benchmark_cold(
        "SELECT * FROM cells WHERE cellId = ?",
        (known_cellId,)
    )
    print(f"🔍 Direct PK lookup (cold): {avg_time*1000:.3f} ms")
