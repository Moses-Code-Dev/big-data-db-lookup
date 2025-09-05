import sqlite3
import random
import time

DB_NAME = "cells.db"

def benchmark_query(conn, query, params=None, repeat=10):
    cursor = conn.cursor()
    total_time = 0.0
    for _ in range(repeat):
        start = time.time()
        cursor.execute(query, params or ())
        cursor.fetchall()  # force fetching results
        total_time += (time.time() - start)
    return total_time / repeat

if __name__ == "__main__":
    conn = sqlite3.connect(DB_NAME)

    # Count total rows
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cells")
    total_rows = cursor.fetchone()[0]
    print(f"📊 Benchmarking on {total_rows} rows")

    # 1. Exact lookup by PRIMARY KEY (cellId)
    random_id = random.randint(1, total_rows)
    avg_time = benchmark_query(conn, "SELECT * FROM cells WHERE cellId = ?", (random_id,))
    print(f"🔍 Lookup by cellId: {avg_time*1000:.3f} ms")

    # 2. Lookup by storeId (random one from DB)
    cursor.execute("SELECT storeId FROM cells ORDER BY RANDOM() LIMIT 1")
    storeId = cursor.fetchone()[0]
    avg_time = benchmark_query(conn, "SELECT * FROM cells WHERE storeId = ?", (storeId,))
    print(f"🏪 Lookup by storeId: {avg_time*1000:.3f} ms")

    # 3. Range query by cellId
    cursor.execute("SELECT MIN(cellId), MAX(cellId) FROM cells")
    min_id, max_id = cursor.fetchone()
    lower = random.randint(min_id, max_id-1000)
    upper = lower + 1000
    avg_time = benchmark_query(conn, "SELECT * FROM cells WHERE cellId BETWEEN ? AND ?", (lower, upper))
    print(f"📈 Range query (1000 ids): {avg_time*1000:.3f} ms")

    # 4. Path lookup (lvl1, lvl2, lvl3, lvl4)
    random_lvl1 = random.randint(0, 3)
    random_lvl2 = random.randint(0, 999)
    avg_time = benchmark_query(conn, "SELECT * FROM cells WHERE lvl1 = ? AND lvl2 = ?", (random_lvl1, random_lvl2))
    print(f"🌍 Path lookup (lvl1+lvl2): {avg_time*1000:.3f} ms")

    conn.close()
