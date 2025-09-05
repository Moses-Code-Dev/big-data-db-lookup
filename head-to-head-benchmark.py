import os
import sqlite3
import random
import struct
import time
from tqdm import tqdm   # progress bar

# ==========================
# DiskHashDB Implementation
# ==========================
class DiskHashDB:
    def __init__(self, filename, num_slots=20_000_000, value_size=64):
        """
        Simple disk-based hash table.
        :param filename: file for storage
        :param num_slots: number of hash slots (>= number of keys to reduce collisions)
        :param value_size: fixed size in bytes for the value
        """
        self.filename = filename
        self.num_slots = num_slots
        self.value_size = value_size
        self.slot_size = 8 + self.value_size  # 8 bytes for key + fixed value

        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.write(b"\x00" * (self.num_slots * self.slot_size))

        self.file = open(filename, "r+b")

    def _hash(self, key):
        return key % self.num_slots

    def put(self, key: int, value: str):
        """Insert (cellId → value)."""
        encoded = value.encode("utf-8")[: self.value_size]
        encoded = encoded.ljust(self.value_size, b"\x00")
        slot = self._hash(key)

        for _ in range(self.num_slots):
            pos = slot * self.slot_size
            self.file.seek(pos)
            raw = self.file.read(8)
            stored_key = struct.unpack("<Q", raw)[0]
            if stored_key in (0, key):  # empty or same key
                self.file.seek(pos)
                self.file.write(struct.pack("<Q", key) + encoded)
                return
            slot = (slot + 1) % self.num_slots
        raise RuntimeError("Hash table full")

    def get(self, key: int):
        """Retrieve value by cellId."""
        slot = self._hash(key)
        for _ in range(self.num_slots):
            pos = slot * self.slot_size
            self.file.seek(pos)
            raw = self.file.read(self.slot_size)
            stored_key = struct.unpack("<Q", raw[:8])[0]
            if stored_key == key:
                return raw[8:].rstrip(b"\x00").decode("utf-8")
            if stored_key == 0:  # empty slot
                return None
            slot = (slot + 1) % self.num_slots
        return None

    def close(self):
        self.file.close()


# ==========================
# Benchmark
# ==========================
def benchmark(sqlite_db="cells_10m.db", hash_db="cells_hash_10m.db",
              num_queries=10_000, num_slots=20_000_000):
    # 1) Load from SQLite
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    cursor.execute("SELECT cellId, storeName FROM cells")
    records = cursor.fetchall()
    conn.close()
    print(f"✅ Loaded {len(records)} records from SQLite")

    # 2) Build DiskHashDB
    if os.path.exists(hash_db):
        os.remove(hash_db)
    db = DiskHashDB(hash_db, num_slots=num_slots)

    start = time.time()
    for cellId, storeName in tqdm(records, desc="Inserting into DiskHashDB"):
        db.put(cellId, storeName)
    print(f"📝 Inserted {len(records)} rows into DiskHashDB in {time.time()-start:.2f}s")

    # 3) Prepare random queries
    sample_ids = [random.choice(records)[0] for _ in range(num_queries)]

    # 4) Benchmark SQLite
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    start = time.time()
    for cid in sample_ids:
        cursor.execute("SELECT storeName FROM cells WHERE cellId=?", (cid,))
        cursor.fetchone()
    sqlite_time = time.time() - start
    conn.close()

    # 5) Benchmark DiskHashDB
    start = time.time()
    for cid in sample_ids:
        db.get(cid)
    hash_time = time.time() - start

    db.close()

    # 6) Print results
    print("\n=== Benchmark Results ===")
    print(f"SQLite:     {sqlite_time:.4f} sec for {num_queries} lookups "
          f"({sqlite_time/num_queries*1000:.6f} ms/lookup)")
    print(f"DiskHashDB: {hash_time:.4f} sec for {num_queries} lookups "
          f"({hash_time/num_queries*1000:.6f} ms/lookup)")


if __name__ == "__main__":
    benchmark()
