import sqlite3
import random
import time
from faker import Faker

DB_NAME = "cells.db"
NUM_ROWS = 100_000
fake = Faker()

def create_schema(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cells (
            cellId     INTEGER PRIMARY KEY,
            lvl1       INTEGER NOT NULL,
            lvl2       INTEGER NOT NULL,
            lvl3       INTEGER NOT NULL,
            lvl4       INTEGER NOT NULL,
            storeId    TEXT,
            storeName  TEXT,
            avatarId   INTEGER,
            lastUpdate INTEGER,
            status     INTEGER
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lvl_path ON cells(lvl1, lvl2, lvl3, lvl4)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_storeId ON cells(storeId)")
    conn.commit()

def generate_cell_id(lvl1, lvl2, lvl3, lvl4):
    """Flattened ID: lvl1 as billions, lvl2 as millions, lvl3 as thousands, lvl4 as units."""
    return lvl1 * 1_000_000_000 + lvl2 * 1_000_000 + lvl3 * 1_000 + lvl4

def insert_dummy_data(conn, num_rows):
    cursor = conn.cursor()
    for _ in range(num_rows):
        lvl1 = random.randint(0, 3)
        lvl2 = random.randint(0, 999)
        lvl3 = random.randint(0, 999)
        lvl4 = random.randint(0, 999)

        cellId = generate_cell_id(lvl1, lvl2, lvl3, lvl4)
        storeId = f"ST{random.randint(1000,9999)}{fake.lexify(text='???')}"
        storeName = fake.company()
        avatarId = random.randint(0, 100)
        lastUpdate = int(time.time())  # epoch seconds
        status = random.randint(0, 1)

        cursor.execute("""
            INSERT OR IGNORE INTO cells
            (cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName, avatarId, lastUpdate, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName, avatarId, lastUpdate, status))

    conn.commit()

if __name__ == "__main__":
    conn = sqlite3.connect(DB_NAME)
    create_schema(conn)
    insert_dummy_data(conn, NUM_ROWS)
    conn.close()
    print(f"✅ Database '{DB_NAME}' created with {NUM_ROWS} rows.")
