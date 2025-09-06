#!/usr/bin/env python3
"""
MinMax99 Comprehensive Benchmark System
=======================================

Benchmarks 3 different data access methods with WARM and COLD start support:
1. SQLite cellId index vs SQLite structured path index
2. SQLite cellId index vs Custom In-Memory 4-level B-Tree
3. Triple comparison: SQLite vs In-Memory vs Direct Disk Hash

Usage:
    pip install matplotlib tqdm
    python minmax99_benchmark.py --db cells_1m.db --samples 10000 --repeat 5
    python minmax99_benchmark.py --db cells_1m.db --samples 5000 --cold-start
    python minmax99_benchmark.py --db cells_1m.db --samples 5000 --both
"""

import sqlite3
import random
import time
import argparse
import os
import struct
import sys
import gc
import subprocess
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# ==========================
# System Cache Management
# ==========================
def clear_system_cache():
    """Clear OS file system cache (Linux/macOS/Windows)."""
    try:
        if os.name == 'posix':  # Linux/macOS
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['sudo', 'purge'], check=False, capture_output=True)
            else:  # Linux
                subprocess.run(['sudo', 'sync'], check=False, capture_output=True)
                subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'], check=False, capture_output=True)
        elif os.name == 'nt':  # Windows
            subprocess.run(['wmic', 'process', 'where', 'name="python.exe"', 'call', 'setpriority', '32768'], 
                         check=False, capture_output=True)
    except Exception:
        pass  # Cache clearing is best-effort


# ==========================
# DiskHashDB Implementation
# ==========================
class DiskHashDB:
    def __init__(self, filename, num_slots=500_000, value_size=128):
        """High-performance disk-based hash table for direct O(1) lookups."""
        self.filename = filename
        self.num_slots = num_slots
        self.value_size = value_size
        self.slot_size = 8 + self.value_size  # 8 bytes for key + fixed value
        
        if not os.path.exists(filename):
            print(f"🔨 Creating disk hash file: {filename}")
            with open(filename, "wb") as f:
                f.write(b"\x00" * (self.num_slots * self.slot_size))
        
        self.file = open(filename, "r+b")

    def _hash(self, key):
        return (key * 2654435761) % self.num_slots

    def put(self, key: int, value: str):
        """Insert (cellId → value) with linear probing."""
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
        
        raise RuntimeError("Hash table full - increase num_slots")

    def get(self, key: int):
        """Retrieve value by cellId with linear probing."""
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
        if hasattr(self, 'file') and self.file:
            self.file.close()


# ==========================
# MinMax99 In-Memory 4-Level Tree
# ==========================
class MinMax99Tree:
    def __init__(self):
        """4-level nested dictionary structure: lvl1[lvl2[lvl3[lvl4]]] → data"""
        self.tree = {}
        self.size = 0

    def put(self, cellId: int, data):
        """Insert data using 4-level path derived from cellId."""
        lvl1, lvl2, lvl3, lvl4 = self._split_cellId(cellId)
        
        # Create nested structure
        if lvl1 not in self.tree:
            self.tree[lvl1] = {}
        if lvl2 not in self.tree[lvl1]:
            self.tree[lvl1][lvl2] = {}
        if lvl3 not in self.tree[lvl1][lvl2]:
            self.tree[lvl1][lvl2][lvl3] = {}
        
        self.tree[lvl1][lvl2][lvl3][lvl4] = data
        self.size += 1

    def get(self, cellId: int):
        """Retrieve data using 4-level path."""
        lvl1, lvl2, lvl3, lvl4 = self._split_cellId(cellId)
        
        try:
            return self.tree[lvl1][lvl2][lvl3][lvl4]
        except KeyError:
            return None

    def _split_cellId(self, cellId: int):
        """Split cellId into 4-level path: lvl1, lvl2, lvl3, lvl4"""
        lvl4 = cellId % 1000
        cellId //= 1000
        lvl3 = cellId % 1000
        cellId //= 1000
        lvl2 = cellId % 1000
        cellId //= 1000
        lvl1 = cellId % 4  # 0-3 range
        
        return lvl1, lvl2, lvl3, lvl4


# ==========================
# Data Loading and Preparation
# ==========================
def load_sqlite_data(db_path, limit=None):
    """Load data from SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    total_count = cursor.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
    print(f"📊 Total records in DB: {total_count:,}")
    
    # Load records
    if limit and limit < total_count:
        query = f"SELECT cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName FROM cells LIMIT {limit}"
        print(f"🔄 Loading {limit:,} records...")
    else:
        query = "SELECT cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName FROM cells"
        print(f"🔄 Loading all {total_count:,} records...")
    
    cursor.execute(query)
    records = cursor.fetchall()
    conn.close()
    
    print(f"✅ Loaded {len(records):,} records")
    return records


def build_minmax99_tree(records):
    """Build MinMax99 4-level tree from records."""
    print("🌳 Building MinMax99 4-level tree...")
    tree = MinMax99Tree()
    
    for record in tqdm(records, desc="Building tree"):
        cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName = record
        data = {
            'cellId': cellId,
            'lvl1': lvl1, 'lvl2': lvl2, 'lvl3': lvl3, 'lvl4': lvl4,
            'storeId': storeId, 'storeName': storeName
        }
        tree.put(cellId, data)
    
    print(f"✅ MinMax99 tree built with {tree.size:,} nodes")
    return tree


def build_disk_hash(records, hash_filename):
    """Build disk-based hash table from records."""
    print("💾 Building disk hash table...")
    
    # Remove existing file
    if os.path.exists(hash_filename):
        os.remove(hash_filename)
    
    # Calculate optimal slot count (1.5x records to reduce collisions)
    num_slots = int(len(records) * 1.5)
    db = DiskHashDB(hash_filename, num_slots=num_slots)
    
    for record in tqdm(records, desc="Building hash"):
        cellId, lvl1, lvl2, lvl3, lvl4, storeId, storeName = record
        value = f"{storeId}|{storeName}"
        db.put(cellId, value)
    
    print(f"✅ Disk hash built with {len(records):,} records")
    return db


# ==========================
# Benchmark Functions
# ==========================
def benchmark_sqlite_cellid(db_path, cellIds, repeat=1, cold_start=False):
    """Benchmark SQLite direct cellId lookup."""
    if cold_start:
        return benchmark_sqlite_cellid_cold(db_path, cellIds, repeat)
    
    # Warm connection benchmark
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_time = 0
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            cursor.execute("SELECT * FROM cells WHERE cellId = ?", (cellId,))
            cursor.fetchone()
        total_time += time.perf_counter() - start
    
    conn.close()
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_sqlite_cellid_cold(db_path, cellIds, repeat=1):
    """Benchmark SQLite with cold starts (new connection per lookup)."""
    total_time = 0
    
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            # Cold start: new connection per lookup
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cells WHERE cellId = ?", (cellId,))
            cursor.fetchone()
            conn.close()
        total_time += time.perf_counter() - start
    
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_sqlite_structured(db_path, records_dict, cellIds, repeat=1, cold_start=False):
    """Benchmark SQLite structured path lookup (lvl1+lvl2+lvl3+lvl4)."""
    if cold_start:
        return benchmark_sqlite_structured_cold(db_path, records_dict, cellIds, repeat)
    
    # Warm connection benchmark
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_time = 0
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            if cellId in records_dict:
                _, lvl1, lvl2, lvl3, lvl4, _, _ = records_dict[cellId]
                cursor.execute(
                    "SELECT * FROM cells WHERE lvl1=? AND lvl2=? AND lvl3=? AND lvl4=?",
                    (lvl1, lvl2, lvl3, lvl4)
                )
                cursor.fetchone()
        total_time += time.perf_counter() - start
    
    conn.close()
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_sqlite_structured_cold(db_path, records_dict, cellIds, repeat=1):
    """Benchmark SQLite structured path with cold starts."""
    total_time = 0
    
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            if cellId in records_dict:
                _, lvl1, lvl2, lvl3, lvl4, _, _ = records_dict[cellId]
                # Cold start: new connection per lookup
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM cells WHERE lvl1=? AND lvl2=? AND lvl3=? AND lvl4=?",
                    (lvl1, lvl2, lvl3, lvl4)
                )
                cursor.fetchone()
                conn.close()
        total_time += time.perf_counter() - start
    
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_minmax99_tree(tree, cellIds, repeat=1, cold_start=False):
    """Benchmark MinMax99 in-memory tree lookup."""
    if cold_start:
        return benchmark_minmax99_tree_cold(tree, cellIds, repeat)
    
    # Warm benchmark
    total_time = 0
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            tree.get(cellId)
        total_time += time.perf_counter() - start
    
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_minmax99_tree_cold(tree, cellIds, repeat=1):
    """Benchmark MinMax99 tree with cold simulation."""
    total_time = 0
    
    for _ in range(repeat):
        # Clear Python GC
        gc.collect()
        if _ == 0:
            time.sleep(0.1)  # Let system settle
            
        start = time.perf_counter()
        for cellId in cellIds:
            tree.get(cellId)
        total_time += time.perf_counter() - start
    
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_disk_hash(hash_db, cellIds, repeat=1, cold_start=False):
    """Benchmark disk-based hash table lookup."""
    if cold_start:
        return benchmark_disk_hash_cold(hash_db, cellIds, repeat)
    
    # Warm benchmark
    total_time = 0
    for _ in range(repeat):
        start = time.perf_counter()
        for cellId in cellIds:
            hash_db.get(cellId)
        total_time += time.perf_counter() - start
    
    return (total_time / repeat) / len(cellIds)  # Average per lookup


def benchmark_disk_hash_cold(hash_db, cellIds, repeat=1):
    """Benchmark disk hash with cold starts (reopen file per lookup)."""
    filename = hash_db.filename
    num_slots = hash_db.num_slots
    value_size = hash_db.value_size
    
    # Close existing connection
    hash_db.close()
    
    total_time = 0
    for _ in range(repeat):
        # Clear Python GC and system cache
        gc.collect()
        if _ == 0:  # Clear cache only on first run to simulate real cold start
            clear_system_cache()
            time.sleep(0.1)  # Let system settle
            
        start = time.perf_counter()
        for cellId in cellIds:
            # Cold start: new file handle per lookup
            temp_db = DiskHashDB.__new__(DiskHashDB)
            temp_db.filename = filename
            temp_db.num_slots = num_slots
            temp_db.value_size = value_size
            temp_db.slot_size = 8 + value_size
            temp_db.file = open(filename, "r+b")
            temp_db._hash = lambda key: (key * 2654435761) % num_slots
            
            temp_db.get(cellId)
            temp_db.file.close()
            
        total_time += time.perf_counter() - start
    
    # Reopen original connection
    hash_db.file = open(filename, "r+b")
    return (total_time / repeat) / len(cellIds)  # Average per lookup


# ==========================
# Visualization
# ==========================
def plot_benchmark_results(results, output_file="minmax99_benchmark.png"):
    """Create comprehensive benchmark visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MinMax99 Benchmark Results', fontsize=16, fontweight='bold')
    
    # Test 1: SQLite cellId vs SQLite Structured
    if 'test1' in results:
        methods = ['SQLite cellId', 'SQLite Structured']
        times = [results['test1']['sqlite_cellid'], results['test1']['sqlite_structured']]
        bars1 = ax1.bar(methods, times, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Test 1: SQLite cellId vs SQLite Structured Path')
        ax1.set_ylabel('Avg Time per Lookup (ms)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.4f}ms', ha='center', va='bottom')
    
    # Test 2: SQLite cellId vs MinMax99 Tree
    if 'test2' in results:
        methods = ['SQLite cellId', 'MinMax99 Tree']
        times = [results['test2']['sqlite_cellid'], results['test2']['minmax99_tree']]
        bars2 = ax2.bar(methods, times, color=['#2E86AB', '#F18F01'])
        ax2.set_title('Test 2: SQLite cellId vs MinMax99 In-Memory Tree')
        ax2.set_ylabel('Avg Time per Lookup (ms)')
        ax2.grid(True, alpha=0.3)
        
        for bar, time in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.4f}ms', ha='center', va='bottom')
    
    # Test 3: Triple Comparison
    if 'test3' in results:
        methods = ['SQLite cellId', 'MinMax99 Tree', 'Disk Hash']
        times = [results['test3']['sqlite_cellid'], 
                results['test3']['minmax99_tree'], 
                results['test3']['disk_hash']]
        bars3 = ax3.bar(methods, times, color=['#2E86AB', '#F18F01', '#C73E1D'])
        ax3.set_title('Test 3: Triple Comparison')
        ax3.set_ylabel('Avg Time per Lookup (ms)')
        ax3.grid(True, alpha=0.3)
        
        for bar, time in zip(bars3, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.4f}ms', ha='center', va='bottom')
    
    # Performance Summary (Log Scale)
    if all(f'test{i}' in results for i in range(1, 4)):
        all_methods = ['SQLite\ncellId', 'SQLite\nStruct', 
                      'MinMax99\nTree', 'Disk\nHash']
        
        # Get unique times (avoiding duplicates)
        sqlite_cellid_avg = np.mean([results['test1']['sqlite_cellid'], 
                                     results['test2']['sqlite_cellid'],
                                     results['test3']['sqlite_cellid']])
        
        all_times = [sqlite_cellid_avg,
                    results['test1']['sqlite_structured'],
                    results['test3']['minmax99_tree'], 
                    results['test3']['disk_hash']]
        
        bars4 = ax4.bar(range(len(all_methods)), all_times, 
                       color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax4.set_title('Performance Summary (All Methods)')
        ax4.set_ylabel('Avg Time per Lookup (ms) - Log Scale')
        ax4.set_yscale('log')
        ax4.set_xticks(range(len(all_methods)))
        ax4.set_xticklabels(all_methods)
        ax4.grid(True, alpha=0.3, which='both')
        
        # Add value labels
        for bar, time in zip(bars4, all_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{time:.4f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 Benchmark graph saved: {output_file}")
    plt.show()


# ==========================
# Main Benchmark Runner
# ==========================
def run_comprehensive_benchmark(db_path, samples=10000, repeat=3, limit=None, cold_start=False):
    """Run all benchmark tests and generate results."""
    print("🚀 Starting MinMax99 Comprehensive Benchmark")
    print(f"🌡️  Mode: {'COLD START' if cold_start else 'WARM'} benchmarking")
    print("=" * 60)
    
    # Load data
    records = load_sqlite_data(db_path, limit=limit)
    records_dict = {record[0]: record for record in records}  # cellId -> record
    
    # Prepare random sample cellIds for benchmarking
    sample_cellIds = random.sample([r[0] for r in records], min(samples, len(records)))
    print(f"🎯 Using {len(sample_cellIds):,} random cellIds for benchmarking")
    print(f"📊 Repeat count: {repeat}")
    
    results = {}
    
    # ==========================
    # TEST 1: SQLite cellId vs SQLite Structured Path
    # ==========================
    print("\n" + "="*60)
    print("🔍 TEST 1: SQLite cellId vs SQLite Structured Path")
    print("="*60)
    
    if cold_start:
        print("❄️  Running COLD START benchmark (new connections per lookup)...")
    
    sqlite_cellid_time = benchmark_sqlite_cellid(db_path, sample_cellIds, repeat, cold_start)
    sqlite_structured_time = benchmark_sqlite_structured(db_path, records_dict, sample_cellIds, repeat, cold_start)
    
    results['test1'] = {
        'sqlite_cellid': sqlite_cellid_time * 1000,  # convert to ms
        'sqlite_structured': sqlite_structured_time * 1000
    }
    
    print(f"SQLite cellId lookup:     {sqlite_cellid_time*1000:.4f} ms/lookup")
    print(f"SQLite structured lookup: {sqlite_structured_time*1000:.4f} ms/lookup")
    print(f"Winner: {'cellId' if sqlite_cellid_time < sqlite_structured_time else 'Structured'}")
    speedup = sqlite_structured_time / sqlite_cellid_time if sqlite_cellid_time > 0 else 0
    print(f"Speedup factor: {speedup:.2f}x")
    
    # ==========================
    # TEST 2: SQLite cellId vs MinMax99 In-Memory Tree
    # ==========================
    print("\n" + "="*60)
    print("🌳 TEST 2: SQLite cellId vs MinMax99 In-Memory Tree")
    print("="*60)
    
    tree = build_minmax99_tree(records)
    
    sqlite_cellid_time2 = benchmark_sqlite_cellid(db_path, sample_cellIds, repeat, cold_start)
    minmax99_tree_time = benchmark_minmax99_tree(tree, sample_cellIds, repeat, cold_start)
    
    results['test2'] = {
        'sqlite_cellid': sqlite_cellid_time2 * 1000,
        'minmax99_tree': minmax99_tree_time * 1000
    }
    
    print(f"SQLite cellId lookup:    {sqlite_cellid_time2*1000:.4f} ms/lookup")
    print(f"MinMax99 tree lookup:    {minmax99_tree_time*1000:.4f} ms/lookup")
    print(f"Winner: {'SQLite' if sqlite_cellid_time2 < minmax99_tree_time else 'MinMax99'}")
    speedup = sqlite_cellid_time2 / minmax99_tree_time if minmax99_tree_time > 0 else 0
    print(f"Speedup factor: {speedup:.2f}x")
    
    # ==========================
    # TEST 3: Triple Comparison
    # ==========================
    print("\n" + "="*60)
    print("⚡ TEST 3: Triple Comparison (SQLite vs MinMax99 vs Disk Hash)")
    print("="*60)
    
    hash_filename = db_path.replace('.db', '_hash.dat')
    hash_db = build_disk_hash(records, hash_filename)
    
    if cold_start:
        print("❄️  Running COLD START benchmark (includes file system cache clearing)...")
    
    sqlite_cellid_time3 = benchmark_sqlite_cellid(db_path, sample_cellIds, repeat, cold_start)
    minmax99_tree_time3 = benchmark_minmax99_tree(tree, sample_cellIds, repeat, cold_start)
    disk_hash_time = benchmark_disk_hash(hash_db, sample_cellIds, repeat, cold_start)
    
    results['test3'] = {
        'sqlite_cellid': sqlite_cellid_time3 * 1000,
        'minmax99_tree': minmax99_tree_time3 * 1000,
        'disk_hash': disk_hash_time * 1000
    }
    
    print(f"SQLite cellId lookup:    {sqlite_cellid_time3*1000:.4f} ms/lookup")
    print(f"MinMax99 tree lookup:    {minmax99_tree_time3*1000:.4f} ms/lookup")
    print(f"Disk hash lookup:        {disk_hash_time*1000:.4f} ms/lookup")
    
    # Find winner
    times = [sqlite_cellid_time3, minmax99_tree_time3, disk_hash_time]
    methods = ['SQLite', 'MinMax99', 'Disk Hash']
    winner_idx = times.index(min(times))
    print(f"Winner: {methods[winner_idx]}")
    
    # Calculate speedups relative to SQLite
    if sqlite_cellid_time3 > 0:
        print(f"MinMax99 speedup over SQLite: {sqlite_cellid_time3/minmax99_tree_time3:.2f}x")
        print(f"Disk Hash speedup over SQLite: {sqlite_cellid_time3/disk_hash_time:.2f}x")
    
    # Cleanup
    hash_db.close()
    
    # Generate graph
    mode_suffix = "_cold" if cold_start else "_warm"
    output_file = f"minmax99_benchmark{mode_suffix}.png"
    plot_benchmark_results(results, output_file)
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE!")
    print("="*60)
    
    # Print summary table
    print("\n📊 Summary Table (ms per lookup):")
    print("-" * 50)
    print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    baseline = results['test3']['sqlite_cellid']
    for method, time in [('SQLite cellId', results['test3']['sqlite_cellid']),
                         ('SQLite Structured', results['test1']['sqlite_structured']),
                         ('MinMax99 Tree', results['test3']['minmax99_tree']),
                         ('Disk Hash', results['test3']['disk_hash'])]:
        speedup = baseline / time if time > 0 else 0
        print(f"{method:<25} {time:<15.4f} {speedup:<10.2f}x")
    print("-" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MinMax99 Comprehensive Benchmark System')
    parser.add_argument('--db', type=str, default='cells_1m.db', 
                       help='SQLite database file (default: cells_1m.db)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of random lookups for benchmarking (default: 10000)')
    parser.add_argument('--repeat', type=int, default=3,
                       help='Number of benchmark repetitions (default: 3)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit records loaded from DB (default: load all)')
    parser.add_argument('--cold-start', action='store_true',
                       help='Run cold start benchmarks (new connections/file handles per lookup)')
    parser.add_argument('--both', action='store_true',
                       help='Run both warm and cold start benchmarks')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database file not found: {args.db}")
        print("Please ensure your SQLite database exists.")
        sys.exit(1)
    
    try:
        if args.both:
            print("🔥 Running WARM benchmark...")
            warm_results = run_comprehensive_benchmark(
                db_path=args.db,
                samples=args.samples,
                repeat=args.repeat,
                limit=args.limit,
                cold_start=False
            )
            
            print("\n" + "="*80)
            print("❄️  Running COLD START benchmark...")
            cold_results = run_comprehensive_benchmark(
                db_path=args.db,
                samples=args.samples,
                repeat=args.repeat,
                limit=args.limit,
                cold_start=True
            )
            
            print(f"\n📊 Results saved:")
            print(f"  - Warm: minmax99_benchmark_warm.png")
            print(f"  - Cold: minmax99_benchmark_cold.png")
            
        else:
            results = run_comprehensive_benchmark(
                db_path=args.db,
                samples=args.samples,
                repeat=args.repeat,
                limit=args.limit,
                cold_start=args.cold_start
            )
            
            mode = "cold" if args.cold_start else "warm"
            print(f"\n📊 Results saved to: minmax99_benchmark_{mode}.png")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()