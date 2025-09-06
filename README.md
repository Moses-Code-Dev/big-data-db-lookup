
## Big Data DB Lookup: MinMax99 Benchmark

This experimental repo benchmarks the custom MinMax99 hierarchical data structure against traditional database approaches (SQLite and disk-based hash tables) for large-scale cell data.

### Main Purpose

- **Benchmark**: Compare lookup performance between:
	- MinMax99 in-memory 4-level tree
	- SQLite (cellId and structured path indexes)
	- Disk-based hash table
- **Data**: Synthetic cell data (with hierarchical IDs) is generated at various scales (100k, 1M, 10M rows).

---

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd scripts
pip install -r requirements.txt
```

### 2. Generate a Database

Choose your desired size:

- **100,000 rows**:
	```bash
	python scripts/generate_db.py
	```
- **1,000,000 rows**:
	```bash
	python scripts/generate_1m.py
	```
- **10,000,000 rows**:
	```bash
	python scripts/generate_10m.py
	```

This will create a SQLite database file (e.g., `cells.db`, `cells_1m.db`, `cells_10m.db`).

### 3. Run the Benchmark

Point the benchmark script to your generated database:

```bash
python minmax99_benchmark.py --db cells_1m.db --samples 10000 --repeat 3
```

- `--db`: Path to your generated database file
- `--samples`: Number of random lookups to benchmark (default: 10000)
- `--repeat`: Number of repetitions for averaging (default: 3)
- `--cold-start`: (Optional) Simulate cold start (new connection/file handle per lookup)
- `--both`: (Optional) Run both warm and cold benchmarks

Example for cold start:
```bash
python minmax99_benchmark.py --db cells_1m.db --samples 5000 --cold-start
```

### 4. Results

- Benchmark results and summary tables are printed to the console.
- Performance graphs are saved as PNG files (e.g., `minmax99_benchmark_warm.png`).

---

## Notes

- All data is synthetic and generated using Faker.
- The MinMax99 structure is a 4-level nested dictionary optimized for hierarchical cellId lookups.
- The repo is for experimental/educational benchmarking only.
