# Big Data DB Lookup

This repo experiments with querying large hierarchical datasets (like geo cells) 
using SQLite as a backend. The goal is to stress-test data retrieval 
before scaling to Android (Room DB).

## Features
- Generate synthetic cell data (~MinMax99 grid structure).
- Store in SQLite with indexed schema.
- Run lookup, range, and neighbor queries.
- Benchmark performance across dataset sizes.

## Steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
