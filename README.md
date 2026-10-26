
![PySpark ETL Tests](https://github.com/aleksandrchepinoga18/defi-risk-pyspark-etl/actions/workflows/tests.yml/badge.svg)

# DeFi Risk Scoring — PySpark ETL Pipeline

Production-style data processing pipeline for DeFi wallet risk scoring, built with Apache Spark, Python, Parquet and Docker.

The project demonstrates distributed data processing, window functions, partitioning, data quality validation and Spark performance analysis.

## 1. Project Overview

The pipeline processes a DeFi credit risk dataset containing 442,961 observations and 78 original features.

**Technology stack:**
- Python
- Apache Spark 3.5.5 / PySpark
- Docker
- Apache Parquet
- Pytest

**Main pipeline:**

`Raw Parquet → Extract → Transform → Validate → Partitioned Parquet`

The project focuses on data engineering rather than model training. The processed dataset can serve as input to a downstream credit risk model.

## 2. ETL Pipeline

The pipeline is implemented in `src/etl_pipeline.py`.

**Extract**

Reads the source dataset from `data/dataset.parquet`.

**Transform**

Performs feature engineering, including:

- Converting Unix timestamps to datetimes in UTC.
- Creating date and month fields.
- Computing previous borrowing timestamps with Spark window functions.
- Calculating time intervals between observations.
- Computing repayment-to-borrowing ratios.

**Validate**

Checks that:

- The resulting dataset is not empty.
- Wallet addresses are present.
- Borrowing timestamps and dates are valid.
- Target values belong to the expected binary classes.

**Load**

Writes processed data as Parquet, partitioned by borrowing month.

## 3. Data Profiling

Initial exploratory analysis identified:

| Metric | Value |
|---|---:|
| Observations | 442,961 |
| Original features | 78 |
| Wallets with multiple observations | 36,415 |
| Exact duplicate rows | 0 |
| Wallets with different target values | 9,742 |

Multiple observations per wallet are retained to preserve historical information.

## 4. Spark Performance Experiments

### Window Functions and Shuffle

Analyzed the physical execution plans of Spark window functions using `explain(mode="formatted")`.

Observed execution operators including:

- `Scan parquet`
- `Exchange`
- `Sort`
- `Window`

Compared automatic distribution with explicit repartitioning by wallet address.

### Benchmark Results

| Strategy | Median duration |
|---|---:|
| Automatic Exchange | 1.257 sec |
| Manual repartition | 1.047 sec |

The manual approach showed a lower median duration in this local benchmark. Results were variable and do not establish a consistent performance advantage.

### Data Skew Investigation

Introduced 1 million synthetic observations for a single wallet to simulate a heavily skewed key.

| Metric | Original dataset | Synthetic skew dataset |
|---|---:|---:|
| Rows | 442,961 | 1,442,961 |
| Partition count | 16 | 16 |
| Max/min partition size | 1.30 | 42.5 |

Spark UI showed an uneven task workload and disk spill under synthetic skew.

### Salting Experiment

Implemented a two-stage salted aggregation for the heavy wallet and compared it with standard `groupBy().count()`.

| Strategy | Duration |
|---|---:|
| Standard aggregation | 2.062 sec |
| Salted aggregation | 1.930 sec |

Validation found zero mismatched wallet counts among matched non-null keys.

These single-run measurements are illustrative, not evidence of a general speedup. Spark already performs partial aggregation for standard counting operations.

## 5. Parquet Storage Optimization

The initial daily partitioning strategy produced excessive partition granularity.

Analysis identified 1,604 distinct observation dates between May 2019 and September 2023.

The pipeline was changed to monthly partitioning.

**Measured output:**

| Metric | Result |
|---|---:|
| Monthly partitions | 53 |
| Parquet files | 265 |
| Total output size | ~129 MB |
| Average file size | ~489 KB |

The remaining small-files overhead is a potential future optimization.

## 6. Running the Project

### Requirements

- Docker Desktop
- Git Bash on Windows, or an equivalent terminal
- A local `data/dataset.parquet` input file

### Build the Docker image

```bash
docker build -t defi-spark .
```

### Run the ETL

From Git Bash on Windows:

```bash
docker run --rm \
  --mount "type=bind,source=$(pwd -W)/src,target=/app/src,readonly" \
  --mount "type=bind,source=$(pwd -W)/data,target=/app/data" \
  defi-spark \
  python3 -u src/etl_pipeline.py
```

The processed data will be written to `data/processed/`.

### Run automated tests

```bash
docker run --rm \
  --mount "type=bind,source=$(pwd -W)/src,target=/app/src,readonly" \
  --mount "type=bind,source=$(pwd -W)/tests,target=/app/tests,readonly" \
  defi-spark \
  python3 -m pytest -v tests/
```

Latest verified result: **2 tests passed**.

## 7. Learning Outcomes

This project provides hands-on experience with:

- Spark DataFrame and SQL execution planning
- Lazy evaluation and actions
- Window functions
- Shuffle and hash partitioning
- Spark UI diagnostics
- Data skew and salting
- Parquet partitioning and small-files analysis
- Data quality validation
- Docker-based reproducibility
- Automated ETL testing

## 8. Limitations

This is a local, Docker-based educational project, not a deployed production Spark cluster.

The source dataset contains precomputed wallet features rather than raw blockchain event logs. Therefore, the pipeline transforms existing features and constructs additional historical features.

Performance measurements were made on a local environment and should not be extrapolated to distributed production systems.

Future improvements include additional data quality checks, deterministic handling of timestamp ties, incremental processing, better file compaction and orchestration.