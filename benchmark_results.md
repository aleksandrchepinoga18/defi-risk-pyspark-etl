
# PySpark Performance Benchmark

## Dataset
- Rows: 442,961
- Shuffle partitions: 16
- Spark: 3.5.5
- Environment: Docker, local mode

## Results

| Method | Median execution time |
|---|---:|
| Automatic Exchange | 1.257 sec |
| Manual repartition | 1.047 sec |

Manual / Automatic ratio: 0.833

## Findings

Both execution plans contain:
- Exchange
- Sort
- Window

Manual repartition showed a lower median execution
time in this run, but the measurements varied.

The experiment does not yet establish a consistent
performance improvement.

## Next steps

- Repeat the benchmark without competing workloads
- Investigate data skew
- Compare Spark UI task duration and shuffle metrics
