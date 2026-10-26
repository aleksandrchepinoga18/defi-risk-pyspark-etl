
from pathlib import Path
from time import perf_counter
from statistics import median

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"

RUNS = 5


def run_benchmark(df, window, use_repartition):
    start = perf_counter()

    if use_repartition:
        df = df.repartition(16, "wallet_address")

    result = df.withColumn(
        "prev_timestamp",
        F.lag("borrow_timestamp").over(window)
    )

    # Записываем результат в память драйвера только как
    # агрегированное значение, чтобы выполнить окно.
    # Простого count() недостаточно:
    # Spark может исключить вычисление lag() из плана.
    checksum = (
        result
        .select(
            F.sum("prev_timestamp").alias("checksum")
        )
        .collect()[0]["checksum"]
    )

    elapsed = perf_counter() - start
    return elapsed, checksum


def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Spark Benchmark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(DATA_PATH))

        # Оставляем только необходимые столбцы
        df = df.select(
            "wallet_address",
            "borrow_timestamp"
        )

        window = (
            Window.partitionBy("wallet_address")
            .orderBy("borrow_timestamp")
        )

        results = {
            "automatic": [],
            "manual": []
        }

        print("\n=== WARM-UP ===")

        for use_repartition in [False, True]:
            run_benchmark(df, window, use_repartition)

        print("\n=== BENCHMARK ===")

        # Чередуем порядок, чтобы уменьшить влияние
        # прогрева JVM и файлового кеша.
        for i in range(RUNS):
            order = [False, True] if i % 2 == 0 else [True, False]

            for use_repartition in order:
                name = "manual" if use_repartition else "automatic"

                elapsed, checksum = run_benchmark(
                    df, window, use_repartition
                )

                results[name].append(elapsed)

                print(
                    f"Run {i + 1} | {name:9s} | "
                    f"{elapsed:.3f} sec | "
                    f"checksum={checksum}"
                )

        print("\n=== FINAL RESULTS ===")

        for name, times in results.items():
            print(
                f"{name:9s}: "
                f"median={median(times):.3f} sec | "
                f"all={[round(x, 3) for x in times]}"
            )

        ratio = (
            median(results["manual"])
            / median(results["automatic"])
        )

        print(f"\nManual / Automatic ratio: {ratio:.3f}")

        if ratio < 1:
            print("Manual repartition was faster in this experiment")
        elif ratio > 1:
            print("Automatic partitioning was faster in this experiment")
        else:
            print("Both variants had equal median times")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
