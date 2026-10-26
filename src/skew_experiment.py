
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"

SKEW_ROWS = 1_000_000
PARTITIONS = 16


def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Data Skew Experiment")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Читаем только необходимые столбцы
        original = (
            spark.read.parquet(str(DATA_PATH))
            .select("wallet_address", "borrow_timestamp")
        )

        # Создаём миллион событий одного кошелька
        # Не собираем миллион строк в Python:
        # Spark генерирует их распределённо.
        heavy_wallet = (
            spark.range(0, SKEW_ROWS, 1, PARTITIONS)
            .select(
                F.lit("HEAVY_WALLET").alias("wallet_address"),
                (
                    F.lit(1_600_000_000.0)
                    + F.col("id").cast("double") * 60.0
                ).alias("borrow_timestamp")
            )
        )

        # Объединяем с исходными данными
        df = original.unionByName(heavy_wallet)

        print("\n=== DATASET ===")
        print(f"Total rows: {df.count()}")

        # Перераспределяем по кошельку
        partitioned = df.repartition(
            PARTITIONS, "wallet_address"
        )

        # Считаем записи внутри каждой партиции
        distribution = (
            partitioned
            .withColumn("partition_id", F.spark_partition_id())
            .groupBy("partition_id")
            .count()
            .orderBy("partition_id")
        )

        print("\n=== PARTITION DISTRIBUTION ===")
        distribution.show(PARTITIONS)

        # Вычисляем оконную функцию
        window = (
            Window.partitionBy("wallet_address")
            .orderBy("borrow_timestamp")
        )

        result = partitioned.withColumn(
            "prev_timestamp",
            F.lag("borrow_timestamp").over(window)
        )

        print("\n=== WINDOW BENCHMARK ===")

        start = perf_counter()

        # Агрегация заставляет Spark вычислить lag()
        checksum = (
            result
            .agg(F.sum("prev_timestamp").alias("checksum"))
            .first()["checksum"]
        )

        elapsed = perf_counter() - start

        print(f"Checksum: {checksum}")
        print(f"Window execution time: {elapsed:.3f} sec")

        print("\nSpark UI:", spark.sparkContext.uiWebUrl)
        input("\nPress Enter to stop Spark...")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
