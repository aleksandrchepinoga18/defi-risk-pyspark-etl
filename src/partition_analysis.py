
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("Spark Partition Analysis")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(DATA_PATH))

        # 1. Распределяем данные по кошелькам
        df_partitioned = df.repartition(
            16, "wallet_address"
        )

        # 2. Определяем номер партиции для каждой строки
        df_with_pid = df_partitioned.withColumn(
            "partition_id",
            F.spark_partition_id()
        )

        # 3. Считаем количество строк в каждой партиции
        partition_stats = (
            df_with_pid
            .groupBy("partition_id")
            .count()
            .orderBy("partition_id")
        )

        print("\n=== PARTITION DISTRIBUTION ===")
        partition_stats.show(16)

        # 4. Находим минимальную и максимальную партиции
        stats = partition_stats.agg(
            F.min("count").alias("min_rows"),
            F.max("count").alias("max_rows"),
            F.avg("count").alias("avg_rows")
        ).first()

        print("\n=== SKEW STATISTICS ===")
        print(f"Min rows: {stats['min_rows']}")
        print(f"Max rows: {stats['max_rows']}")
        print(f"Avg rows: {stats['avg_rows']:.2f}")
        print(
            f"Skew ratio (max/min): "
            f"{stats['max_rows'] / stats['min_rows']:.2f}"
        )
        print("\nSpark UI:", spark.sparkContext.uiWebUrl)
        input("\nPress Enter to stop Spark...")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
