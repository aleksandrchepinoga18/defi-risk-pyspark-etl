
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"

PARTITIONS = 16
SKEW_ROWS = 1_000_000
SALT_BUCKETS = 8


def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Salting Experiment")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 1. Читаем реальные данные
        original = (
            spark.read.parquet(str(DATA_PATH))
            .select("wallet_address")
        )

        # 2. Генерируем тяжёлый кошелёк
        heavy = (
            spark.range(0, SKEW_ROWS, 1, PARTITIONS)
            .select(
                F.lit("HEAVY_WALLET").alias("wallet_address"),
                F.col("id").alias("event_id")
            )
        )

        original = original.withColumn(
            "event_id", F.lit(None).cast("long")
        )

        df = original.unionByName(heavy).cache()

        # Материализуем cache
        print("\nTotal rows:", df.count())

        # 3. Обычная агрегация
        print("\n=== WITHOUT SALTING ===")

        start = perf_counter()

        normal = (
            df.groupBy("wallet_address")
            .count()
        )

        normal_checksum = (
            normal.agg(F.sum("count")).first()[0]
        )

        normal_time = perf_counter() - start

        print(f"Time: {normal_time:.3f} sec")
        print(f"Total rows checked: {normal_checksum}")

        # 4. Добавляем salt для тяжёлого ключа
        salted = df.withColumn(
            "salt",
            F.when(
                F.col("wallet_address") == "HEAVY_WALLET",
                F.pmod(F.col("event_id"), F.lit(SALT_BUCKETS))
            ).otherwise(F.lit(0))
        )

        # 5. Первая агрегация
        partial = (
            salted.groupBy("wallet_address", "salt")
            .count()
        )

        # 6. Вторая агрегация
        final = (
            partial.groupBy("wallet_address")
            .agg(F.sum("count").alias("count"))
        )

        print("\n=== WITH SALTING ===")

        start = perf_counter()

        salted_checksum = (
            final.agg(F.sum("count")).first()[0]
        )

        salted_time = perf_counter() - start

        print(f"Time: {salted_time:.3f} sec")
        print(f"Total rows checked: {salted_checksum}")

        # 7. Проверяем корректность
        # Сравниваем результаты для каждого кошелька
        comparison = (
            normal.alias("n")
            .join(
                final.alias("s"),
                on="wallet_address",
                how="full_outer"
            )
            .filter(
                F.col("n.count").isNull()
                | F.col("s.count").isNull()
                | (F.col("n.count") != F.col("s.count"))
            )
        )

        mismatches = comparison.count()

        print("\n=== DATA VALIDATION ===")
        print(f"Mismatched wallets: {mismatches}")

        assert mismatches == 0, "Salting produced incorrect results!"

        print("Validation passed!")
        assert normal_checksum == salted_checksum

        print("\n=== RESULTS ===")
        print(f"Normal: {normal_time:.3f} sec")
        print(f"Salted: {salted_time:.3f} sec")
        print(f"Ratio: {salted_time / normal_time:.3f}")

        print("\nSpark UI:", spark.sparkContext.uiWebUrl)
        input("\nPress Enter to stop Spark...")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
