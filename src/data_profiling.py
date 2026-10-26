
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Data Profiling")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 1. Читаем Parquet
        df = spark.read.parquet(str(DATA_PATH))

        # 2. Размер датасета
        print("\n=== DATASET INFO ===")
        print(f"Rows: {df.count()}")
        print(f"Columns: {len(df.columns)}")
        print(f"Partitions: {df.rdd.getNumPartitions()}")

        # 3. Распределение целевой переменной
        print("\n=== TARGET DISTRIBUTION ===")
        df.groupBy("target").count().orderBy("target").show()

        # 4. Проверка дубликатов
        print("\n=== DUPLICATE WALLETS ===")

        duplicates = (
            df.groupBy("wallet_address")
            .count()
            .filter(F.col("count") > 1)
        )

        print("\n=== TOP 10 REPEATED WALLETS ===")

        (
            df.groupBy("wallet_address")
            .count()
            .orderBy(F.desc("count"))
            .show(10, truncate=False)
        )
        print(f"Wallets with duplicates: {duplicates.count()}")

        print("\n=== EXACT DUPLICATES ===")

        total_rows = df.count()
        unique_rows = df.distinct().count()

        print(f"Exact duplicate rows: {total_rows - unique_rows}")

  # Проверяем, есть ли кошельки с разными значениями target
        print("\n=== WALLETS WITH DIFFERENT TARGETS ===")

        conflicting_wallets = (
            df.groupBy("wallet_address")
            .agg(F.countDistinct("target").alias("unique_targets"))
            .filter(F.col("unique_targets") > 1)
            .count()
        )

        print(f"Wallets with different targets: {conflicting_wallets}")

        # Проверяем временные поля
        print("\n=== TIMESTAMP COLUMNS ===")

        for column in [
            "borrow_timestamp",
            "first_tx_timestamp",
            "last_tx_timestamp"
        ]:
            print(f"{column}: {df.schema[column].dataType}")

        # Смотрим историю одного кошелька
        print("\n=== WALLET HISTORY ===")

        wallet = "0x586e32930ac05127de429bd566eaa2758fcbd9bc"

        (
            df.filter(F.col("wallet_address") == wallet)
            .select(
                "wallet_address",
                "borrow_timestamp",
                "target"
            )
            .orderBy("borrow_timestamp")
            .show(15, truncate=False)
        )

        # 5. Анализ пропусков
        print("\n=== MISSING VALUES ===")

        null_counts = df.select([
            F.sum(
                F.when(F.col(c).isNull(), 1).otherwise(0)
            ).alias(c)
            for c in df.columns
        ])

        null_counts.show(truncate=False, vertical=True)

        # 6. План выполнения
        print("\n=== EXECUTION PLAN ===")

        df.groupBy("target").count().explain(mode="formatted")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
