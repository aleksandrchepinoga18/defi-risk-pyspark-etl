
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("Window Keys Validation")
        .master("local[4]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(DATA_PATH))

        # Проверяем повторы комбинации ключей
        duplicates = (
            df.groupBy(
                "wallet_address",
                "borrow_timestamp",
                "borrow_block_number"
            )
            .count()
            .filter(F.col("count") > 1)
        )

        print("\n=== WINDOW KEY VALIDATION ===")

        duplicate_groups = duplicates.count()

        print(f"Duplicate key groups: {duplicate_groups}")

        print("\n=== TOP DUPLICATE GROUPS ===")

        duplicates.orderBy(
            F.desc("count")
        ).show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
