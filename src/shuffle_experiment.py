
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("Spark Shuffle Experiment")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(DATA_PATH))

        window = (
            Window.partitionBy("wallet_address")
            .orderBy("borrow_timestamp")
        )

        print("\n=== EXPERIMENT 1: WITHOUT REPARTITION ===")

        result_1 = df.withColumn(
            "prev_timestamp",
            F.lag("borrow_timestamp").over(window)
        )

        result_1.explain(mode="formatted")

        print("\n=== EXPERIMENT 2: WITH REPARTITION ===")

        repartitioned = df.repartition(
            16, "wallet_address"
        )

        result_2 = repartitioned.withColumn(
            "prev_timestamp",
            F.lag("borrow_timestamp").over(window)
        )

        result_2.explain(mode="formatted")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
