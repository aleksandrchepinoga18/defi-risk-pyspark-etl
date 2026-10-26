
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"

spark = (
    SparkSession.builder
    .appName("Check Date Partitions")
    .master("local[4]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

try:
    df = spark.read.parquet(str(DATA_PATH))

    df = df.withColumn(
        "borrow_date",
        F.to_date(
            F.from_unixtime(
                F.col("borrow_timestamp").cast("long")
            )
        )
    )

    print("\n=== DATE STATISTICS ===")

    df.agg(
        F.countDistinct("borrow_date").alias("unique_dates"),
        F.min("borrow_date").alias("min_date"),
        F.max("borrow_date").alias("max_date")
    ).show()

finally:
    spark.stop()
