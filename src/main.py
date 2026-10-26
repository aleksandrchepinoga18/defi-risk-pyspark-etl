
from pathlib import Path

from pyspark.sql import SparkSession

print("=== PYTHON STARTED ===", flush=True)
# Путь к корню проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Путь к датасету
DATA_PATH = BASE_DIR / "data" / "dataset.parquet"


def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Risk Scoring ETL")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(DATA_PATH))

        print("\n=== SCHEMA ===")
        df.printSchema()

        print("\n=== NUMBER OF ROWS ===")
        print(df.count())

        print("\n=== FIRST 5 ROWS ===")
        df.show(5, truncate=False)

        print("\n=== NUMBER OF PARTITIONS ===")
        print(df.rdd.getNumPartitions())

        print("\n=== EXECUTION PLAN ===")
        df.groupBy("target").count().explain(mode="formatted")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
