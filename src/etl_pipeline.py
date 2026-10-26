
from pathlib import Path
import logging

from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "dataset.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# 1. EXTRACT
def extract_data(spark: SparkSession) -> DataFrame:
    logger.info("Reading source Parquet")
    return spark.read.parquet(str(INPUT_PATH))


# 2. TRANSFORM
def transform_data(df: DataFrame) -> DataFrame:
    logger.info("Starting transformations")

    # Преобразуем Unix timestamp в дату
    df = df.withColumn(
        "borrow_datetime",
        F.to_timestamp(
            F.from_unixtime(
                F.col("borrow_timestamp").cast("long")
            )
        )
    )

    # Дата наблюдения для партиционирования
    df = df.withColumn(
        "borrow_date",
        F.to_date("borrow_datetime")
    )
    # Месяц наблюдения для партиционирования
    df = df.withColumn(
        "borrow_month",
        F.date_format(F.col("borrow_date"), "yyyy-MM")
    )
    # Окно по кошельку и времени
    wallet_window = (
        Window.partitionBy("wallet_address")
        .orderBy("borrow_timestamp", "borrow_block_number")
    )

    # Предыдущая операция
    df = df.withColumn(
        "prev_borrow_timestamp",
        F.lag("borrow_timestamp").over(wallet_window)
    )

    # Время между операциями в часах
    df = df.withColumn(
        "hours_since_prev_borrow",
        (
            F.col("borrow_timestamp")
            - F.col("prev_borrow_timestamp")
        ) / 3600.0
    )

    # Отношение погашений к заимствованиям
    df = df.withColumn(
        "repay_to_borrow_ratio",
        F.when(
            F.col("borrow_amount_sum_eth") > 0,
            F.col("repay_amount_sum_eth")
            / F.col("borrow_amount_sum_eth")
        )
    )

    return df


# 3. VALIDATE
def validate_data(df: DataFrame) -> None:
    logger.info("Running data quality checks")

    total_rows = df.count()

    if total_rows == 0:
        raise ValueError("Dataset is empty")

    invalid_rows = df.filter(
        F.col("wallet_address").isNull()
        | F.col("borrow_timestamp").isNull()
        | F.col("borrow_date").isNull()
        | (~F.col("target").isin(0, 1))
        | F.col("target").isNull()
    ).count()

    if invalid_rows > 0:
        raise ValueError(
            f"Data quality failed: {invalid_rows} invalid rows"
        )

    logger.info("Validation passed. Rows: %s", total_rows)


# 4. LOAD
def load_data(df: DataFrame) -> None:
    logger.info("Writing processed Parquet")

    (
        df.write
        .mode("overwrite")
        .partitionBy("borrow_month")
        .parquet(str(OUTPUT_PATH))
    )

    logger.info("Parquet saved to %s", OUTPUT_PATH)


# MAIN
def main():
    spark = (
        SparkSession.builder
        .appName("DeFi Risk ETL Pipeline")
        .master("local[4]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        source_df = extract_data(spark)

        logger.info("Source rows: %s", source_df.count())

        transformed_df = transform_data(source_df)

        validate_data(transformed_df)

        load_data(transformed_df)

        logger.info("ETL completed successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
