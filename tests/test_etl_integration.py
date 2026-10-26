
import sys
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etl_pipeline import transform_data, validate_data


def test_etl_parquet_roundtrip(tmp_path):
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("ETL Integration Test")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    try:
        # Создаём небольшой тестовый датасет
        data = [
            ("wallet_A", 1600000000.0, 1, 10.0, 5.0, 0),
            ("wallet_A", 1600003600.0, 2, 20.0, 10.0, 1),
            ("wallet_B", 1600007200.0, 3, 5.0, 2.5, 0),
        ]

        columns = [
            "wallet_address",
            "borrow_timestamp",
            "borrow_block_number",
            "borrow_amount_sum_eth",
            "repay_amount_sum_eth",
            "target",
        ]

        source_df = spark.createDataFrame(data, columns)

        # Transform
        transformed_df = transform_data(source_df)

        # Validate
        validate_data(transformed_df)

        # Load
        output_path = str(tmp_path / "processed")

        (
            transformed_df.write
            .mode("overwrite")
            .partitionBy("borrow_month")
            .parquet(output_path)
        )

        # Read
        result_df = spark.read.parquet(output_path)

        # Проверяем количество записей
        assert result_df.count() == source_df.count()

        # Проверяем новые признаки
        assert "hours_since_prev_borrow" in result_df.columns
        assert "repay_to_borrow_ratio" in result_df.columns
        assert "borrow_month" in result_df.columns

        # Проверяем правильность lag()
        row = (
            result_df
            .filter(
                (result_df.wallet_address == "wallet_A")
                & (result_df.borrow_timestamp == 1600003600.0)
            )
            .first()
        )

        assert row["hours_since_prev_borrow"] == 1.0
        assert row["repay_to_borrow_ratio"] == 0.5

    finally:
        spark.stop()
