
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

# Позволяем импортировать функции из src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etl_pipeline import transform_data, validate_data


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("DeFi ETL Tests")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    yield session
    session.stop()


def test_window_and_ratio(spark):
    """Проверяем lag() и отношение погашений к займам."""

    data = [
        ("wallet_A", 1000.0, 1, 10.0, 5.0, 0),
        ("wallet_A", 4600.0, 2, 20.0, 10.0, 1),
        ("wallet_B", 2000.0, 3, 0.0, 0.0, 0),
    ]

    columns = [
        "wallet_address",
        "borrow_timestamp",
        "borrow_block_number",
        "borrow_amount_sum_eth",
        "repay_amount_sum_eth",
        "target",
    ]

    df = spark.createDataFrame(data, columns)

    result = transform_data(df)

    rows = {
        (r["wallet_address"], r["borrow_timestamp"]): r
        for r in result.collect()
    }

    # Первая операция кошелька A
    assert rows[("wallet_A", 1000.0)]["prev_borrow_timestamp"] is None

    # Вторая операция через 1 час
    assert rows[("wallet_A", 4600.0)]["hours_since_prev_borrow"] == 1.0

    # Отношение погашений к заимствованиям
    assert rows[("wallet_A", 4600.0)]["repay_to_borrow_ratio"] == 0.5

    # Проверяем деление на ноль
    assert rows[("wallet_B", 2000.0)]["repay_to_borrow_ratio"] is None


def test_invalid_target(spark):
    """Некорректный target должен вызвать ошибку."""

    data = [
        ("wallet_A", 1000.0, 1, 10.0, 5.0, 5)
    ]

    columns = [
        "wallet_address",
        "borrow_timestamp",
        "borrow_block_number",
        "borrow_amount_sum_eth",
        "repay_amount_sum_eth",
        "target",
    ]

    df = spark.createDataFrame(data, columns)
    transformed = transform_data(df)

    with pytest.raises(ValueError):
        validate_data(transformed)
