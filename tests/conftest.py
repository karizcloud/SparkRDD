"""Shared pytest fixtures: one Spark session for the whole test run."""

import pytest

from src.rdd_engine import get_spark_session, stop_spark_session


@pytest.fixture(scope="session")
def spark_context():
    spark = get_spark_session(app_name="SparkRDD-Lab-Tests", master="local[2]")
    yield spark.sparkContext
    stop_spark_session()
