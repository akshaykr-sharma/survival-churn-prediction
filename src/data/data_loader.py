"""PySpark data loader — reads raw Parquet datasets and registers temp views."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Explicit schemas (faster + safer than schema inference) ───────────────────

CUSTOMERS_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("registration_date", DateType(), False),
        StructField("first_purchase_date", DateType(), False),
        StructField("segment", StringType(), False),
        StructField("city", StringType(), False),
        StructField("state", StringType(), False),
        StructField("city_tier", IntegerType(), False),
        StructField("gender", StringType(), False),
        StructField("age_years", IntegerType(), True),
        StructField("acquisition_channel", StringType(), False),
        StructField("product_category", StringType(), False),
        StructField("product_price_inr", DoubleType(), False),
        StructField("warranty_months", IntegerType(), False),
        StructField("amc_active", IntegerType(), False),
        StructField("is_churned", IntegerType(), False),
        StructField("churn_date", DateType(), True),
        StructField("observation_end_date", DateType(), False),
        StructField("duration_days", IntegerType(), False),
    ]
)

TRANSACTIONS_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("transaction_date", DateType(), False),
        StructField("transaction_type", StringType(), False),
        StructField("product_category", StringType(), True),
        StructField("amount_inr", DoubleType(), False),
        StructField("channel", StringType(), True),
        StructField("is_returned", IntegerType(), False),
    ]
)

CLICKSTREAM_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        # Read as Long (INT64) to handle both TIMESTAMP(NANOS,false) written by older
        # PyArrow and TIMESTAMP(MICROS) written by the fixed generator. Spark cannot
        # create a Parquet row-converter for nanosecond timestamps. We cast to
        # TimestampType below after loading.
        StructField("event_timestamp", LongType(), False),
        StructField("event_type", StringType(), False),
        StructField("session_duration_s", IntegerType(), False),
        StructField("pages_in_session", IntegerType(), False),
        StructField("device_type", StringType(), True),
        StructField("utm_source", StringType(), True),
    ]
)

SUPPORT_SCHEMA = StructType(
    [
        StructField("ticket_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("ticket_date", DateType(), False),
        StructField("category", StringType(), True),
        StructField("resolution_days", IntegerType(), False),
        StructField("is_escalated", IntegerType(), False),
        StructField("is_resolved", IntegerType(), False),
        StructField("customer_satisfied", IntegerType(), True),
        StructField("channel", StringType(), True),
    ]
)

_SCHEMA_MAP = {
    "customers": CUSTOMERS_SCHEMA,
    "transactions": TRANSACTIONS_SCHEMA,
    "clickstream": CLICKSTREAM_SCHEMA,
    "support_tickets": SUPPORT_SCHEMA,
}


class DataLoader:
    def __init__(self, spark: SparkSession, raw_data_dir: str = "data/raw"):
        self.spark = spark
        self.raw_dir = Path(raw_data_dir)

    def load(self, dataset: str, register_view: bool = True) -> DataFrame:
        """Load a raw Parquet dataset with its explicit schema."""
        path = str(self.raw_dir / dataset)
        schema = _SCHEMA_MAP.get(dataset)

        log.info("Loading dataset", dataset=dataset, path=path)
        reader = self.spark.read.format("parquet")
        if schema:
            reader = reader.schema(schema)

        df = reader.load(path)

        # event_timestamp is read as Long. Detect unit by magnitude:
        #   year 2024 in microseconds ≈ 1.7e15
        #   year 2024 in nanoseconds  ≈ 1.7e18
        # Threshold 1e17 cleanly separates them. Spark TimestampType cast expects
        # microseconds, so nanos must be divided by 1000.
        if dataset == "clickstream":
            sample_val = df.select(F.first("event_timestamp")).collect()[0][0]
            if sample_val is not None and sample_val > 1e17:
                # Nanosecond precision (old generator) → divide by 1000
                df = df.withColumn(
                    "event_timestamp",
                    (F.col("event_timestamp") / F.lit(1000)).cast(TimestampType()),
                )
            else:
                # Microsecond precision (fixed generator) → direct cast
                df = df.withColumn(
                    "event_timestamp",
                    F.col("event_timestamp").cast(TimestampType()),
                )

        if register_view:
            df.createOrReplaceTempView(dataset)
            log.info("Registered Spark temp view", view=dataset)

        log.info("Dataset loaded", dataset=dataset, count=df.count())
        return df

    def load_all(self) -> dict[str, DataFrame]:
        """Load all four raw datasets and return them keyed by name."""
        datasets = ["customers", "transactions", "clickstream", "support_tickets"]
        return {name: self.load(name) for name in datasets}

    def filter_train(
        self,
        df: DataFrame,
        date_col: str,
        train_cutoff: str = "2024-06-30",
    ) -> DataFrame:
        """Filter to training window (events <= train_cutoff)."""
        return df.filter(F.col(date_col) <= F.lit(train_cutoff))

    def filter_score(
        self,
        df: DataFrame,
        date_col: str,
        score_start: str = "2024-07-01",
    ) -> DataFrame:
        """Filter to scoring window (events > score_start)."""
        return df.filter(F.col(date_col) > F.lit(score_start))
