"""
PySpark feature engineering pipeline.

Builds the survival model feature matrix from the four raw tables.
All aggregations are windowed relative to a configurable `as_of_date`
so the same code runs for both training (historical) and batch scoring.

Output schema: one row per customer with all features + survival labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.utils.logger import get_logger
from src.utils.spark_utils import write_parquet

log = get_logger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates all feature transforms and joins them into a single
    modelling DataFrame.

    Usage:
        pipeline = FeatureEngineeringPipeline(spark, as_of_date="2024-06-30")
        features_df = pipeline.run(customers, transactions, clickstream, support)
    """

    def __init__(
        self,
        spark: SparkSession,
        as_of_date: str = "2024-06-30",
        lookback_windows: Optional[list[int]] = None,
        output_dir: str = "data/processed",
    ):
        self.spark = spark
        self.as_of_date = as_of_date
        self._as_of_ts = F.lit(as_of_date).cast("date")
        self.windows = lookback_windows or [30, 60, 90, 180, 365]
        self.output_dir = Path(output_dir)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(
        self,
        customers: DataFrame,
        transactions: DataFrame,
        clickstream: DataFrame,
        support_tickets: DataFrame,
        write_output: bool = True,
    ) -> DataFrame:
        """Build and return the full feature matrix."""
        log.info("Feature engineering started", as_of_date=self.as_of_date)

        # Filter all event tables to events on or before as_of_date
        txn = transactions.filter(F.col("transaction_date") <= self._as_of_ts)
        cs = clickstream.filter(F.col("event_timestamp").cast("date") <= self._as_of_ts)
        sup = support_tickets.filter(F.col("ticket_date") <= self._as_of_ts)

        # Build feature blocks
        static_feats   = self._static_features(customers)
        rfm_feats      = self._rfm_features(txn)
        windowed_feats  = self._windowed_transaction_features(txn)
        behavioral_feats = self._behavioral_features(cs)
        trend_feats    = self._session_trend_features(cs)
        support_feats  = self._support_features(sup)

        # Join everything
        df = (
            static_feats
            .join(rfm_feats,       on="customer_id", how="left")
            .join(windowed_feats,   on="customer_id", how="left")
            .join(behavioral_feats, on="customer_id", how="left")
            .join(trend_feats,      on="customer_id", how="left")
            .join(support_feats,    on="customer_id", how="left")
        )

        df = self._derived_features(df)
        df = self._fill_nulls(df)

        log.info("Feature matrix built", rows=df.count(), cols=len(df.columns))

        if write_output:
            out_path = str(self.output_dir / "features")
            write_parquet(df, out_path, mode="overwrite")

        return df

    # ── Static / demographic features ─────────────────────────────────────────

    def _static_features(self, customers: DataFrame) -> DataFrame:
        """Pass-through static columns + compute tenure."""
        return customers.select(
            "customer_id",
            "segment",
            "city_tier",
            "gender",
            "age_years",
            "acquisition_channel",
            "product_category",
            "product_price_inr",
            "warranty_months",
            "amc_active",
            "first_purchase_date",
            "observation_end_date",
            "is_churned",
            "duration_days",
        ).withColumn(
            "tenure_days",
            F.datediff(self._as_of_ts, F.col("first_purchase_date")),
        ).withColumn(
            "warranty_expiry_date",
            F.date_add(F.col("first_purchase_date"), F.col("warranty_months") * 30),
        ).withColumn(
            "warranty_remaining_days",
            F.datediff(F.col("warranty_expiry_date"), self._as_of_ts),
        ).withColumn(
            "is_warranty_expired",
            (F.col("warranty_remaining_days") < 0).cast(IntegerType()),
        )

    # ── RFM features ─────────────────────────────────────────────────────────

    def _rfm_features(self, txn: DataFrame) -> DataFrame:
        """Compute lifetime Recency, Frequency, Monetary features."""
        agg = txn.groupBy("customer_id").agg(
            F.count("transaction_id").alias("total_transactions"),
            F.sum("amount_inr").alias("total_spend_inr"),
            F.avg("amount_inr").alias("avg_transaction_value_inr"),
            F.max("transaction_date").alias("last_transaction_date"),
        )

        return agg.withColumn(
            "days_since_last_transaction",
            F.datediff(self._as_of_ts, F.col("last_transaction_date")),
        ).withColumn(
            "months_since_last_transaction",
            (F.col("days_since_last_transaction") / 30.44).cast(DoubleType()),
        ).drop("last_transaction_date")

    # ── Windowed transaction features ─────────────────────────────────────────

    def _windowed_transaction_features(self, txn: DataFrame) -> DataFrame:
        """Transaction counts and amounts in trailing N-day windows."""
        base = txn.select("customer_id", "transaction_date", "amount_inr")
        frames = []

        for window_days in [30, 60, 90, 180]:
            window_start = F.date_sub(self._as_of_ts, window_days)
            agg = (
                base.filter(F.col("transaction_date") >= window_start)
                .groupBy("customer_id")
                .agg(
                    F.count("*").alias(f"tx_count_{window_days}d"),
                    F.sum("amount_inr").alias(f"tx_amount_{window_days}d"),
                )
            )
            frames.append(agg)

        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, on="customer_id", how="outer")

        return result

    # ── Behavioural / clickstream features ────────────────────────────────────

    def _behavioral_features(self, cs: DataFrame) -> DataFrame:
        """Aggregate clickstream into per-customer behavioural metrics."""
        cs_dated = cs.withColumn("event_date", F.col("event_timestamp").cast("date"))

        # Trailing 30d and 90d session windows
        frames = []
        for window_days in [30, 90]:
            window_start = F.date_sub(self._as_of_ts, window_days)
            agg = (
                cs_dated.filter(F.col("event_date") >= window_start)
                .groupBy("customer_id")
                .agg(
                    F.countDistinct("event_date").alias(f"active_days_{window_days}d"),
                    F.count("*").alias(f"total_events_{window_days}d"),
                    F.avg("session_duration_s").alias(f"avg_session_dur_s_{window_days}d"),
                    F.avg("pages_in_session").alias(f"avg_pages_{window_days}d"),
                    F.sum(
                        (F.col("event_type") == "product_comparison").cast(IntegerType())
                    ).alias(f"product_comparison_events_{window_days}d"),
                    F.sum(
                        (F.col("event_type") == "wishlist_add").cast(IntegerType())
                    ).alias(f"wishlist_add_events_{window_days}d"),
                )
            )
            # Derive sessions-per-week from active_days
            agg = agg.withColumn(
                f"avg_weekly_sessions_{window_days}d",
                (F.col(f"active_days_{window_days}d") / (window_days / 7.0)).cast(DoubleType()),
            )
            frames.append(agg)

        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, on="customer_id", how="outer")

        # Last visit date → recency
        last_visit = cs_dated.groupBy("customer_id").agg(
            F.max("event_date").alias("last_web_visit_date")
        ).withColumn(
            "days_since_last_web_visit",
            F.datediff(self._as_of_ts, F.col("last_web_visit_date")),
        ).drop("last_web_visit_date")

        return result.join(last_visit, on="customer_id", how="outer")

    # ── Session trend features ─────────────────────────────────────────────────

    def _session_trend_features(self, cs: DataFrame) -> DataFrame:
        """
        Compute the linear slope of weekly session counts over trailing 90 days.
        A negative slope indicates declining engagement — strong churn signal.

        We approximate the slope as:
            slope = (sessions_last_30d/30 - sessions_first_30d/30) / (60/7)
        This avoids a full OLS regression inside Spark.
        """
        cs_dated = cs.withColumn("event_date", F.col("event_timestamp").cast("date"))
        as_of = self._as_of_ts

        early_start = F.date_sub(as_of, 90)
        early_end   = F.date_sub(as_of, 61)
        late_start  = F.date_sub(as_of, 30)

        early = (
            cs_dated
            .filter(
                (F.col("event_date") >= early_start) &
                (F.col("event_date") <= early_end)
            )
            .groupBy("customer_id")
            .agg(F.countDistinct("event_date").alias("early_active_days"))
        )

        late = (
            cs_dated
            .filter(F.col("event_date") >= late_start)
            .groupBy("customer_id")
            .agg(F.countDistinct("event_date").alias("late_active_days"))
        )

        trend = (
            early.join(late, on="customer_id", how="outer")
            .withColumn(
                "session_frequency_trend_90d",
                (
                    (F.coalesce(F.col("late_active_days"), F.lit(0)) / 30.0) -
                    (F.coalesce(F.col("early_active_days"), F.lit(0)) / 30.0)
                ).cast(DoubleType()),
            )
            .select("customer_id", "session_frequency_trend_90d")
        )
        return trend

    # ── Support ticket features ───────────────────────────────────────────────

    def _support_features(self, sup: DataFrame) -> DataFrame:
        """Aggregate support ticket metrics."""
        lifetime = sup.groupBy("customer_id").agg(
            F.count("ticket_id").alias("total_support_tickets"),
            F.avg("resolution_days").alias("avg_resolution_days"),
            F.avg("is_escalated").alias("escalation_rate"),
            F.avg("customer_satisfied").alias("avg_satisfaction"),
            F.max("ticket_date").alias("last_support_date"),
        ).withColumn(
            "days_since_last_support_contact",
            F.datediff(self._as_of_ts, F.col("last_support_date")),
        ).drop("last_support_date")

        frames = [lifetime]
        for window_days in [30, 90]:
            window_start = F.date_sub(self._as_of_ts, window_days)
            agg = (
                sup.filter(F.col("ticket_date") >= window_start)
                .groupBy("customer_id")
                .agg(F.count("ticket_id").alias(f"support_tickets_{window_days}d"))
            )
            frames.append(agg)

        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, on="customer_id", how="outer")

        return result

    # ── Derived / composite features ──────────────────────────────────────────

    def _derived_features(self, df: DataFrame) -> DataFrame:
        """Composite scores derived from base features."""
        df = df.withColumn(
            "rfm_recency_score",
            F.when(F.col("days_since_last_transaction") <= 30,  5)
             .when(F.col("days_since_last_transaction") <= 90,  4)
             .when(F.col("days_since_last_transaction") <= 180, 3)
             .when(F.col("days_since_last_transaction") <= 270, 2)
             .otherwise(1),
        ).withColumn(
            "rfm_frequency_score",
            F.when(F.col("total_transactions") >= 10, 5)
             .when(F.col("total_transactions") >= 6,  4)
             .when(F.col("total_transactions") >= 4,  3)
             .when(F.col("total_transactions") >= 2,  2)
             .otherwise(1),
        ).withColumn(
            "rfm_monetary_score",
            F.when(F.col("total_spend_inr") >= 200_000, 5)
             .when(F.col("total_spend_inr") >= 100_000, 4)
             .when(F.col("total_spend_inr") >= 60_000,  3)
             .when(F.col("total_spend_inr") >= 30_000,  2)
             .otherwise(1),
        ).withColumn(
            "rfm_score",
            F.col("rfm_recency_score") + F.col("rfm_frequency_score") + F.col("rfm_monetary_score"),
        )

        # Engagement score [0,1]: weighted combo of web activity + support satisfaction
        df = df.withColumn(
            "engagement_score",
            F.least(
                F.lit(1.0),
                F.greatest(
                    F.lit(0.0),
                    (
                        0.5 * F.coalesce(F.col("avg_weekly_sessions_90d"), F.lit(0.0)) / 5.0 +
                        0.3 * F.coalesce(F.col("avg_satisfaction"), F.lit(0.5)) +
                        0.2 * (1.0 - F.least(F.lit(1.0),
                               F.coalesce(F.col("days_since_last_transaction"), F.lit(365)) / 365.0))
                    ).cast(DoubleType()),
                ),
            ),
        )
        return df

    # ── Null filling ──────────────────────────────────────────────────────────

    def _fill_nulls(self, df: DataFrame) -> DataFrame:
        """
        Zero-fill numeric nulls for customers with no events in a window.
        String columns get a sentinel.
        """
        fill_zero_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, (DoubleType, IntegerType))
            and f.name not in ("customer_id", "is_churned", "duration_days", "age_years")
        ]
        fill_dict = {col: 0 for col in fill_zero_cols}
        return df.fillna(fill_dict)
