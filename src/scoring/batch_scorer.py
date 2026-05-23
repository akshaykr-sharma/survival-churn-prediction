"""
PySpark batch scoring pipeline.

Architecture:
    1. Load the champion Cox PH model from MLflow registry
    2. Load the feature matrix from the feature store (Parquet)
    3. Broadcast the model to all Spark executors (model is pandas-based)
    4. Apply model.predict via pandas_udf on partitions
    5. Write scored output (survival probs + risk segment) to Parquet

Why PySpark for scoring?
    The feature matrix for all 55K+ customers is joined from multiple
    large tables.  Spark handles that join efficiently.  The model itself
    runs in pandas inside each partition via pandas_udf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)

from src.utils.logger import get_logger
from src.utils.mlflow_utils import load_production_model
from src.utils.spark_utils import write_parquet

log = get_logger(__name__)

SCORE_SCHEMA = StructType([
    StructField("customer_id",        StringType(),  False),
    StructField("score_date",         StringType(),  False),
    StructField("survival_prob_30d",  DoubleType(),  True),
    StructField("survival_prob_60d",  DoubleType(),  True),
    StructField("survival_prob_90d",  DoubleType(),  True),
    StructField("survival_prob_180d", DoubleType(),  True),
    StructField("median_survival_days", DoubleType(), True),
    StructField("churn_risk_segment", StringType(),  True),
    StructField("partial_hazard",     DoubleType(),  True),
])


class BatchScorer:
    """
    Orchestrates the batch survival scoring pipeline.

    Usage:
        scorer = BatchScorer(spark, score_date="2024-12-31")
        scores_df = scorer.score(features_df)
        scorer.write_scores(scores_df, output_path)
    """

    def __init__(
        self,
        spark: SparkSession,
        score_date: str,
        model_name: str = "xyz_churn_cox_ph",
        model_stage: str = "Production",
        high_risk_threshold: float = 0.30,
        medium_risk_threshold: float = 0.60,
        output_dir: str = "data/processed/scores",
    ):
        self.spark = spark
        self.score_date = score_date
        self.model_name = model_name
        self.model_stage = model_stage
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.output_dir = Path(output_dir)
        self._model = None  # loaded lazily

    def _load_model(self):
        if self._model is None:
            log.info("Loading champion model from registry",
                     name=self.model_name, stage=self.model_stage)
            self._model = load_production_model(self.model_name, self.model_stage)
        return self._model

    # ── Main entry point ─────────────────────────────────────────────────────

    def score(self, features_df: DataFrame) -> DataFrame:
        """
        Score all customers in features_df.

        Strategy: collect feature partitions to pandas, score with the
        lifelines model, and return a Spark DataFrame.  For very large
        datasets (>1M rows), switch to mapInPandas (see _score_large).
        """
        log.info("Batch scoring started", score_date=self.score_date,
                 rows=features_df.count())

        model = self._load_model()

        # Convert to pandas (55K rows fits comfortably in driver memory)
        feature_pdf = features_df.toPandas()
        log.info("Feature matrix collected to driver", rows=len(feature_pdf))

        scores_pdf = self._score_pandas(feature_pdf, model)
        scores_df = self.spark.createDataFrame(scores_pdf, schema=SCORE_SCHEMA)

        log.info("Scoring complete", scored_rows=scores_df.count())
        return scores_df

    def score_large(self, features_df: DataFrame) -> DataFrame:
        """
        Alternative path for very large datasets using mapInPandas.
        The model is broadcast so each executor gets its own copy.
        """
        model = self._load_model()
        score_date = self.score_date
        high_t = self.high_risk_threshold
        mid_t = self.medium_risk_threshold

        # Broadcast model artifact
        model_bc = self.spark.sparkContext.broadcast(model)

        def score_partition(iterator):
            m = model_bc.value
            for pdf in iterator:
                yield _score_partition_pandas(pdf, m, score_date, high_t, mid_t)

        return features_df.mapInPandas(score_partition, schema=SCORE_SCHEMA)

    def write_scores(self, scores_df: DataFrame, output_path: Optional[str] = None) -> str:
        path = output_path or str(self.output_dir / f"score_date={self.score_date}")
        write_parquet(scores_df, path, mode="overwrite")
        log.info("Scores written", path=path)
        return path

    # ── Internal scoring logic ────────────────────────────────────────────────

    def _score_pandas(self, pdf: pd.DataFrame, model) -> pd.DataFrame:
        return _score_partition_pandas(
            pdf, model,
            self.score_date,
            self.high_risk_threshold,
            self.medium_risk_threshold,
        )


def _score_partition_pandas(
    pdf: pd.DataFrame,
    model,
    score_date: str,
    high_risk_threshold: float,
    medium_risk_threshold: float,
) -> pd.DataFrame:
    """
    Pure-function scoring logic — runs inside driver (small) or executor (large).
    Returns a pandas DataFrame matching SCORE_SCHEMA.
    """
    if len(pdf) == 0:
        return pd.DataFrame(columns=[f.name for f in SCORE_SCHEMA])

    # Get the underlying CoxPH object if wrapped by mlflow.pyfunc
    cox_model = _unwrap_model(model)

    # Predict survival function for all customers
    try:
        sf = cox_model.predict_survival_function(pdf)  # index=timeline, cols=row indices
    except Exception as exc:
        log.error("Survival function prediction failed", error=str(exc))
        raise

    # Extract S(t) at key horizons
    def get_sf_at(t: int) -> pd.Series:
        idx = sf.index.get_indexer([t], method="nearest")[0]
        return pd.Series(sf.iloc[idx].values, index=pdf.index)

    survival_30d  = get_sf_at(30)
    survival_60d  = get_sf_at(60)
    survival_90d  = get_sf_at(90)
    survival_180d = get_sf_at(180)

    median_days = cox_model.predict_median_survival(pdf)
    partial_haz  = cox_model.predict_partial_hazard(pdf)

    # Risk segmentation based on 90d survival probability
    def risk_segment(s90: float) -> str:
        if s90 < high_risk_threshold:
            return "High"
        elif s90 < medium_risk_threshold:
            return "Medium"
        return "Low"

    segments = survival_90d.apply(risk_segment)

    cust_ids = pdf.get("customer_id", pd.Series(range(len(pdf)), dtype=str))

    result = pd.DataFrame({
        "customer_id":          cust_ids.values,
        "score_date":           score_date,
        "survival_prob_30d":    survival_30d.values.astype(float),
        "survival_prob_60d":    survival_60d.values.astype(float),
        "survival_prob_90d":    survival_90d.values.astype(float),
        "survival_prob_180d":   survival_180d.values.astype(float),
        "median_survival_days": median_days.values.astype(float),
        "churn_risk_segment":   segments.values,
        "partial_hazard":       partial_haz.values.astype(float),
    })
    return result


def _unwrap_model(model):
    """Return the raw CoxPHModel regardless of whether it's mlflow-wrapped."""
    if hasattr(model, "_model_impl"):
        return model._model_impl.python_model
    if hasattr(model, "predict_survival_function"):
        return model
    # mlflow pyfunc wrapper
    if hasattr(model, "_model"):
        return model._model
    return model
