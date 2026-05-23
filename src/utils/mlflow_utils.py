"""MLflow helpers — experiment setup, run logging, model registration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional

import mlflow
import mlflow.pyfunc
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

log = get_logger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Idempotently create experiment; return experiment_id."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        log.info("MLflow experiment created", name=experiment_name, id=experiment_id)
    else:
        experiment_id = experiment.experiment_id
        log.info("MLflow experiment found", name=experiment_name, id=experiment_id)

    mlflow.set_experiment(experiment_name)
    return experiment_id


@contextmanager
def mlflow_run(
    run_name: str,
    tags: Optional[dict] = None,
    nested: bool = False,
) -> Generator[Run, None, None]:
    """Context manager — wraps mlflow.start_run with logging."""
    _tags = {"project": "xyz-churn-survival", **(tags or {})}
    with mlflow.start_run(run_name=run_name, tags=_tags, nested=nested) as run:
        log.info("MLflow run started", run_name=run_name, run_id=run.info.run_id)
        try:
            yield run
        except Exception as exc:
            mlflow.set_tag("run_status", "FAILED")
            log.error("MLflow run failed", run_id=run.info.run_id, error=str(exc))
            raise
        else:
            mlflow.set_tag("run_status", "SUCCESS")
            log.info("MLflow run finished", run_id=run.info.run_id)


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of parameters (auto-stringify non-string values)."""
    str_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(str_params)


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact_path(local_path: str, artifact_path: Optional[str] = None) -> None:
    mlflow.log_artifact(local_path, artifact_path=artifact_path)


def register_model(
    model_uri: str,
    registered_model_name: str,
    stage: str = "Staging",
    description: Optional[str] = None,
) -> str:
    """Register a model version and transition it to the requested stage."""
    client = MlflowClient()
    result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = result.version

    if description:
        client.update_model_version(
            name=registered_model_name,
            version=version,
            description=description,
        )

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    log.info(
        "Model registered",
        name=registered_model_name,
        version=version,
        stage=stage,
    )
    return version


def load_production_model(registered_model_name: str, stage: str = "Production"):
    """Load the champion model from the registry."""
    model_uri = f"models:/{registered_model_name}/{stage}"
    log.info("Loading model from registry", uri=model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def get_best_run(
    experiment_name: str,
    metric: str = "concordance_index",
    higher_is_better: bool = True,
) -> Optional[Run]:
    """Return the run with the best value of `metric` in the experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    return runs[0] if runs else None
