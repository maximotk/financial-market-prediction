from __future__ import annotations

import json
import os
import joblib
from typing import Any, Dict, Optional

import pandas as pd


def ensure_directory(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_joblib_artifact(obj: Any, filepath: str) -> None:
    """
    Save Python object using joblib.
    """
    ensure_directory(os.path.dirname(filepath))
    joblib.dump(obj, filepath)


def load_joblib_artifact(filepath: str) -> Any:
    """
    Load Python object saved with joblib.
    """
    return joblib.load(filepath)


def save_json_artifact(payload: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON.
    """
    ensure_directory(os.path.dirname(filepath))

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default_converter)


def load_json_artifact(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_columns(feature_columns, filepath: str) -> None:
    """
    Save ordered feature column list.
    """
    payload = {"feature_columns": list(feature_columns)}
    save_json_artifact(payload, filepath)


def load_feature_columns(filepath: str):
    """
    Load ordered feature column list.
    """
    payload = load_json_artifact(filepath)
    return payload["feature_columns"]


def save_run_metadata(
    experiment_name: str,
    model_name: str,
    feature_set_name: str,
    n_features: int,
    metrics: Dict[str, Any],
    filepath: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save summary metadata for a run.
    """
    payload = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "n_features": n_features,
        "metrics": metrics,
    }

    if extra_metadata is not None:
        payload["extra_metadata"] = extra_metadata

    save_json_artifact(payload, filepath)


def save_final_artifacts(
    model: Any,
    config,
    feature_columns,
    experiment_name: str,
    artifacts_dir: str = "artifacts",
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save all artifacts needed for later inference.

    Saved artifacts:
        model.joblib
        config.json
        feature_columns.json
        metadata.json
    """
    run_dir = os.path.join(artifacts_dir, experiment_name)

    ensure_directory(run_dir)

    model_path = os.path.join(run_dir, "model.joblib")
    config_path = os.path.join(run_dir, "config.json")
    feature_columns_path = os.path.join(run_dir, "feature_columns.json")
    metadata_path = os.path.join(run_dir, "metadata.json")

    save_joblib_artifact(model, model_path)
    config.to_json(config_path)
    save_feature_columns(feature_columns, feature_columns_path)

    save_run_metadata(
        experiment_name=experiment_name,
        model_name=config.model.model_name,
        feature_set_name=config.features.feature_set_name,
        n_features=len(feature_columns),
        metrics=metrics or {},
        filepath=metadata_path,
        extra_metadata=extra_metadata,
    )

    return {
        "run_dir": run_dir,
        "model_path": model_path,
        "config_path": config_path,
        "feature_columns_path": feature_columns_path,
        "metadata_path": metadata_path,
    }


def load_final_artifacts(
    experiment_name: str,
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Load saved artifacts for inference.
    """
    run_dir = os.path.join(artifacts_dir, experiment_name)

    model_path = os.path.join(run_dir, "model.joblib")
    config_path = os.path.join(run_dir, "config.json")
    feature_columns_path = os.path.join(run_dir, "feature_columns.json")
    metadata_path = os.path.join(run_dir, "metadata.json")

    artifacts = {
        "run_dir": run_dir,
        "model_path": model_path,
        "config_path": config_path,
        "feature_columns_path": feature_columns_path,
        "metadata_path": metadata_path,
        "model": load_joblib_artifact(model_path),
        "config_dict": load_json_artifact(config_path),
        "feature_columns": load_feature_columns(feature_columns_path),
        "metadata": load_json_artifact(metadata_path),
    }

    return artifacts


def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    """
    Save DataFrame to CSV.
    """
    ensure_directory(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)


def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    """
    return pd.read_csv(filepath, **kwargs)


def _json_default_converter(obj: Any):
    """
    Convert numpy / datetime types for JSON serialization.
    """
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

    except Exception:
        pass

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(obj)