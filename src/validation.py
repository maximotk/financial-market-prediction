from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features import build_label_next_return
from src.models import make_model, fit_model, predict_stacked, stacked_predictions_to_panel
from src.preprocessing import align_features_and_label, slice_features_by_date


def generate_rebalancing_dates(
    start_date: str,
    end_date: str,
    retrain_frequency: str = "ME",
    min_folds_before_start: int = 2,
) -> pd.DatetimeIndex:
    """
    Generate walk-forward rebalancing dates.

    Parameters
    ----------
    start_date : str
        Start date of the overall period.
    end_date : str
        End date of the overall period.
    retrain_frequency : str, default="ME"
        Pandas date_range frequency for rebalancing.
    min_folds_before_start : int, default=2
        Number of initial rebalance dates to skip, following the teacher notebook.

    Returns
    -------
    pd.DatetimeIndex
        Rebalancing dates used for fold endpoints.
    """
    rebalancing_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=retrain_frequency,
    )

    if min_folds_before_start > 0:
        rebalancing_dates = rebalancing_dates[min_folds_before_start:]

    return rebalancing_dates


def get_fold_dates(
    last_date_train_fold: pd.Timestamp,
    train_window_days: int,
    validation_window_days: int,
    prediction_delay_hours: int,
) -> Dict[str, pd.Timestamp]:
    """
    Build train/validation fold boundaries.

    Parameters
    ----------
    last_date_train_fold : pd.Timestamp
        End of the training window.
    train_window_days : int
        Length of rolling training window in days.
    validation_window_days : int
        Length of validation window in days.
    prediction_delay_hours : int
        Delay between end of training and start of prediction.

    Returns
    -------
    dict
        Dictionary with fold boundary timestamps.
    """
    start_date_train_fold = last_date_train_fold - pd.Timedelta(days=train_window_days)
    start_date_validate_fold = last_date_train_fold + pd.Timedelta(hours=prediction_delay_hours)
    last_date_validate_fold = last_date_train_fold + pd.Timedelta(days=validation_window_days)

    return {
        "start_date_train_fold": start_date_train_fold,
        "last_date_train_fold": last_date_train_fold,
        "start_date_validate_fold": start_date_validate_fold,
        "last_date_validate_fold": last_date_validate_fold,
    }


def run_walk_forward_validation(
    data: pd.DataFrame,
    features_processed: pd.DataFrame,
    config,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run teacher-style walk-forward validation.

    Workflow:
    - generate rebalance dates
    - for each fold:
        * define rolling train and validation windows
        * build training label
        * align training features and label
        * slice validation features
        * fit model
        * predict on validation fold
    - aggregate stacked predictions
    - convert to panel

    Parameters
    ----------
    data : pd.DataFrame
        Raw market data with multi-index columns.
    features_processed : pd.DataFrame
        Preprocessed stacked feature matrix.
    config : ExperimentConfig
        Full experiment config.
    verbose : bool, default=True
        Whether to print fold information.

    Returns
    -------
    dict
        Dictionary containing fold metadata and predictions.
    """
    rebalancing_dates = generate_rebalancing_dates(
        start_date=config.dates.start_date_train,
        end_date=config.dates.last_date_validate,
        retrain_frequency=config.walk_forward.retrain_frequency,
        min_folds_before_start=config.walk_forward.min_folds_before_start,
    )

    fold_predictions = []
    fold_details: List[Dict[str, object]] = []

    for last_date_train_fold in rebalancing_dates:
        fold_dates = get_fold_dates(
            last_date_train_fold=last_date_train_fold,
            train_window_days=config.walk_forward.train_window_days,
            validation_window_days=config.walk_forward.validation_window_days,
            prediction_delay_hours=config.walk_forward.prediction_delay_hours,
        )

        if verbose:
            print(
                f"Train: {fold_dates['start_date_train_fold']} -> {fold_dates['last_date_train_fold']}"
            )
            print(
                f"Predict: {fold_dates['start_date_validate_fold']} -> {fold_dates['last_date_validate_fold']}"
            )
            print("")

        label_fold = build_label_next_return(
            data=data,
            start_date=str(fold_dates["start_date_train_fold"]),
            end_date=str(fold_dates["last_date_train_fold"]),
        )

        X_train_fold, y_train_fold = align_features_and_label(
            features=features_processed,
            label=label_fold,
        )

        X_validate_fold = slice_features_by_date(
            features=features_processed,
            start_date=str(fold_dates["start_date_validate_fold"]),
            end_date=str(fold_dates["last_date_validate_fold"]),
        )

        if X_train_fold.empty or y_train_fold.empty or X_validate_fold.empty:
            if verbose:
                print("Skipping fold because one of X_train, y_train, or X_validate is empty.")
                print("")
            continue

        model = make_model(config.model)
        model = fit_model(model, X_train_fold, y_train_fold)
        predictions_fold = predict_stacked(model, X_validate_fold)

        fold_predictions.append(predictions_fold)

        fold_details.append(
            {
                "last_date_train_fold": last_date_train_fold,
                "start_date_train_fold": fold_dates["start_date_train_fold"],
                "start_date_validate_fold": fold_dates["start_date_validate_fold"],
                "last_date_validate_fold": fold_dates["last_date_validate_fold"],
                "n_train_rows": len(X_train_fold),
                "n_validate_rows": len(X_validate_fold),
            }
        )

    if len(fold_predictions) == 0:
        raise ValueError("No walk-forward fold predictions were generated.")

    predictions_stacked = pd.concat(fold_predictions).sort_index()
    predictions_stacked = predictions_stacked[~predictions_stacked.index.duplicated(keep="last")]

    predictions_panel = stacked_predictions_to_panel(predictions_stacked)
    predictions_panel = predictions_panel.dropna(axis=0, how="all")

    fold_details_df = pd.DataFrame(fold_details)

    return {
        "rebalancing_dates": rebalancing_dates,
        "fold_details": fold_details_df,
        "predictions_stacked": predictions_stacked,
        "predictions_panel": predictions_panel,
    }