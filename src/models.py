from typing import Any

import pandas as pd
import sklearn.linear_model


def make_model(model_config) -> Any:
    """
    Create a model instance from configuration.

    Supported model names for the baseline skeleton:
    - 'ols'
    - 'elastic_net'

    Parameters
    ----------
    model_config : ModelConfig
        Model section from ExperimentConfig.

    Returns
    -------
    Any
        Instantiated model object.
    """
    model_name = model_config.model_name.lower()
    params = dict(model_config.params)

    if model_name == "ols":
        allowed_params = {
            "fit_intercept",
            "copy_X",
            "n_jobs",
            "positive",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return sklearn.linear_model.LinearRegression(**filtered_params)

    if model_name == "elastic_net":
        allowed_params = {
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "precompute",
            "max_iter",
            "copy_X",
            "tol",
            "warm_start",
            "positive",
            "random_state",
            "selection",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return sklearn.linear_model.ElasticNet(**filtered_params)

    raise ValueError(f"Unknown model_name: {model_config.model_name}")


def fit_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """
    Fit a model on stacked features and labels.

    Parameters
    ----------
    model : Any
        Model instance.
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training label.

    Returns
    -------
    Any
        Fitted model.
    """
    model.fit(X=X, y=y)
    return model


def predict_stacked(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Predict on a stacked feature matrix.

    Parameters
    ----------
    model : Any
        Fitted model.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.Series
        Predictions indexed like X.
    """
    predictions = model.predict(X)
    predictions = pd.Series(predictions, index=X.index, name="prediction")
    return predictions


def stacked_predictions_to_panel(predictions: pd.Series) -> pd.DataFrame:
    """
    Convert stacked predictions into a datetime x instrument panel.

    Parameters
    ----------
    predictions : pd.Series
        Stacked prediction series with MultiIndex (datetime, instrument).

    Returns
    -------
    pd.DataFrame
        Prediction panel.
    """
    return predictions.unstack()


def fit_and_predict(
    model_config,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
) -> pd.Series:
    """
    Convenience wrapper: create model, fit it, predict on new data.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    X_predict : pd.DataFrame
        Feature matrix to predict on.

    Returns
    -------
    pd.Series
        Stacked predictions.
    """
    model = make_model(model_config)
    model = fit_model(model, X_train, y_train)
    predictions = predict_stacked(model, X_predict)
    return predictions