from typing import Any

import pandas as pd
import sklearn.linear_model
from xgboost import XGBRegressor


def make_model(model_config) -> Any:
    """
    Create a model instance from configuration.

    Supported model names:
    - 'ols'
    - 'ridge'
    - 'lasso'
    - 'elastic_net'
    - 'xgboost'
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

    if model_name == "ridge":
        allowed_params = {
            "alpha",
            "fit_intercept",
            "copy_X",
            "max_iter",
            "tol",
            "solver",
            "random_state",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return sklearn.linear_model.Ridge(**filtered_params)

    if model_name == "lasso":
        allowed_params = {
            "alpha",
            "fit_intercept",
            "precompute",
            "copy_X",
            "max_iter",
            "tol",
            "warm_start",
            "positive",
            "random_state",
            "selection",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return sklearn.linear_model.Lasso(**filtered_params)

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

    if model_name == "xgboost":
        allowed_params = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "objective",
            "random_state",
            "n_jobs",
            "tree_method",
            "min_child_weight",
            "gamma",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return XGBRegressor(**filtered_params)

    if model_name == "bayesian_ridge":
        allowed_params = {
            "max_iter",
            "tol",
            "alpha_1",
            "alpha_2",
            "lambda_1",
            "lambda_2",
            "alpha_init",
            "lambda_init",
            "fit_intercept",
            "copy_X",
            "verbose",
        }
        filtered_params = {k: v for k, v in params.items() if k in allowed_params}
        return sklearn.linear_model.BayesianRidge(**filtered_params)

    raise ValueError(f"Unknown model_name: {model_config.model_name}")


def fit_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """
    Fit a model on stacked features and labels.
    """
    model.fit(X=X, y=y)
    return model


def predict_stacked(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Predict on a stacked feature matrix.
    """
    predictions = model.predict(X)
    predictions = pd.Series(predictions, index=X.index, name="prediction")
    return predictions


def stacked_predictions_to_panel(predictions: pd.Series) -> pd.DataFrame:
    """
    Convert stacked predictions into a datetime x instrument panel.
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
    """
    model = make_model(model_config)
    model = fit_model(model, X_train, y_train)
    predictions = predict_stacked(model, X_predict)
    return predictions