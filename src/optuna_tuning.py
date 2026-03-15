import copy
import pandas as pd
import optuna

from src.validation import run_walk_forward_validation
from src.evaluation import get_hourly_risk_free_rate, analyze_expected_returns
from src.data_utils import get_field_panel


def _score_config_walk_forward(
    data: pd.DataFrame,
    features_processed: pd.DataFrame,
    config,
) -> tuple[float, float]:
    """
    Returns:
        (walk_forward_sharpe_lag0, walk_forward_turnover)
    """
    wf_results = run_walk_forward_validation(
        data=data,
        features_processed=features_processed,
        config=config,
        verbose=False,
    )

    wf_pred_panel = wf_results["predictions_panel"]
    returns_panel = get_field_panel(data, "return")
    rfr_hourly = get_hourly_risk_free_rate(config.evaluation.risk_free_rate_annual)

    wf_stats = analyze_expected_returns(
        expected_returns=wf_pred_panel.loc[
            config.dates.start_date_validate:config.dates.last_date_validate
        ],
        returns=returns_panel.loc[
            config.dates.start_date_validate:config.dates.last_date_validate
        ],
        rfr_hourly=rfr_hourly,
        title=f"{config.experiment_name} - Optuna objective",
        lags=[0],
        tc=config.evaluation.transaction_cost,
        plot_option=config.evaluation.plot_option,
        output_stats=True,
    )

    sharpe = float(wf_stats.loc["Statistics", "sharpe"])
    turnover = float(wf_stats.loc["Statistics", "turnover"])
    return sharpe, turnover


def tune_elastic_net_optuna(
    data: pd.DataFrame,
    features_processed: pd.DataFrame,
    config,
    n_trials: int = 25,
    study_name: str | None = None,
    random_seed: int = 0,
):
    """
    Tune Elastic Net by maximizing walk-forward Sharpe at lag 0.
    """
    results = []

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(config)
        trial_config.model.model_name = "elastic_net"
        trial_config.model.params = {
            "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0),
            "fit_intercept": True,
            "tol": trial.suggest_float("tol", 1e-3, 5e-2, log=True),
            "selection": "random",
            "max_iter": trial.suggest_int("max_iter", 300, 1200, step=300),
            "random_state": random_seed,
        }

        sharpe, turnover = _score_config_walk_forward(
            data=data,
            features_processed=features_processed,
            config=trial_config,
        )

        results.append(
            {
                "trial": trial.number,
                "model_name": "elastic_net",
                "alpha": trial_config.model.params["alpha"],
                "l1_ratio": trial_config.model.params["l1_ratio"],
                "tol": trial_config.model.params["tol"],
                "max_iter": trial_config.model.params["max_iter"],
                "walk_forward_sharpe_lag0": sharpe,
                "walk_forward_turnover": turnover,
            }
        )
        return sharpe

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    results_df = pd.DataFrame(results).sort_values(
        "walk_forward_sharpe_lag0", ascending=False
    ).reset_index(drop=True)

    return study, results_df


def tune_bayesian_ridge_optuna(
    data: pd.DataFrame,
    features_processed: pd.DataFrame,
    config,
    n_trials: int = 25,
    study_name: str | None = None,
    random_seed: int = 0,
):
    """
    Tune Bayesian Ridge by maximizing walk-forward Sharpe at lag 0.
    """
    results = []

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(config)
        trial_config.model.model_name = "bayesian_ridge"
        trial_config.model.params = {
            "max_iter": trial.suggest_int("max_iter", 300, 1200, step=300),
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
            "alpha_1": trial.suggest_float("alpha_1", 1e-8, 1e-3, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-8, 1e-3, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-8, 1e-3, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-8, 1e-3, log=True),
            "fit_intercept": True,
            "copy_X": True,
            "verbose": False,
        }

        sharpe, turnover = _score_config_walk_forward(
            data=data,
            features_processed=features_processed,
            config=trial_config,
        )

        results.append(
            {
                "trial": trial.number,
                "model_name": "bayesian_ridge",
                "max_iter": trial_config.model.params["max_iter"],
                "tol": trial_config.model.params["tol"],
                "alpha_1": trial_config.model.params["alpha_1"],
                "alpha_2": trial_config.model.params["alpha_2"],
                "lambda_1": trial_config.model.params["lambda_1"],
                "lambda_2": trial_config.model.params["lambda_2"],
                "walk_forward_sharpe_lag0": sharpe,
                "walk_forward_turnover": turnover,
            }
        )
        return sharpe

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    results_df = pd.DataFrame(results).sort_values(
        "walk_forward_sharpe_lag0", ascending=False
    ).reset_index(drop=True)

    return study, results_df