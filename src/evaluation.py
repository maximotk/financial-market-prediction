from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_hourly_risk_free_rate(risk_free_rate_annual: float) -> float:
    """
    Convert an annual risk-free rate into an hourly rate.

    Parameters
    ----------
    risk_free_rate_annual : float
        Annual risk-free rate expressed in decimal form, e.g. 0.05 for 5%.

    Returns
    -------
    float
        Hourly risk-free rate.
    """
    return (1 + risk_free_rate_annual) ** (1 / (24 * 365)) - 1


def expected_returns_to_positions(expected_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expected returns into an investable cross-sectional portfolio.

    The logic follows the teacher's notebook:
    1) rank predictions cross-sectionally at each timestamp
    2) rescale by total absolute exposure
    3) demean to obtain a dollar-neutral portfolio

    Parameters
    ----------
    expected_returns : pd.DataFrame
        DataFrame indexed by datetime, columns are instruments.

    Returns
    -------
    pd.DataFrame
        Portfolio positions with same shape as expected_returns.
    """
    positions = expected_returns.rank(axis=1)
    positions = positions.div(positions.abs().sum(axis=1), axis=0)
    positions = positions.sub(positions.mean(axis=1), axis=0)
    return positions


def get_sharpe(
    pnl_portfolio: pd.Series,
    rfr_hourly: float,
    annualization_factor: int = 24 * 365,
) -> float:
    """
    Compute the annualized Sharpe ratio from hourly portfolio returns.

    Parameters
    ----------
    pnl_portfolio : pd.Series
        Time series of portfolio returns.
    rfr_hourly : float
        Hourly risk-free rate.
    annualization_factor : int, default=24*365
        Annualization factor for hourly data.

    Returns
    -------
    float
        Rounded Sharpe ratio.
    """
    pnl_portfolio = pnl_portfolio.dropna()

    if pnl_portfolio.empty:
        return np.nan

    excess_returns = pnl_portfolio - rfr_hourly
    std = excess_returns.std()

    if std == 0 or pd.isna(std):
        return np.nan

    sharpe_ratio = excess_returns.mean() / std * np.sqrt(annualization_factor)
    return round(float(sharpe_ratio), 2)


def pnl_analytics(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    rfr_hourly: float,
    lag: int,
    tc: float = 0.0,
    annualization_factor: int = 24 * 365,
) -> Dict[str, object]:
    """
    Compute strategy P&L analytics for a given execution lag.

    Gross P&L follows the teacher's logic:
        pnl = positions.shift(1 + lag) * returns

    Transaction costs are applied on position changes.

    Parameters
    ----------
    positions : pd.DataFrame
        Positions over time.
    returns : pd.DataFrame
        Realized asset returns over time.
    rfr_hourly : float
        Hourly risk-free rate.
    lag : int
        Additional execution lag in hours.
    tc : float, default=0.0
        Transaction cost per unit of turnover.
    annualization_factor : int, default=24*365
        Annualization factor for Sharpe.

    Returns
    -------
    Dict[str, object]
        Dictionary with Sharpe and pnl series.
    """
    common_index = positions.index.intersection(returns.index)
    common_columns = positions.columns.intersection(returns.columns)

    positions_aligned = positions.reindex(index=common_index, columns=common_columns)
    returns_aligned = returns.reindex(index=common_index, columns=common_columns)

    pnl = positions_aligned.shift(1 + lag).mul(returns_aligned).sum(axis=1)

    trades = positions_aligned.fillna(0).diff()
    costs = trades.abs().sum(axis=1) * tc
    pnl = pnl.sub(costs, fill_value=0)

    sharpe = get_sharpe(
        pnl_portfolio=pnl,
        rfr_hourly=rfr_hourly,
        annualization_factor=annualization_factor,
    )

    return {
        "sharpe": sharpe,
        "pnl": pnl,
    }


def compute_turnover(positions: pd.DataFrame, periods_per_day: int = 24) -> float:
    """
    Compute average daily turnover from hourly positions.

    Parameters
    ----------
    positions : pd.DataFrame
        Portfolio positions over time.
    periods_per_day : int, default=24
        Number of observations per day.

    Returns
    -------
    float
        Rounded daily turnover.
    """
    turnover = positions.fillna(0).diff().abs().sum(axis=1).mean()
    return round(float(turnover * periods_per_day), 2)


def analyze_expected_returns(
    expected_returns: pd.DataFrame,
    returns: pd.DataFrame,
    rfr_hourly: float,
    title: str = "a Nice Try",
    lags: Optional[List[int]] = None,
    tc: float = 0.0,
    plot_option: str = "matplotlib",
    output_stats: bool = False,
    annualization_factor: int = 24 * 365,
) -> Optional[pd.DataFrame]:
    """
    Analyze expected returns economically using teacher-style portfolio logic.

    Parameters
    ----------
    expected_returns : pd.DataFrame
        Model predictions or alpha values.
    returns : pd.DataFrame
        Realized asset returns.
    rfr_hourly : float
        Hourly risk-free rate.
    title : str, default="a Nice Try"
        Plot title.
    lags : list[int], optional
        Execution lags to test.
    tc : float, default=0.0
        Transaction cost level.
    plot_option : str, default="matplotlib"
        Currently supports "matplotlib".
    output_stats : bool, default=False
        Whether to return turnover and Sharpe summary.
    annualization_factor : int, default=24*365
        Annualization factor for Sharpe.

    Returns
    -------
    Optional[pd.DataFrame]
        Summary statistics if output_stats=True, else None.
    """
    if lags is None:
        lags = [0, 1, 2, 3, 6, 12]

    positions = expected_returns_to_positions(expected_returns)

    pnl_lags = {}
    for lag in lags:
        analytics_lag = pnl_analytics(
            positions=positions,
            returns=returns,
            rfr_hourly=rfr_hourly,
            lag=lag,
            tc=tc,
            annualization_factor=annualization_factor,
        )
        lag_label = f"Lag {lag}, sharpe={analytics_lag['sharpe']}"
        pnl_lags[lag_label] = analytics_lag["pnl"]

    pnl_lags = pd.concat(pnl_lags, axis=1).dropna(how="all")
    cumulative_pnl_lags = (1 + pnl_lags).cumprod().resample("24h").last()

    if plot_option == "matplotlib":
        ax = cumulative_pnl_lags.plot(
            title=f"Cumulative returns of {title}",
            logy=True,
            grid=False,
            figsize=(10, 4),
        )
        ax.set_xlabel("")
        plt.show()
    else:
        raise ValueError("plot_option must be 'matplotlib'.")

    if output_stats:
        statistics = {}
        statistics["turnover"] = compute_turnover(positions=positions, periods_per_day=24)
        statistics["sharpe"] = pnl_analytics(
            positions=positions,
            returns=returns,
            rfr_hourly=rfr_hourly,
            lag=lags[0],
            tc=tc,
            annualization_factor=annualization_factor,
        )["sharpe"]

        return pd.Series(statistics).to_frame("Statistics").T

    return None