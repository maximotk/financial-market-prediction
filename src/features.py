from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_utils import get_field_panel

import os
from typing import Dict, List, Optional

# ============================================================
# Core helpers
# ============================================================

def _get_base_index(data: pd.DataFrame) -> pd.MultiIndex:
    """
    Return the canonical stacked index used across all features.
    """
    return data["return"].stack().index


def _stack_panel(panel: pd.DataFrame, base_index: pd.MultiIndex) -> pd.Series:
    """
    Stack a datetime x instrument panel to the canonical stacked index.
    """
    return panel.stack().reindex(base_index)


def _safe_pct_change(panel: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Percent change without forward-filling.
    """
    return panel.pct_change(periods=periods, fill_method=None)


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """
    Safe elementwise division that avoids inf values.
    """
    out = numerator.div(denominator.replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)


def _ewm_zscore(panel: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """
    Exponentially weighted z-score.
    """
    ewm_mean = panel.ewm(halflife=halflife, min_periods=2).mean()
    ewm_std = panel.ewm(halflife=halflife, min_periods=2).std()
    z = (panel - ewm_mean).div(ewm_std.replace(0, np.nan))
    return z.replace([np.inf, -np.inf], np.nan)


def _macd_ratio(panel: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    MACD-style normalized difference of moving averages.
    """
    short_ma = panel.rolling(short_window, min_periods=1).mean()
    long_ma = panel.rolling(long_window, min_periods=1).mean()
    macd = _safe_divide(short_ma - long_ma, long_ma)
    return macd


def _distance_to_mean(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Ratio to rolling mean.
    """
    rolling_mean = panel.rolling(window, min_periods=1).mean()
    dist = _safe_divide(panel, rolling_mean)
    return dist


def _cross_sectional_rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Optional helper for alpha-style panels if needed later.
    """
    return panel.rank(axis=1, pct=True)


def _merge_feature_dicts(feature_dicts: List[Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
    """
    Merge multiple feature dictionaries.
    """
    merged: Dict[str, pd.Series] = {}
    for d in feature_dicts:
        overlap = set(merged).intersection(d)
        if overlap:
            raise ValueError(f"Duplicate feature names detected: {sorted(overlap)}")
        merged.update(d)
    return merged


def _feature_dict_to_dataframe(feature_dict: Dict[str, pd.Series], base_index: pd.MultiIndex) -> pd.DataFrame:
    """
    Convert a dict of stacked feature series into a DataFrame.
    """
    features = pd.DataFrame(index=base_index)
    for feature_name, feature_series in feature_dict.items():
        features[feature_name] = feature_series.reindex(base_index)
    return features


# ============================================================
# Baseline statistical features
# ============================================================

def _compute_baseline_feature_from_panel(raw_data: pd.DataFrame, feature_style: str) -> pd.DataFrame:
    """
    Baseline statistical transformations, close to the original teacher notebook.
    """
    if feature_style == "level":
        return raw_data

    if feature_style == "delta_1":
        return _safe_pct_change(raw_data, 1)

    if feature_style == "delta_6":
        return _safe_pct_change(raw_data, 6)

    if feature_style == "shift_1":
        return raw_data.shift(1)

    if feature_style == "shift_3":
        return raw_data.shift(3)

    if feature_style == "shift_6":
        return raw_data.shift(6)

    if feature_style == "shift_24":
        return raw_data.shift(24)

    if feature_style == "mean_6":
        return raw_data.dropna(axis=1, how="all").rolling(6, min_periods=1).mean()

    if feature_style == "mean_24":
        return raw_data.dropna(axis=1, how="all").rolling(24, min_periods=1).mean()

    if feature_style == "mean_120":
        return raw_data.dropna(axis=1, how="all").rolling(120, min_periods=1).mean()

    if feature_style == "std_6":
        return raw_data.dropna(axis=1, how="all").rolling(6, min_periods=2).std()

    if feature_style == "std_24":
        return raw_data.dropna(axis=1, how="all").rolling(24, min_periods=2).std()

    if feature_style == "std_120":
        return raw_data.dropna(axis=1, how="all").rolling(120, min_periods=2).std()

    if feature_style == "skew_6":
        return raw_data.dropna(axis=1, how="all").rolling(6, min_periods=3).skew()

    if feature_style == "skew_24":
        return raw_data.dropna(axis=1, how="all").rolling(24, min_periods=3).skew()

    if feature_style == "skew_120":
        return raw_data.dropna(axis=1, how="all").rolling(120, min_periods=3).skew()

    if feature_style == "kurt_6":
        return raw_data.dropna(axis=1, how="all").rolling(6, min_periods=4).kurt()

    if feature_style == "kurt_24":
        return raw_data.dropna(axis=1, how="all").rolling(24, min_periods=4).kurt()

    if feature_style == "kurt_120":
        return raw_data.dropna(axis=1, how="all").rolling(120, min_periods=4).kurt()

    raise ValueError(f"Unknown baseline feature_style: {feature_style}")


def build_baseline_features(
    data: pd.DataFrame,
    raw_fields: List[str],
    feature_styles: List[str],
) -> pd.DataFrame:
    """
    Original baseline feature family.
    """
    base_index = _get_base_index(data)
    feature_dict: Dict[str, pd.Series] = {}

    for field in raw_fields:
        raw_panel = get_field_panel(data, field)

        for feature_style in feature_styles:
            panel = _compute_baseline_feature_from_panel(raw_panel, feature_style)
            feature_name = f"{field}_{feature_style}"
            feature_dict[feature_name] = _stack_panel(panel, base_index)

    return _feature_dict_to_dataframe(feature_dict, base_index)


# ============================================================
# Teacher-inspired alpha families
# ============================================================

def build_open_interest_reversal_features(
    data: pd.DataFrame,
    windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Open-interest-driven reversal features inspired directly by the teacher notebook.
    """
    if windows is None:
        windows = {
            "delta_fast": 1,
            "delta_slow": 6,
            "mean_window": 3,
            "z_halflife": 6,
            "macd_short": 2,
            "macd_long": 12,
        }

    base_index = _get_base_index(data)

    oi = get_field_panel(data, "open_interest")
    oi_value = get_field_panel(data, "open_interest_value")

    feature_dict: Dict[str, pd.Series] = {}

    # Raw reversal deltas
    feature_dict["alpha_oi_delta_1_rev"] = _stack_panel(-_safe_pct_change(oi, windows["delta_fast"]), base_index)
    feature_dict["alpha_oi_delta_6_rev"] = _stack_panel(-_safe_pct_change(oi, windows["delta_slow"]), base_index)
    feature_dict["alpha_oi_value_delta_1_rev"] = _stack_panel(-_safe_pct_change(oi_value, windows["delta_fast"]), base_index)
    feature_dict["alpha_oi_value_delta_6_rev"] = _stack_panel(-_safe_pct_change(oi_value, windows["delta_slow"]), base_index)

    # Level / mean reversion flavor
    feature_dict["alpha_oi_value_level_rev"] = _stack_panel(-oi_value, base_index)
    feature_dict["alpha_oi_value_dist_mean_rev"] = _stack_panel(
        -_distance_to_mean(oi_value, windows["mean_window"]),
        base_index,
    )

    # EWM z-score reversion
    feature_dict["alpha_oi_value_zscore_rev"] = _stack_panel(
        -_ewm_zscore(oi_value, halflife=windows["z_halflife"]),
        base_index,
    )
    feature_dict["alpha_oi_zscore_rev"] = _stack_panel(
        -_ewm_zscore(oi, halflife=windows["z_halflife"]),
        base_index,
    )

    # MACD-style reversion
    feature_dict["alpha_oi_value_macd_rev"] = _stack_panel(
        -_macd_ratio(oi_value, windows["macd_short"], windows["macd_long"]),
        base_index,
    )
    feature_dict["alpha_oi_macd_rev"] = _stack_panel(
        -_macd_ratio(oi, windows["macd_short"], windows["macd_long"]),
        base_index,
    )

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_price_reversal_features(
    data: pd.DataFrame,
    reversal_windows: Optional[List[int]] = None,
    momentum_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Price-return-based reversal / momentum features.
    """
    if reversal_windows is None:
        reversal_windows = [1, 3, 6, 24]

    if momentum_windows is None:
        momentum_windows = [24, 72]

    base_index = _get_base_index(data)
    returns = get_field_panel(data, "return")
    close = get_field_panel(data, "close")

    feature_dict: Dict[str, pd.Series] = {}

    # Simple return reversal
    for w in reversal_windows:
        if w == 1:
            panel = -returns
            feature_name = "alpha_return_1_rev"
        else:
            panel = -returns.rolling(w, min_periods=1).sum()
            feature_name = f"alpha_return_{w}_rev"
        feature_dict[feature_name] = _stack_panel(panel, base_index)

    # Medium-horizon momentum flavor
    for w in momentum_windows:
        panel = close.pct_change(w, fill_method=None)
        feature_name = f"alpha_price_mom_{w}"
        feature_dict[feature_name] = _stack_panel(panel, base_index)

    # Distance to short-term mean
    feature_dict["alpha_price_dist_mean_24"] = _stack_panel(
        _distance_to_mean(close, 24) - 1.0,
        base_index,
    )

    # Price z-score
    feature_dict["alpha_price_zscore_24"] = _stack_panel(
        _ewm_zscore(close, halflife=24),
        base_index,
    )

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_volume_conditioned_features(
    data: pd.DataFrame,
    volume_window: int = 24 * 10,
) -> pd.DataFrame:
    """
    Volume-conditioned open-interest reversal features, directly inspired by the teacher notebook.
    """
    base_index = _get_base_index(data)

    oi = get_field_panel(data, "open_interest")
    volume_usd = get_field_panel(data, "volume_usd")

    delta_oi = -_safe_pct_change(oi, 1)

    # Time-series relative volume: lower-than-usual volume may amplify reversion
    relative_volume_ts = _safe_divide(
        volume_usd,
        volume_usd.rolling(volume_window, min_periods=1).mean().shift(1),
    )

    # Cross-sectional / rolling level proxy used by teacher
    relative_volume_xs = volume_usd.rolling(volume_window, min_periods=1).mean()

    combined_volume = relative_volume_ts * relative_volume_xs

    feature_dict: Dict[str, pd.Series] = {}

    feature_dict["alpha_oi_rev_vol_ts"] = _stack_panel(
        _safe_divide(delta_oi, relative_volume_ts),
        base_index,
    )
    feature_dict["alpha_oi_rev_vol_xs"] = _stack_panel(
        _safe_divide(delta_oi, relative_volume_xs),
        base_index,
    )
    feature_dict["alpha_oi_rev_vol_xsts"] = _stack_panel(
        _safe_divide(delta_oi, combined_volume),
        base_index,
    )

    # "Sell back the alpha" / chasing smart flows idea
    delta_oi_flow = delta_oi.sub(delta_oi.shift(1))
    feature_dict["alpha_oi_flow_chase_vol_xsts"] = _stack_panel(
        _safe_divide(delta_oi_flow, combined_volume),
        base_index,
    )

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_funding_open_interest_features(
    data: pd.DataFrame,
    windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Funding + positioning interaction features for crypto.
    """
    if windows is None:
        windows = {
            "funding_mean": 24,
            "funding_z_halflife": 24,
            "oi_delta": 1,
        }

    base_index = _get_base_index(data)

    funding = get_field_panel(data, "funding_rate")
    oi = get_field_panel(data, "open_interest")
    oi_value = get_field_panel(data, "open_interest_value")
    volume_usd = get_field_panel(data, "volume_usd")

    oi_delta = _safe_pct_change(oi, windows["oi_delta"])
    oi_value_delta = _safe_pct_change(oi_value, windows["oi_delta"])

    funding_mean = funding.rolling(windows["funding_mean"], min_periods=1).mean()
    funding_z = _ewm_zscore(funding, halflife=windows["funding_z_halflife"])

    relative_volume_ts = _safe_divide(
        volume_usd,
        volume_usd.rolling(24 * 10, min_periods=1).mean().shift(1),
    )

    feature_dict: Dict[str, pd.Series] = {}

    # Funding as direct signal
    feature_dict["alpha_funding_level"] = _stack_panel(funding, base_index)
    feature_dict["alpha_funding_mean_24"] = _stack_panel(funding_mean, base_index)
    feature_dict["alpha_funding_zscore_24"] = _stack_panel(funding_z, base_index)

    # Funding + OI interaction
    feature_dict["alpha_funding_x_oi_delta"] = _stack_panel(funding * oi_delta, base_index)
    feature_dict["alpha_funding_x_oi_value_delta"] = _stack_panel(funding * oi_value_delta, base_index)

    # Contrarian funding + positioning pressure
    feature_dict["alpha_contrarian_funding_oi"] = _stack_panel(-(funding_z * oi_delta), base_index)
    feature_dict["alpha_contrarian_funding_oi_value"] = _stack_panel(-(funding_z * oi_value_delta), base_index)

    # Volume-adjusted interaction
    feature_dict["alpha_contrarian_funding_oi_vol"] = _stack_panel(
        _safe_divide(-(funding_z * oi_delta), relative_volume_ts),
        base_index,
    )

    return _feature_dict_to_dataframe(feature_dict, base_index)


# ============================================================
# Combined families
# ============================================================

def build_teacher_core_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Strong first high-signal family:
    open interest reversal + price reversal + volume conditioning.
    """
    base_index = _get_base_index(data)

    feature_dict = _merge_feature_dicts([
        build_open_interest_reversal_features(data).to_dict("series"),
        build_price_reversal_features(data).to_dict("series"),
        build_volume_conditioned_features(data).to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_teacher_plus_baseline_features(
    data: pd.DataFrame,
    raw_fields: List[str],
    feature_styles: List[str],
) -> pd.DataFrame:
    """
    Combined family: generic baseline + teacher-inspired alpha families.
    """
    base_index = _get_base_index(data)

    feature_dict = _merge_feature_dicts([
        build_baseline_features(data, raw_fields, feature_styles).to_dict("series"),
        build_open_interest_reversal_features(data).to_dict("series"),
        build_price_reversal_features(data).to_dict("series"),
        build_volume_conditioned_features(data).to_dict("series"),
        build_funding_open_interest_features(data).to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)














def build_teacher_alpha_exact_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exact or near-exact teacher in-class alpha research family.
    """
    base_index = _get_base_index(data)

    returns = get_field_panel(data, "return")
    oi = get_field_panel(data, "open_interest")
    oi_value = get_field_panel(data, "open_interest_value")
    volume_usd = get_field_panel(data, "volume_usd")

    feature_dict: Dict[str, pd.Series] = {}

    # 1) Open interest value level
    alpha_oi_value_level = -oi_value
    feature_dict["teacher_oi_value_level_rev"] = _stack_panel(alpha_oi_value_level, base_index)

    # 2) Open interest value pct-change(1)
    alpha_oi_value_pct_1 = -oi_value.pct_change(1, fill_method=None)
    feature_dict["teacher_oi_value_pct_1_rev"] = _stack_panel(alpha_oi_value_pct_1, base_index)

    # 3) Open interest value diff(1)
    alpha_oi_value_diff_1 = -oi_value.diff(1)
    feature_dict["teacher_oi_value_diff_1_rev"] = _stack_panel(alpha_oi_value_diff_1, base_index)

    # 4) Distance to mean (window=3)
    distance_to_mean = oi_value / oi_value.rolling(3, min_periods=1).mean()
    alpha_dist_mean = -distance_to_mean
    feature_dict["teacher_oi_value_dist_mean_3_rev"] = _stack_panel(alpha_dist_mean, base_index)

    # 5) Z-score with halflife=1
    z_score_hl = 1
    z_score = (
        oi_value - oi_value.ewm(halflife=z_score_hl, min_periods=2).mean()
    ) / oi_value.ewm(halflife=z_score_hl, min_periods=2).std()
    alpha_zscore = -z_score.replace([np.inf, -np.inf], np.nan)
    feature_dict["teacher_oi_value_zscore_hl1_rev"] = _stack_panel(alpha_zscore, base_index)

    # 6) MACD trend reversion, short=2, long=12
    st_w = 2
    lt_w = 12
    short_term_ma = oi_value.rolling(st_w, min_periods=1).mean()
    long_term_ma = oi_value.rolling(lt_w, min_periods=1).mean()
    alpha_macd = -((short_term_ma - long_term_ma) / long_term_ma.replace(0, np.nan))
    alpha_macd = alpha_macd.replace([np.inf, -np.inf], np.nan)
    feature_dict["teacher_oi_value_macd_2_12_rev"] = _stack_panel(alpha_macd, base_index)

    # 7) Alternative variable: open_interest pct-change(1)
    delta_oi = -oi.pct_change(1, fill_method=None)
    feature_dict["teacher_oi_pct_1_rev"] = _stack_panel(delta_oi, base_index)

    # 8) Classic price reversal
    classic_reversal = -returns
    feature_dict["teacher_price_reversal_1"] = _stack_panel(classic_reversal, base_index)

    # 9) Volume-conditioned variants
    w_volume = 24 * 10

    relative_volume_ts = (
        volume_usd / volume_usd.rolling(w_volume, min_periods=1).mean().shift(1)
    )
    relative_volume_ts = relative_volume_ts.replace([np.inf, -np.inf], np.nan)

    relative_volume_xs = volume_usd.rolling(w_volume, min_periods=1).mean()
    relative_volume_xs = relative_volume_xs.replace([np.inf, -np.inf], np.nan)

    alpha_vol_ts = delta_oi / relative_volume_ts.replace(0, np.nan)
    alpha_vol_xs = delta_oi / relative_volume_xs.replace(0, np.nan)
    alpha_vol_xsts = delta_oi / (relative_volume_ts * relative_volume_xs).replace(0, np.nan)

    feature_dict["teacher_oi_rev_vol_ts"] = _stack_panel(alpha_vol_ts, base_index)
    feature_dict["teacher_oi_rev_vol_xs"] = _stack_panel(alpha_vol_xs, base_index)
    feature_dict["teacher_oi_rev_vol_xsts"] = _stack_panel(alpha_vol_xsts, base_index)

    # 10) Residualized OI reversal vs classic reversal
    alpha = delta_oi
    corr = alpha.corrwith(classic_reversal, axis=1)
    std_x = classic_reversal.std(axis=1).replace(0, np.nan)
    std_y = alpha.std(axis=1)
    coeff = corr.mul(std_y).div(std_x)

    residuals = alpha.sub(classic_reversal.mul(coeff, axis=0))
    feature_dict["teacher_oi_rev_residualized_price"] = _stack_panel(residuals, base_index)

    # 11) Selling back / chasing smart flows
    alpha_flow = delta_oi.sub(delta_oi.shift(1))
    alpha_flow = alpha_flow / (relative_volume_ts * relative_volume_xs).replace(0, np.nan)
    alpha_flow = alpha_flow.replace([np.inf, -np.inf], np.nan)
    feature_dict["teacher_oi_flow_chase_xsts"] = _stack_panel(alpha_flow, base_index)

    # 12) Parameter robustness family from class
    for w in [1, 2, 3, 6, 9, 12, 24, 48, 240]:
        alpha_w = -oi_value.pct_change(w, fill_method=None)
        feature_dict[f"teacher_oi_value_pct_{w}_rev"] = _stack_panel(alpha_w, base_index)

    return _feature_dict_to_dataframe(feature_dict, base_index)





def build_teacher_exact_plus_baseline_features(
    data: pd.DataFrame,
    raw_fields: List[str],
    feature_styles: List[str],
) -> pd.DataFrame:
    """
    Combine the exact teacher alpha family with the generic baseline features.
    """
    base_index = _get_base_index(data)

    feature_dict = _merge_feature_dicts([
        build_baseline_features(data, raw_fields, feature_styles).to_dict("series"),
        build_teacher_alpha_exact_features(data).to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)




def _select_feature_subset(
    features: pd.DataFrame,
    max_features: Optional[int],
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Reproducibly select a subset of columns if max_features is provided.
    """
    if max_features is None or max_features >= features.shape[1]:
        return features.copy()

    rng = np.random.RandomState(random_seed)
    selected_cols = rng.choice(
        features.columns,
        size=max_features,
        replace=False,
    )
    selected_cols = list(selected_cols)
    return features[selected_cols].copy()

def build_teacher_mixed_plus_oi_features(
    data: pd.DataFrame,
    random_seed: int = 0,
    max_baseline_features: int = 20,
) -> pd.DataFrame:
    """
    Teacher-style mixed feature universe:
    - broad statistical baseline features
    - reproducible feature budget / subset selection
    - always include the strong oi_reversal family
    """
    teacher_raw_fields = [
        "return",
        "close",
        "nb_trades",
        "volume_usd",
        "funding_rate",
        "open_interest_value",
    ]

    teacher_feature_styles = [
        "level",
        "delta_1",
        "delta_6",
        "shift_1",
        "shift_3",
        "shift_6",
        "shift_24",
        "mean_6",
        "mean_24",
        "mean_120",
        "std_6",
        "std_24",
        "std_120",
        "skew_6",
        "skew_24",
        "skew_120",
        "kurt_6",
        "kurt_24",
        "kurt_120",
    ]

    base_index = _get_base_index(data)

    baseline_features = build_baseline_features(
        data=data,
        raw_fields=teacher_raw_fields,
        feature_styles=teacher_feature_styles,
    )

    baseline_features = _select_feature_subset(
        baseline_features,
        max_features=max_baseline_features,
        random_seed=random_seed,
    )

    oi_features = build_open_interest_reversal_features(data)

    feature_dict = _merge_feature_dicts([
        baseline_features.to_dict("series"),
        oi_features.to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_teacher_mixed_plus_oi_plus_exact_features(
    data: pd.DataFrame,
    random_seed: int = 0,
    max_baseline_features: int = 20,
) -> pd.DataFrame:
    """
    Teacher-style mixed features +
    compact oi_reversal +
    exact teacher alpha family.
    """
    teacher_raw_fields = [
        "return",
        "close",
        "nb_trades",
        "volume_usd",
        "funding_rate",
        "open_interest_value",
    ]

    teacher_feature_styles = [
        "level",
        "delta_1",
        "delta_6",
        "shift_1",
        "shift_3",
        "shift_6",
        "shift_24",
        "mean_6",
        "mean_24",
        "mean_120",
        "std_6",
        "std_24",
        "std_120",
        "skew_6",
        "skew_24",
        "skew_120",
        "kurt_6",
        "kurt_24",
        "kurt_120",
    ]

    base_index = _get_base_index(data)

    baseline_features = build_baseline_features(
        data=data,
        raw_fields=teacher_raw_fields,
        feature_styles=teacher_feature_styles,
    )

    baseline_features = _select_feature_subset(
        baseline_features,
        max_features=max_baseline_features,
        random_seed=random_seed,
    )

    oi_features = build_open_interest_reversal_features(data)
    exact_teacher_features = build_teacher_alpha_exact_features(data)

    feature_dict = _merge_feature_dicts([
        baseline_features.to_dict("series"),
        oi_features.to_dict("series"),
        exact_teacher_features.to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)







def build_creative_alpha_pack_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handcrafted alpha pack inspired by candle geometry, positioning, funding,
    microstructure, and short-horizon reversal signals.
    """
    base_index = _get_base_index(data)

    open_ = get_field_panel(data, "open")
    high = get_field_panel(data, "high")
    low = get_field_panel(data, "low")
    close = get_field_panel(data, "close")
    returns = get_field_panel(data, "return")
    funding_rate = get_field_panel(data, "funding_rate")
    oi_value = get_field_panel(data, "open_interest_value")
    volume_usd = get_field_panel(data, "volume_usd")
    nb_trades = get_field_panel(data, "nb_trades")

    feature_dict: Dict[str, pd.Series] = {}

    # Candle geometry
    hl = (high - low).replace(0, np.nan)

    ccp = (close - low) / hl
    feature_dict["alpha_ccp"] = _stack_panel(-ccp, base_index)

    body_dir = (close - open_) / hl
    feature_dict["alpha_body_dir"] = _stack_panel(-body_dir, base_index)

    # OI signals
    oi_dp = oi_value.pct_change(fill_method=None)

    oi24 = oi_dp.rolling(24, min_periods=6)
    oi_dz = (oi_dp - oi24.mean()) / oi24.std().replace(0, np.nan)
    oi_ac = oi_dp - oi_dp.shift(1)

    feature_dict["alpha_oi_delta"] = _stack_panel(-oi_dp, base_index)
    feature_dict["alpha_oi_delta_z"] = _stack_panel(-oi_dz, base_index)
    feature_dict["alpha_oi_accel"] = _stack_panel(-oi_ac, base_index)

    # Return reversal / idiosyncratic return
    mkt_ret = returns.mean(axis=1)
    signal_idio = -(returns.sub(mkt_ret, axis=0))
    feature_dict["alpha_idio_ret"] = _stack_panel(signal_idio, base_index)

    feature_dict["alpha_ret2"] = _stack_panel(-returns.rolling(2, min_periods=1).sum(), base_index)
    feature_dict["alpha_ret6"] = _stack_panel(-returns.rolling(6, min_periods=1).sum(), base_index)

    # Funding / OI / trade structure
    fr24 = funding_rate.rolling(24, min_periods=6)
    signal_fr_z = -(funding_rate - fr24.mean()) / fr24.std().replace(0, np.nan)
    feature_dict["alpha_fr_z"] = _stack_panel(signal_fr_z, base_index)

    vol_6h = volume_usd.rolling(6, min_periods=1).mean()
    signal_oi_vol = -(oi_value / vol_6h.replace(0, np.nan))
    feature_dict["alpha_oi_vol"] = _stack_panel(signal_oi_vol, base_index)

    signal_avg_trade = -(volume_usd / nb_trades.replace(0, np.nan))
    feature_dict["alpha_avg_trade"] = _stack_panel(signal_avg_trade, base_index)

    # Regime-robust signals
    hl_range_raw = high - low
    hl_roll = hl_range_raw.rolling(24, min_periods=6)
    signal_hl_z = -((hl_range_raw - hl_roll.mean()) / hl_roll.std().replace(0, np.nan))
    feature_dict["alpha_hl_z"] = _stack_panel(signal_hl_z, base_index)

    typical_price = (high + low + close) / 3
    vwap_roll = (
        (typical_price * volume_usd).rolling(6, min_periods=1).sum()
        / volume_usd.rolling(6, min_periods=1).sum().replace(0, np.nan)
    )
    signal_vwap_dev = -((close - vwap_roll) / vwap_roll.replace(0, np.nan))
    feature_dict["alpha_vwap_dev"] = _stack_panel(signal_vwap_dev, base_index)

    # Multi-hour CCP
    low_4h = low.rolling(4, min_periods=2).min()
    high_4h = high.rolling(4, min_periods=2).max()
    low_6h_r = low.rolling(6, min_periods=3).min()
    high_6h_r = high.rolling(6, min_periods=3).max()

    signal_ccp4h = -((close - low_4h) / (high_4h - low_4h).replace(0, np.nan))
    signal_ccp6h = -((close - low_6h_r) / (high_6h_r - low_6h_r).replace(0, np.nan))
    feature_dict["alpha_ccp4h"] = _stack_panel(signal_ccp4h, base_index)
    feature_dict["alpha_ccp6h"] = _stack_panel(signal_ccp6h, base_index)

    # Cross-sectional volume rank
    signal_cs_vol = -(volume_usd.rank(axis=1, pct=True) - 0.5)
    feature_dict["alpha_cs_vol"] = _stack_panel(signal_cs_vol, base_index)

    # Funding level
    signal_fr_level = -funding_rate
    feature_dict["alpha_fr_level"] = _stack_panel(signal_fr_level, base_index)

    # OI per dollar traded
    signal_oi_per_vol = -(oi_dp / volume_usd.replace(0, np.nan))
    feature_dict["alpha_oi_per_vol"] = _stack_panel(signal_oi_per_vol, base_index)

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_creative_alpha_plus_oi_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creative handcrafted alpha pack + the existing oi_reversal family.
    """
    base_index = _get_base_index(data)

    feature_dict = _merge_feature_dicts([
        build_creative_alpha_pack_features(data).to_dict("series"),
        build_open_interest_reversal_features(data).to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)





def build_precomputed_feature_pack_features(
    data: pd.DataFrame,
    data_dir: str = "data/all/",
    max_nb_features: int = 20,
    random_seed: int = 0,
    filename_keyword: str = "feature",
) -> pd.DataFrame:
    """
    Load a reproducible subset of precomputed feature CSV files from disk.

    Expected format of each feature file:
    - index: datetime
    - columns: instruments

    These are then stacked to the canonical (datetime, instrument) index.
    """
    base_index = _get_base_index(data)

    feature_files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(data_dir):
        valid_files = [
            f for f in filenames
            if filename_keyword in f and f.endswith(".csv")
        ]
        valid_files = sorted(valid_files)
        feature_files.extend([os.path.join(dirpath, f) for f in valid_files])

    if len(feature_files) == 0:
        raise ValueError(
            f"No precomputed feature files found in '{data_dir}' "
            f"with keyword '{filename_keyword}'."
        )

    rng = np.random.RandomState(random_seed)
    selected_files = rng.choice(
        feature_files,
        size=min(max_nb_features, len(feature_files)),
        replace=False,
    )
    selected_files = sorted(selected_files)

    feature_dict: Dict[str, pd.Series] = {}

    for filepath in selected_files:
        feature = pd.read_csv(filepath, index_col=0, header=[0])
        feature.index = pd.to_datetime(feature.index)

        feature_name = os.path.basename(filepath).replace(".csv", "")
        feature_dict[feature_name] = feature.stack().reindex(base_index)

    return _feature_dict_to_dataframe(feature_dict, base_index)


def build_precomputed_plus_oi_features(
    data: pd.DataFrame,
    data_dir: str = "data/all/",
    max_nb_features: int = 20,
    random_seed: int = 0,
    filename_keyword: str = "feature",
) -> pd.DataFrame:
    """
    Precomputed feature subset + the strong oi_reversal family.
    """
    base_index = _get_base_index(data)

    feature_dict = _merge_feature_dicts([
        build_precomputed_feature_pack_features(
            data=data,
            data_dir=data_dir,
            max_nb_features=max_nb_features,
            random_seed=random_seed,
            filename_keyword=filename_keyword,
        ).to_dict("series"),
        build_open_interest_reversal_features(data).to_dict("series"),
    ])

    return _feature_dict_to_dataframe(feature_dict, base_index)

# ============================================================
# Main wrapper
# ============================================================

def build_features_from_config(
    data: pd.DataFrame,
    feature_config,
) -> pd.DataFrame:
    """
    Main feature wrapper.

    Supported feature_set_name values:
    - 'baseline'
    - 'oi_reversal'
    - 'price_reversal'
    - 'volume_conditioned'
    - 'funding_oi'
    - 'teacher_core'
    - 'teacher_plus_baseline'
    """
    feature_set_name = feature_config.feature_set_name

    if feature_set_name == "baseline":
        return build_baseline_features(
            data=data,
            raw_fields=feature_config.raw_fields,
            feature_styles=feature_config.feature_styles,
        )

    if feature_set_name == "oi_reversal":
        return build_open_interest_reversal_features(data)

    if feature_set_name == "price_reversal":
        return build_price_reversal_features(data)

    if feature_set_name == "volume_conditioned":
        return build_volume_conditioned_features(data)

    if feature_set_name == "funding_oi":
        return build_funding_open_interest_features(data)

    if feature_set_name == "teacher_core":
        return build_teacher_core_features(data)

    if feature_set_name == "teacher_plus_baseline":
        return build_teacher_plus_baseline_features(
            data=data,
            raw_fields=feature_config.raw_fields,
            feature_styles=feature_config.feature_styles,
        )
    
    if feature_set_name == "teacher_alpha_exact":
        return build_teacher_alpha_exact_features(data)

    if feature_set_name == "teacher_exact_plus_baseline":
        return build_teacher_exact_plus_baseline_features(
            data=data,
            raw_fields=feature_config.raw_fields,
            feature_styles=feature_config.feature_styles,
        )

    if feature_set_name == "teacher_mixed_plus_oi":
        return build_teacher_mixed_plus_oi_features(
            data=data,
            random_seed=0,
            max_baseline_features=20,
        )

    if feature_set_name == "teacher_mixed_plus_oi_plus_exact":
        return build_teacher_mixed_plus_oi_plus_exact_features(
            data=data,
            random_seed=0,
            max_baseline_features=20,
        )
    
    if feature_set_name == "creative_alpha_pack":
        return build_creative_alpha_pack_features(data)

    if feature_set_name == "creative_alpha_plus_oi":
        return build_creative_alpha_plus_oi_features(data)
    
    if feature_set_name == "precomputed_feature_pack":
        return build_precomputed_feature_pack_features(
            data=data,
            data_dir="data/all/",
            max_nb_features=feature_config.max_nb_features,
            random_seed=getattr(feature_config, "random_seed", 0),
            filename_keyword="feature",
        )

    if feature_set_name == "precomputed_plus_oi":
        return build_precomputed_plus_oi_features(
            data=data,
            data_dir="data/all/",
            max_nb_features=feature_config.max_nb_features,
            random_seed=getattr(feature_config, "random_seed", 0),
            filename_keyword="feature",
        )
    

    raise ValueError(f"Unknown feature_set_name: {feature_set_name}")



# ============================================================
# Label
# ============================================================

def build_label_next_return(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Build next-period stacked return label.

    Follows the teacher logic:
        data['return'].loc[start:end].shift(-1).stack()
    """
    label = data["return"].loc[start_date:end_date].shift(-1).stack()
    label.name = "label_next_return"
    return label