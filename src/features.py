from typing import List

import numpy as np
import pandas as pd

from src.data_utils import get_field_panel


def _compute_feature_from_panel(
    raw_data: pd.DataFrame,
    feature_style: str,
) -> pd.DataFrame:
    """
    Compute one feature transformation from a field panel.

    Parameters
    ----------
    raw_data : pd.DataFrame
        DataFrame indexed by datetime, columns are instruments.
    feature_style : str
        Feature transformation name.

    Returns
    -------
    pd.DataFrame
        Feature panel with same general structure as raw_data.
    """
    if feature_style == "level":
        return raw_data

    if feature_style == "delta_1":
        return raw_data.pct_change(1, fill_method=None)

    if feature_style == "delta_6":
        return raw_data.pct_change(6, fill_method=None)

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

    raise ValueError(f"Unknown feature_style: {feature_style}")


def build_baseline_features(
    data: pd.DataFrame,
    raw_fields: List[str],
    feature_styles: List[str],
) -> pd.DataFrame:
    """
    Build a baseline stacked feature matrix from selected raw fields and feature styles.

    Output format follows the teacher notebook:
    - index: stacked (datetime, instrument)
    - columns: feature names like 'return_level', 'close_mean_24', ...

    Parameters
    ----------
    data : pd.DataFrame
        Multi-index market panel data.
    raw_fields : list[str]
        Top-level field names to use.
    feature_styles : list[str]
        Transformations to apply to each field.

    Returns
    -------
    pd.DataFrame
        Stacked feature matrix.
    """
    base_index = data["return"].stack().index
    features = pd.DataFrame(index=base_index)

    for field in raw_fields:
        raw_panel = get_field_panel(data, field)

        for feature_style in feature_styles:
            feature_panel = _compute_feature_from_panel(
                raw_data=raw_panel,
                feature_style=feature_style,
            )

            feature_series = feature_panel.stack().reindex(base_index)
            feature_name = f"{field}_{feature_style}"
            features[feature_name] = feature_series

    return features


def build_features_from_config(
    data: pd.DataFrame,
    feature_config,
) -> pd.DataFrame:
    """
    Main wrapper for feature generation.

    For now, only the baseline feature set is implemented.
    Later we can expand this to include engineered features and
    custom alpha families.

    Parameters
    ----------
    data : pd.DataFrame
        Market panel data.
    feature_config : FeatureConfig
        Config section from ExperimentConfig.

    Returns
    -------
    pd.DataFrame
        Stacked feature matrix.
    """
    if feature_config.feature_set_name == "baseline":
        return build_baseline_features(
            data=data,
            raw_fields=feature_config.raw_fields,
            feature_styles=feature_config.feature_styles,
        )

    raise ValueError(
        f"Unknown feature_set_name: {feature_config.feature_set_name}"
    )


def build_label_next_return(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Build next-period return label in stacked format.

    This follows the teacher's logic:
        label = data['return'].loc[start:end].shift(-1).stack()

    Parameters
    ----------
    data : pd.DataFrame
        Market panel data.
    start_date : str
        Training start date.
    end_date : str
        Training end date.

    Returns
    -------
    pd.Series
        Stacked next-period return label.
    """
    label = data["return"].loc[start_date:end_date].shift(-1).stack()
    label.name = "label_next_return"
    return label