from typing import Tuple

import pandas as pd


def cross_sectional_rank_standardize_feature(
    feature_series: pd.Series,
    rank_to_pct: bool = True,
    center_cross_sectionally: bool = True,
    scale_cross_sectionally: bool = True,
) -> pd.Series:
    """
    Apply teacher-style cross-sectional preprocessing to one stacked feature.

    Steps:
    1) unstack to (datetime x instruments)
    2) rank cross-sectionally at each timestamp
    3) optionally center each cross section
    4) optionally scale each cross section
    5) stack back to original format

    Parameters
    ----------
    feature_series : pd.Series
        Stacked feature with MultiIndex (datetime, instrument).
    rank_to_pct : bool, default=True
        Whether to convert ranks to percentiles.
    center_cross_sectionally : bool, default=True
        Whether to subtract the cross-sectional mean at each timestamp.
    scale_cross_sectionally : bool, default=True
        Whether to divide by the cross-sectional std at each timestamp.

    Returns
    -------
    pd.Series
        Preprocessed stacked feature.
    """
    feature_panel = feature_series.unstack()

    if rank_to_pct:
        feature_panel = feature_panel.rank(axis=1, pct=True)
    else:
        feature_panel = feature_panel.rank(axis=1)

    if center_cross_sectionally:
        feature_panel = feature_panel.sub(feature_panel.mean(axis=1), axis=0)

    if scale_cross_sectionally:
        std = feature_panel.std(axis=1).replace(0, 1)
        feature_panel = feature_panel.div(std, axis=0)

    feature_series_processed = feature_panel.stack().reindex(feature_series.index)
    return feature_series_processed


def preprocess_features(
    features: pd.DataFrame,
    preprocess_config,
) -> pd.DataFrame:
    """
    Preprocess a stacked feature matrix according to config.

    Parameters
    ----------
    features : pd.DataFrame
        Stacked feature matrix with MultiIndex rows (datetime, instrument).
    preprocess_config : PreprocessConfig
        Preprocessing section from ExperimentConfig.

    Returns
    -------
    pd.DataFrame
        Preprocessed feature matrix.
    """
    if preprocess_config.method != "cross_sectional_rank_standardize":
        raise ValueError(
            f"Unknown preprocessing method: {preprocess_config.method}"
        )

    features_processed = pd.DataFrame(index=features.index)

    for feature_name in features.columns:
        feature_series = features[feature_name]

        feature_series_processed = cross_sectional_rank_standardize_feature(
            feature_series=feature_series,
            rank_to_pct=preprocess_config.rank_to_pct,
            center_cross_sectionally=preprocess_config.center_cross_sectionally,
            scale_cross_sectionally=preprocess_config.scale_cross_sectionally,
        )

        features_processed[feature_name] = feature_series_processed

    features_processed = features_processed.fillna(preprocess_config.fillna_value)

    if preprocess_config.cast_column_names_to_str:
        features_processed = features_processed.rename(str, axis="columns")

    return features_processed


def align_features_and_label(
    features: pd.DataFrame,
    label: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Keep only the common stacked index between features and label.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    label : pd.Series
        Target series.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        Aligned features and label.
    """
    common_index = features.index.intersection(label.index)
    features_aligned = features.reindex(common_index)
    label_aligned = label.reindex(common_index)
    return features_aligned, label_aligned


def slice_features_by_date(
    features: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Slice a stacked feature matrix by datetime range.

    Assumes the first level of the index is datetime.

    Parameters
    ----------
    features : pd.DataFrame
        Stacked feature matrix.
    start_date : str
        Start date.
    end_date : str
        End date.

    Returns
    -------
    pd.DataFrame
        Date-sliced feature matrix.
    """
    datetime_index = features.index.get_level_values(0)
    mask = (datetime_index >= pd.Timestamp(start_date)) & (
        datetime_index <= pd.Timestamp(end_date)
    )
    return features.loc[mask]