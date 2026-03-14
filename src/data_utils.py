from typing import List, Optional

import pandas as pd


def load_market_data(
    filepath: str,
    index_col: int = 0,
    header_rows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load the main market panel data.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    index_col : int, default=0
        Column to use as row index.
    header_rows : list[int], optional
        Header rows for multi-index columns. If None, defaults to [0, 1].

    Returns
    -------
    pd.DataFrame
        Loaded market data with datetime index.
    """
    if header_rows is None:
        header_rows = [0, 1]

    data = pd.read_csv(
        filepath,
        index_col=index_col,
        header=header_rows,
    )
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def get_available_fields(data: pd.DataFrame) -> List[str]:
    """
    Return the top-level field names from a multi-index column DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Market panel data with multi-index columns.

    Returns
    -------
    list[str]
        Unique top-level field names.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected multi-index columns for market data.")

    return list(data.columns.get_level_values(0).drop_duplicates())


def get_field_panel(data: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """
    Extract one field panel from the market data.

    Example:
        data['return']
        data['close']

    Parameters
    ----------
    data : pd.DataFrame
        Market panel data with multi-index columns.
    field_name : str
        Top-level field name to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime, columns are instruments.
    """
    available_fields = get_available_fields(data)
    if field_name not in available_fields:
        raise KeyError(
            f"Field '{field_name}' not found. Available fields: {available_fields}"
        )

    field_panel = data[field_name].copy()
    field_panel = field_panel.sort_index()
    return field_panel


def validate_market_data_structure(data: pd.DataFrame) -> None:
    """
    Validate that the dataset has the expected basic structure.

    Checks:
    - datetime index
    - multi-index columns
    - non-empty shape

    Parameters
    ----------
    data : pd.DataFrame
        Loaded market data.

    Raises
    ------
    ValueError
        If structure is not as expected.
    """
    if data.empty:
        raise ValueError("Loaded market data is empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Market data index must be a pandas DatetimeIndex.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Market data columns must be a pandas MultiIndex.")

    if data.columns.nlevels < 2:
        raise ValueError("Market data columns must have at least 2 levels.")


def summarize_market_data(data: pd.DataFrame) -> pd.Series:
    """
    Return a compact summary of the loaded market dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Loaded market panel.

    Returns
    -------
    pd.Series
        Summary stats for quick inspection.
    """
    validate_market_data_structure(data)

    fields = get_available_fields(data)
    n_timestamps = len(data.index)

    instruments = data.columns.get_level_values(1).drop_duplicates()
    n_instruments = len(instruments)

    summary = pd.Series(
        {
            "n_timestamps": n_timestamps,
            "n_columns_total": data.shape[1],
            "n_fields": len(fields),
            "n_instruments": n_instruments,
            "start_date": data.index.min(),
            "end_date": data.index.max(),
        }
    )

    return summary