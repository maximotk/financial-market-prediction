from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class PathsConfig:
    data_dir: str = "data/all/"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    in_sample_filename: str = "data_in_sample.csv"
    test_filename: str = "data_test.csv"


@dataclass
class DataConfig:
    datetime_col: Optional[str] = None
    index_col: int = 0
    header_rows: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class DateConfig:
    start_date_train: str = "2023-01-24"
    last_date_train: str = "2024-01-24"
    start_date_validate: str = "2024-01-25"
    last_date_validate: str = "2024-07-24"


@dataclass
class EvalConfig:
    risk_free_rate_annual: float = 0.05
    transaction_cost: float = 0.0
    evaluation_lags: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 6, 12])
    plot_option: str = "matplotlib"
    annualization_factor: int = 24 * 365


@dataclass
class FeatureConfig:
    feature_set_name: str = "baseline"

    raw_fields: List[str] = field(default_factory=lambda: [
        "return",
        "close",
        "nb_trades",
        "volume_usd",
        "funding_rate",
        "open_interest_value",
    ])

    feature_styles: List[str] = field(default_factory=lambda: [
        "level",
        "delta_1",
        "shift_1",
        "shift_6",
        "mean_24",
        "std_24",
    ])

    use_engineered_features: bool = False
    engineered_feature_prefix: str = "feature"
    max_nb_features: int = 20
    pct_engineered_features: float = 0.25

    fillna_value: float = 0.0


@dataclass
class PreprocessConfig:
    method: str = "cross_sectional_rank_standardize"
    rank_to_pct: bool = True
    center_cross_sectionally: bool = True
    scale_cross_sectionally: bool = True
    fillna_value: float = 0.0
    cast_column_names_to_str: bool = True


@dataclass
class ModelConfig:
    model_name: str = "elastic_net"

    params: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 1e-5,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "tol": 2e-2,
        "selection": "random",
        "max_iter": 500,
        "random_state": 0,
    })


@dataclass
class WalkForwardConfig:
    enabled: bool = True
    retrain_frequency: str = "ME"
    train_window_days: int = 365
    validation_window_days: int = 31
    min_folds_before_start: int = 2
    prediction_delay_hours: int = 1


@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline_elastic_net"
    random_seed: int = 0

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dates: DateConfig = field(default_factory=DateConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_json(filepath: str) -> "ExperimentConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return ExperimentConfig(
            experiment_name=raw["experiment_name"],
            random_seed=raw["random_seed"],
            paths=PathsConfig(**raw["paths"]),
            data=DataConfig(**raw["data"]),
            dates=DateConfig(**raw["dates"]),
            evaluation=EvalConfig(**raw["evaluation"]),
            features=FeatureConfig(**raw["features"]),
            preprocessing=PreprocessConfig(**raw["preprocessing"]),
            model=ModelConfig(**raw["model"]),
            walk_forward=WalkForwardConfig(**raw["walk_forward"]),
        )


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()