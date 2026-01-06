from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator, AliasChoices
from pydantic.config import ConfigDict
import yaml

class PathCfg(BaseModel):
    raw_data : str
    artifacts_dir : str = 'artifacts'

class DecisionCfg(BaseModel):
    # Decision policy for turning probabilities into labels.
    # Accept both `metric` (correct) and legacy `metrics` from older YAML.
    metric: Literal["f1", "cost"] = Field(
        default="f1",
        validation_alias=AliasChoices("metric", "metrics"),
    )
    cost_fp: float = Field(1.0, gt=0.0)
    cost_fn: float = Field(5.0, gt=0.0)

class PreprocessCfg(BaseModel):
    drop_columns: List[str] = []

class ModelCfg(BaseModel):
    # Which model main.py should run
    name: Literal["catboost", "hgb", "logreg"] = "catboost"


class TuningCfg(BaseModel):
    # Simple randomized tuning (no CV)
    n_iter: int = Field(25, ge=1)
    scorer: Literal["pr_auc", "roc_auc"] = "pr_auc"
    iterations: int = Field(8000, ge=100)
    early_stopping_rounds: int = Field(
        200,
        ge=1,
        # Accept both old and correct YAML keys if present
        validation_alias=AliasChoices("early_stopping_rounds", "early_stopping_round"),
    )
    verbose: int = 0


class CatBoostCfg(BaseModel):
    iterations: int = Field(20000, ge=100)
    learning_rate: float = Field(0.05, gt=0.0)
    depth: int = Field(6, ge=1, le=16)
    l2_leaf_reg: float = Field(3.0, gt=0.0)
    min_data_in_leaf: int = Field(50, ge=1)
    rsm: float = Field(0.9, gt=0.0, le=1.0)
    bootstrap_type: Literal["Bayesian", "Bernoulli", "MVS"] = "Bayesian"
    bagging_temperature: float = Field(3.0, ge=0.0)

    early_stopping_rounds: int = Field(
        300,
        ge=1,
        validation_alias=AliasChoices("early_stopping_rounds", "early_stopping_round"),
    )
    verbose: int = 200
    auto_class_weights: Literal["Balanced", "SqrtBalanced"] = "Balanced"


class HGBCfg(BaseModel):
    max_iter: int = Field(300, ge=1)
    learning_rate: float = Field(0.1, gt=0.0)


class InferenceCfg(BaseModel):
    # Allow null in YAML. If None, inference code should load the saved best threshold artifact.
    threshold: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def _check_threshold(self):
        if self.threshold is None:
            return self
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("inference.threshold must be between 0 and 1")
        return self

class ProjectCfg(BaseModel):
    random_seed : int = 42

class SchemaCfg(BaseModel):
    target_column : str

class SplitCfg(BaseModel):
    test_size : float = Field(0.2, ge = 0.0, le = 1.0)
    val_size: float = Field(0.1, ge=0.0, le=1.0)
    stratify : bool = True
    @model_validator(mode = 'after')
    def _check_split_size(self):
        if self.val_size + self.test_size >= 1.0:
            raise ValueError(
                f'Invalid split size: val_size({self.val_size}) + test_size({self.test_size}) must be < 1.0'
            )
        return self
class AppConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    paths : PathCfg
    schema_: SchemaCfg = Field(alias = 'schema')
    split : SplitCfg = SplitCfg()
    project : ProjectCfg = ProjectCfg()
    inference: InferenceCfg = InferenceCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    model: ModelCfg = ModelCfg()
    tuning: TuningCfg = TuningCfg()
    catboost: CatBoostCfg = CatBoostCfg()
    hgb: HGBCfg = HGBCfg()
    decision: DecisionCfg = DecisionCfg()


def load_config(path : str = 'configs/config.yaml') -> AppConfig:
    cfg_path = Path(path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found :{cfg_path.resolve()}")
    
    
    with cfg_path.open( 'r', encoding = 'utf-8') as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError('Config.yaml must contain a YAML mapping (a dict at the root).')
    try:
        return AppConfig.model_validate(data)
    except AttributeError:
        return AppConfig.parse_obj(data)