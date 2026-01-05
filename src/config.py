from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic.config import ConfigDict
import yaml

class PathCfg(BaseModel):
    raw_data : str
    artifacts_dir : str = 'artifacts'

class PreprocessCfg(BaseModel):
    drop_columns: List[str] = []

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
    model_config = ConfigDict(populate_by_name=True)
    paths : PathCfg
    schema_: SchemaCfg = Field(alias = 'schema')
    split : SplitCfg = SplitCfg()
    project : ProjectCfg = ProjectCfg()
    inference: InferenceCfg = InferenceCfg()
    preprocess: PreprocessCfg = PreprocessCfg()

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