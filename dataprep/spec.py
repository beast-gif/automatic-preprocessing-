"""
ProjectSpec: the structured version of the user's project idea.

Every later stage (query expansion, search, ranking) works from this,
never from the raw text.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "time_series_forecasting",
    "clustering",
    "anomaly_detection",
]
Modality = Literal["tabular", "text", "image", "audio", "video"]


class ProjectSpec(BaseModel):
    summary: str = Field(description="one-sentence restatement of the project")
    task_type: TaskType
    modality: Modality
    target_hint: Optional[str] = Field(
        default=None, description="likely target column name in snake_case; null for clustering"
    )
    domain_keywords: list[str] = Field(min_length=2, max_length=8)
    feature_hints: list[str] = Field(default_factory=list, max_length=8)
    min_rows: int = Field(default=1000, ge=50)
    constraints: list[str] = Field(default_factory=list, description="only what the user stated")

    @field_validator("domain_keywords", "feature_hints", "constraints")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for s in v:
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    @field_validator("target_hint")
    @classmethod
    def _snake(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        return "_".join(v.strip().lower().replace("-", " ").split())
