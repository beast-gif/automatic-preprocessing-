"""
Candidate: one dataset found by search, in the same shape whichever source it came from.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["huggingface", "kaggle"]


class Candidate(BaseModel):
    source: Source
    id: str                                  # owner/slug on both sources
    title: str
    url: str
    description: str = ""                    # Kaggle: from metadata (top K only); HF: from search, often empty
    downloads: int = 0                       # HF: recent downloads; Kaggle: all-time downloads
    likes: Optional[int] = None
    size_bytes: Optional[int] = None         # Kaggle only
    rows_low: Optional[int] = None           # HF only, lower bound of size_categories
    rows_est: Optional[int] = None           # HF only, geometric midpoint of size_categories
    rows_exact: Optional[int] = None         # HF only, from dataset_structure (top K only)
    columns: list[dict] = Field(default_factory=list)        # HF schema: [{"name": ..., "dtype": ...}]
    download_urls: list[str] = Field(default_factory=list)   # HF parquet export URLs
    license: Optional[str] = None
    formats: list[str] = Field(default_factory=list)      # HF format: tags (parquet, csv, json ...)
    modalities: list[str] = Field(default_factory=list)   # HF modality: tags
    usability: Optional[float] = None        # Kaggle usabilityRating, 0-1
    query_hits: list[str] = Field(default_factory=list)   # which search queries found it
    pre_score: float = 0.0
    enriched: bool = False                   # True once metadata / structure has been fetched

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"