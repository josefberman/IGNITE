"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IGNITE_", extra="ignore")

    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    seed_csv: Path = ROOT / "sample_start_molecules.csv"
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'ignite.db'}"
    redis_url: str = ""
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    max_beam_depth: int = 20
    max_beam_width: int = 50
    max_top_k: int = 25
    openai_api_key: str = ""
    epo_ops_key: str = ""
    lens_api_key: str = ""
    pubchem_enabled: bool = True
    openalex_enabled: bool = True
    learned_checkpoint_dir: Path = ROOT / "ml" / "checkpoints"
    processed_data_dir: Path = ROOT / "data" / "processed"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
