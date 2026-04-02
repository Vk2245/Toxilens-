"""
ToxiLens Configuration — Settings from environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


# Project root = 4 levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────
    llm_provider: str = "groq"
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    mistral_api_key: str = ""

    # ── Paths ────────────────────────────────────────────
    model_dir: Path = PROJECT_ROOT / "ml" / "artifacts"
    tox21_data_dir: Path = PROJECT_ROOT / "ml" / "data" / "raw"

    # ── Server ───────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── ML ───────────────────────────────────────────────
    device: str = "cuda"  # "cuda" or "cpu"
    batch_size: int = 32

    # ── Tox21 Assay names (ordered) ──────────────────────
    assay_names: list[str] = [
        "NR-AR", "NR-AhR", "NR-AR-LBD", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
