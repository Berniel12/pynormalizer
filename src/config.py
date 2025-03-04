"""
Configuration settings for the tender normalizer.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

# No need to load .env file on Apify
# from dotenv import load_dotenv
# load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # Supabase configuration
    supabase_url: str = Field(default_factory=lambda: os.environ.get("SUPABASE_URL", ""))
    supabase_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("SUPABASE_KEY", ""))
    )

    # OpenAI configuration
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(
            os.environ.get("OPENAI_API_KEY", os.environ.get("OPENAI_KEY", ""))
        )
    )
    openai_model: str = Field(default="gpt-4o-mini")

    # Logfire configuration (optional)
    logfire_token: SecretStr | None = Field(
        default_factory=lambda: SecretStr(os.environ.get("LOGFIRE_TOKEN", ""))
        if os.environ.get("LOGFIRE_TOKEN")
        else None
    )

    # Processing configuration - read from environment with fallbacks
    batch_size: int = Field(
        default_factory=lambda: int(os.environ.get("BATCH_SIZE", "25"))
    )
    llm_timeout_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    )
    use_llm_normalization: bool = Field(
        default_factory=lambda: os.environ.get("USE_LLM_NORMALIZATION", "true").lower() == "true"
    )
    concurrent_requests: int = Field(
        default_factory=lambda: int(os.environ.get("CONCURRENT_REQUESTS", "5"))
    )
    max_retries: int = Field(default=3)


class NormalizerConfig(BaseModel):
    """Configuration for tender normalization."""

    # Mode settings
    mode: Literal["strict", "relaxed"] = Field(default="relaxed")
    
    # Source-specific settings
    use_llm_for_sources: dict[str, bool] = Field(
        default_factory=lambda: {
            # By default, use LLM for all sources
            "_default": True,
            # Source-specific overrides
            "sam_gov": True,
            "wb": True,
            "adb": True,
            "un_tenders": True,
            "ted": True,
            "dgmarket": True,
        }
    )
    
    # Fields that must be present in normalized tenders
    critical_fields: list[str] = Field(
        default_factory=lambda: ["title", "description", "country"]
    )


# Create instances
settings = Settings()
normalizer_config = NormalizerConfig() 