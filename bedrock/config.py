import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is required")
    return value


@dataclass
class Settings:
    aws_region: str
    bedrock_model_id: str


def load_settings() -> Settings:
    return Settings(
        aws_region=_env("AWS_REGION", "us-east-1"),
        bedrock_model_id=_env("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0"),
    )
