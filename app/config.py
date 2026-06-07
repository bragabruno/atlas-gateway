"""Runtime configuration (env-driven, no secrets in code).

Per-key auth uses an allowlist; override via the ATLAS_API_KEYS env var
(JSON list) in real deployments — secrets come from Key Vault via the CSI
driver, never from the image. The default dev key exists only for local tests.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    api_keys: tuple[str, ...] = ("dev-key",)


def get_settings() -> Settings:
    return Settings()
