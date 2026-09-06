from __future__ import annotations

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class Settings:
    """Настройки берутся из окружения; значения по умолчанию годятся для локального запуска."""

    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "riskml/artifacts"))
    demo_data: Path = Path(os.getenv("DEMO_DATA", "data/student-por.csv"))

    max_upload_bytes: int = _int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)   # 10 МБ
    max_rows: int = _int("MAX_ROWS", 20_000)

    rate_limit_per_minute: int = _int("RATE_LIMIT_PER_MINUTE", 30)

    cors_origins: list[str] = [
        o.strip() for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",") if o.strip()
    ]

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
