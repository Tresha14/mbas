"""Backend settings from the environment / repo .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env(ROOT / ".env")


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("MBAS_DATABASE_URL")
                              or os.environ.get("DATABASE_URL", "postgresql://mbas:mbas@localhost:5442/mbas"))
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6389/0"))
    jwt_secret: str = field(default_factory=lambda: os.environ.get("MBAS_JWT_SECRET", "mbas-dev-jwt-secret-change-me-0123456789"))
    jwt_ttl_s: int = 12 * 3600
    livekit_key: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_KEY", "mbasdevkey"))
    livekit_secret: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_SECRET",
                                                                       "mbas_dev_secret_change_me_0123456789abcdef"))
    livekit_url: str = field(default_factory=lambda: os.environ.get("LIVEKIT_URL", "ws://localhost:7880"))
    public_url: str = field(default_factory=lambda: os.environ.get("MBAS_PUBLIC_URL", "http://localhost:3020"))
    questions_path: Path = ROOT / "recorder" / "questions.json"


def settings() -> Settings:
    return Settings()
