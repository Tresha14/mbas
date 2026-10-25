"""Create the local demo interviewer account (credentials from MBAS_DEMO_EMAIL / MBAS_DEMO_PASSWORD in .env)
and print a login token. For local demos and end-to-end tests only.

    python -m tools.seed_demo
"""
from __future__ import annotations

import os
import sys

import httpx

from backend.config import settings  # noqa: F401  (loads .env)


def main() -> int:
    api = os.environ.get("MBAS_API_URL", "http://localhost:8020")
    email = os.environ.get("MBAS_DEMO_EMAIL", "demo.interviewer@mbas.local")
    pw = os.environ.get("MBAS_DEMO_PASSWORD", "mbas-demo-pass-2026")
    r = httpx.post(f"{api}/api/auth/register", json={"email": email, "name": "Demo Interviewer", "password": pw,
                                                     "role": "interviewer"})
    if r.status_code == 409:
        r = httpx.post(f"{api}/api/auth/login", json={"email": email, "password": pw})
    r.raise_for_status()
    print(r.json()["token"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
