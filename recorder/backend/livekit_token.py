"""Mint LiveKit access tokens (HS256 JWT) without the LiveKit SDK."""
from __future__ import annotations

import json
import time

import jwt


def make_token(api_key: str, api_secret: str, room: str, identity: str, name: str, role: str, ttl_s: int = 6 * 3600) -> str:
    now = int(time.time())
    claims = {
        "iss": api_key,
        "sub": identity,
        "name": name,
        "nbf": now - 10,
        "exp": now + ttl_s,
        "metadata": json.dumps({"role": role}),
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    return jwt.encode(claims, api_secret, algorithm="HS256")
