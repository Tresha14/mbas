"""Passwords (PBKDF2-SHA256), JWTs and role checks.

Roles: interviewer, meeting_host (accounts); candidate, participant (per-interview tokens from the invite link).
Only interviewer / meeting_host tokens that own the interview may open the analytics channel.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

import jwt

ROLES = ("interviewer", "candidate", "meeting_host", "participant")
HOST_ROLES = ("interviewer", "meeting_host")


def hash_password(pw: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iterations)
    return f"pbkdf2${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, dk = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), base64.b64decode(salt), int(it))
        return hmac.compare_digest(got, base64.b64decode(dk))
    except ValueError:
        return False


def make_jwt(secret: str, sub: str, role: str, ttl_s: int, **extra) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    now = int(time.time())
    return jwt.encode({"sub": sub, "role": role, "iat": now, "exp": now + ttl_s, **extra}, secret, algorithm="HS256")


def read_jwt(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"])
