"""Minimal Label Studio REST client (projects, local storage, import, export).

Auth: LABEL_STUDIO_USERNAME / LABEL_STUDIO_PASSWORD (session login, the
default from .env), or LABEL_STUDIO_TOKEN with a personal access token from
Account & Settings (a JWT refresh token exchanged for an access token).
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "infra" / "labelstudio"

# Inside the Label Studio container (see docker-compose.yml):
#   ./data/processed/labelstudio  ->  /label-studio/files/mbas
DOC_ROOT = "/label-studio/files"
MEDIA_PREFIX = "mbas/media"

PROJECTS = {
    # key: (title, config file, maximum annotations per task)
    # default (fast) workflow: whole-session timeline + answers; the per-window projects are optional
    "sessions": ("MBAS sessions (timeline)", "sessions.xml", 1),
    "sessions_double": ("MBAS sessions (timeline, double-labelled)", "sessions.xml", 2),
    "windows": ("MBAS windows", "windows.xml", 1),
    "windows_double": ("MBAS windows (double-labelled)", "windows.xml", 2),
    "answers": ("MBAS answers", "answers.xml", 1),
    "answers_double": ("MBAS answers (double-labelled)", "answers.xml", 2),
}


def _load_env(path: Path) -> None:
    """Pick up LABEL_STUDIO_* from the repo .env (same minimal loader as the recorder backend)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env(ROOT / ".env")


def media_url(rel_path: str) -> str:
    """URL Label Studio serves a file under data/processed/labelstudio/media/ from."""
    return f"/data/local-files/?d={MEDIA_PREFIX}/{rel_path}"


class LabelStudio:
    def __init__(self, url: str | None = None, token: str | None = None, username: str | None = None,
                 password: str | None = None, timeout: float = 120.0):
        self.url = (url or os.environ.get("LABEL_STUDIO_URL") or "http://localhost:8090").rstrip("/")
        self.http = httpx.Client(base_url=self.url, timeout=timeout)
        token = token or os.environ.get("LABEL_STUDIO_TOKEN")
        username = username or os.environ.get("LABEL_STUDIO_USERNAME")
        password = password or os.environ.get("LABEL_STUDIO_PASSWORD")
        if token:
            self.http.headers["Authorization"] = self._auth_header(token)
        elif username and password:
            self._login(username, password)
        else:
            raise SystemExit("set LABEL_STUDIO_USERNAME + LABEL_STUDIO_PASSWORD (see .env) or LABEL_STUDIO_TOKEN")

    def _auth_header(self, token: str) -> str:
        if token.count(".") == 2:  # JWT personal access token -> short-lived access token
            r = self.http.post("/api/token/refresh", json={"refresh": token})
            r.raise_for_status()
            return f"Bearer {r.json()['access']}"
        return f"Token {token}"  # legacy token (disabled by default since Label Studio 1.18)

    def _login(self, username: str, password: str) -> None:
        """Session login, same as the web form. Works without enabling API tokens."""
        self.http.get("/user/login/")
        r = self.http.post("/user/login/", data={
            "email": username, "password": password, "csrfmiddlewaretoken": self.http.cookies.get("csrftoken")},
            headers={"Referer": f"{self.url}/user/login/"})
        if r.status_code != 302 or "sessionid" not in self.http.cookies:
            raise SystemExit(f"Label Studio login failed for {username} (status {r.status_code})")
        self.http.headers["X-CSRFToken"] = self.http.cookies.get("csrftoken")
        self.http.headers["Referer"] = f"{self.url}/"

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Label Studio {method} {path} -> {r.status_code}: {r.text[:500]}")
        return r

    def projects(self) -> list[dict]:
        out, page = [], 1
        while True:
            body = self._req("GET", "/api/projects", params={"page": page, "page_size": 100}).json()
            items = body["results"] if isinstance(body, dict) else body
            out.extend(items)
            if not isinstance(body, dict) or not body.get("next"):
                return out
            page += 1

    def project_by_title(self, title: str) -> dict | None:
        return next((p for p in self.projects() if p["title"] == title), None)

    def ensure_project(self, key: str) -> dict:
        title, cfg, max_ann = PROJECTS[key]
        label_config = (CONFIG_DIR / cfg).read_text(encoding="utf-8")
        p = self.project_by_title(title)
        payload = {"title": title, "label_config": label_config, "maximum_annotations": max_ann,
                   "show_skip_button": False, "enable_empty_annotation": False}
        if p is None:
            p = self._req("POST", "/api/projects", json=payload).json()
        else:
            p = self._req("PATCH", f"/api/projects/{p['id']}", json=payload).json()
        self.ensure_local_storage(p["id"])
        return p

    def ensure_local_storage(self, project_id: int) -> None:
        """Local-files storage is what authorises Label Studio to serve /data/local-files/ for this project."""
        path = f"{DOC_ROOT}/{MEDIA_PREFIX}"
        have = self._req("GET", "/api/storages/localfiles", params={"project": project_id}).json()
        if any(s.get("path") == path for s in have):
            return
        self._req("POST", "/api/storages/localfiles", json={
            "project": project_id, "path": path, "title": "MBAS media", "use_blob_urls": False, "regex_filter": ""})

    def export_tasks(self, project_id: int) -> list[dict]:
        return self._req("GET", f"/api/projects/{project_id}/export",
                         params={"exportType": "JSON", "download_all_tasks": "true"}).json()

    def import_tasks(self, project_id: int, tasks: list[dict], batch: int = 500) -> int:
        n = 0
        for i in range(0, len(tasks), batch):
            self._req("POST", f"/api/projects/{project_id}/import", json=tasks[i:i + batch])
            n += len(tasks[i:i + batch])
        return n

    def users(self) -> dict[int, str]:
        body = self._req("GET", "/api/users").json()
        items = body["results"] if isinstance(body, dict) else body
        return {u["id"]: u.get("email") or u.get("username") or f"user{u['id']}" for u in items}
