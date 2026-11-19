"""Pre-demo check: are the trained models, tokenizers and services ready?

    python -m tools.preflight            # files + model consistency (no services needed)
    python -m tools.preflight --services # also check Docker services, the API and the web app

Exit code 0 only when nothing is wrong. Warnings (e.g. a missing ONNX file) do not fail the check.
"""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.request

import backend.config  # noqa: F401  (loads .env, so MBAS_ASR is seen)
from ml.common.config import CHECKPOINTS, PROCESSED, ROOT

REQUIRED = ["vad", "asr", "disfluency", "face_detector", "headpose", "gaze_iris", "window_models", "answer_quality",
            "fusion", "behaviour", "meeting_head"]
OK, WARN, BAD = "ok  ", "warn", "FAIL"


def _ckpt(name: str) -> str | None:
    for f in ("averaged.pt", "best.pt") if name == "asr" else ("best.pt",):
        if (CHECKPOINTS / name / f).exists():
            return f
    return None


def check_files(rows: list[tuple[str, str, str]]) -> None:
    for n in REQUIRED:
        f = _ckpt(n)
        rows.append((OK if f else BAD, f"checkpoint {n}", f or "missing: copy checkpoints/ from the team's models zip"))
    for p in ("tokenizers/asr.model", "tokenizers/answer.model", "lm/interview_4gram.arpa"):
        ok = (PROCESSED / p).exists()
        rows.append((OK if ok else BAD, f"data/processed/{p}", "" if ok else "missing"))
    import os

    backend = os.environ.get("MBAS_ASR", "own").lower()
    if backend == "whisper":
        try:
            import faster_whisper  # noqa: F401

            rows.append((OK, "speech recogniser", f"Whisper {os.environ.get('MBAS_WHISPER_MODEL', 'base.en')} (pre-trained, optional)"))
        except ImportError:
            rows.append((WARN, "speech recogniser", "MBAS_ASR=whisper but faster-whisper is not installed: pip install faster-whisper (our own ASR is used)"))
    else:
        rows.append((OK, "speech recogniser", "our own Conformer-CTC (set MBAS_ASR=whisper for the pre-trained Whisper)"))
    onnx = ROOT / "ml" / "export" / "onnx"
    n = len(list(onnx.glob("*.onnx"))) if onnx.exists() else 0
    rows.append((OK if n else WARN, "ONNX exports", f"{n} files" if n else "none: the PyTorch fallback is used (slower)"))


def check_answer_model(rows: list[tuple[str, str, str]]) -> None:
    """The answer-quality checkpoint and tokenizer must come from the same run, or every transcript crashes it."""
    if not (CHECKPOINTS / "answer_quality" / "best.pt").exists() or not (PROCESSED / "tokenizers" / "answer.model").exists():
        return
    try:
        import torch

        from ml.common import text

        st = torch.load(CHECKPOINTS / "answer_quality" / "best.pt", map_location="cpu", weights_only=False)
        tok = text.load("answer")
        need, have = int(st["vocab"]), len(tok) + 2        # + CLS and SEP
        ok = need == have
        rows.append((OK if ok else WARN, "answer model vs tokenizer",
                     f"vocab {need}" if ok else f"checkpoint expects {need} ids, tokenizer gives {have}: answer relevance "
                                                f"scores are skipped (the rest of each answer still works). Get the matching "
                                                f"answer.model or retrain (python -m ml.response.answer_quality.train)"))
    except Exception as e:  # noqa: BLE001
        rows.append((BAD, "answer model vs tokenizer", f"{type(e).__name__}: {e}"))


def _port(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def _http(url: str, timeout: float = 3) -> bool:
    try:
        return urllib.request.urlopen(url, timeout=timeout).status < 500
    except Exception:  # noqa: BLE001
        return False


def check_services(rows: list[tuple[str, str, str]]) -> None:
    for name, port in (("postgres", 5442), ("redis", 6389), ("livekit", 7880)):
        ok = _port("127.0.0.1", port)
        rows.append((OK if ok else BAD, f"docker: {name} :{port}", "" if ok else r"run .\tasks.ps1 up"))
    for name, url, hint in (("API", "http://localhost:8020/docs", r".\tasks.ps1 api"),
                            ("web app", "http://localhost:3020/login", r".\tasks.ps1 web")):
        # the Next.js dev server compiles a page on its first request (about 10 s), so the web app gets longer
        ok = _http(url, 60 if name == "web app" else 3)
        rows.append((OK if ok else BAD, name, "" if ok else f"not reachable: {hint}"))
    rows.append((WARN, "workers", r"cannot be probed: after .\tasks.ps1 workers look for '[vision] ready' and '[speech] ready' "
                                  "and no 'disabled for interview' line"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--services", action="store_true", help="also check Docker services, the API and the web app")
    args = ap.parse_args(argv)
    rows: list[tuple[str, str, str]] = []
    check_files(rows)
    check_answer_model(rows)
    if args.services:
        check_services(rows)
    for level, what, note in rows:
        print(f"[{level}] {what:34s} {note}")
    bad = sum(r[0] == BAD for r in rows)
    print(f"\n{'READY' if not bad else 'NOT READY'}: {bad} problem(s), {sum(r[0] == WARN for r in rows)} warning(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
