"""Switch the whole system between the two demo modes by editing .env (the workers must be restarted afterwards).

    python -m tools.set_mode own            # our own trained models only (the project's claim): MBAS_ASR=own, MBAS_FACE=own
    python -m tools.set_mode pretrained     # with the optional pre-trained add-ons: MBAS_ASR=whisper, MBAS_FACE=mediapipe
    python -m tools.set_mode                # just show the current mode

Only those two lines are changed (a backup is kept as .env.bak); everything else in .env is left exactly as it is.
See docs/pretrained_models.md for what each mode shows and why the choice matters.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = {"own": {"MBAS_ASR": "own", "MBAS_FACE": "own"}, "pretrained": {"MBAS_ASR": "whisper", "MBAS_FACE": "mediapipe"}}


def read_mode(env: Path) -> dict[str, str]:
    vals = {"MBAS_ASR": "own", "MBAS_FACE": "own"}                   # the defaults when a line is absent
    seen = set()
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(MBAS_ASR|MBAS_FACE)\s*=\s*(\S*)", line)
            if m and m.group(1) not in seen:                          # the app keeps the FIRST line of a repeated setting
                vals[m.group(1)] = m.group(2).strip().lower()
                seen.add(m.group(1))
    return vals


def describe(vals: dict[str, str]) -> str:
    if vals["MBAS_ASR"] == "own" and vals["MBAS_FACE"] == "own":
        return "OWN: only models trained by us (speech: our Conformer-CTC, face: our detector)"
    parts = []
    if vals["MBAS_ASR"] != "own":
        parts.append(f"speech = pre-trained {vals['MBAS_ASR']}")
    if vals["MBAS_FACE"] != "own":
        parts.append(f"face = pre-trained {vals['MBAS_FACE']}")
    return "PRE-TRAINED ADD-ONS ON: " + ", ".join(parts)


def set_mode(env: Path, mode: str) -> dict[str, str]:
    want = MODES[mode]
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    if env.exists():
        shutil.copyfile(env, env.with_name(env.name + ".bak"))
    done = set()
    out = []
    for line in lines:
        m = re.match(r"\s*(MBAS_ASR|MBAS_FACE)\s*=", line)
        if m and m.group(1) not in done:
            out.append(f"{m.group(1)}={want[m.group(1)]}")
            done.add(m.group(1))
        elif m:
            continue                                               # a duplicate line for the same setting: drop it
        else:
            out.append(line)
    for k, v in want.items():
        if k not in done:
            out.append(f"{k}={v}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    return read_mode(env)


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    env = ROOT / ".env"
    if not args:
        print(describe(read_mode(env)))
        return 0
    if args[0] not in MODES:
        print(f"unknown mode '{args[0]}': use own or pretrained")
        return 2
    print(describe(set_mode(env, args[0])))
    print("Restart the workers (close their window, then .\\tasks.ps1 workers) for this to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
