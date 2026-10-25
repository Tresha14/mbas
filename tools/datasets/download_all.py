"""Download + convert every public dataset that has a direct download, one after another.

Manual-access datasets (Columbia Gaze, DAiSEE, Common Voice, MIT Interview) need a
request or login first; this prints the command to run once you have the file.
Re-running is safe: finished downloads, extractions and conversions are skipped.

    python -m tools.datasets.download_all
    python -m tools.datasets.download_all --only librispeech ami
Total download: about 45 GB (AMI headset audio is ~35 GB of it; use --ami-audio none to skip).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback

AUTO = ["wider_face", "mpiigaze", "librispeech", "sep28k", "ami"]
MANUAL = {
    "columbia_gaze": 'python -m tools.datasets.columbia_gaze --archive "<Columbia Gaze Data Set.zip>"',
    "daisee": "python -m tools.datasets.daisee --archive <DAiSEE.zip>",
    "common_voice": "python -m tools.datasets.common_voice --archive <cv-corpus-XX-en.tar.gz>",
    "mit_interview": "python -m tools.datasets.mit_interview --archive <MIT interview zip>",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=AUTO)
    ap.add_argument("--ami-audio", default="all", choices=["all", "headsets", "mix", "none"])
    args = ap.parse_args(argv)
    failed = []
    for name in args.only or AUTO:
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        try:
            mod = importlib.import_module(f"tools.datasets.{name}")
            mod.main(["--audio", args.ami_audio] if name == "ami" else [])
            print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - keep going with the other datasets
            failed.append(name)
            print(f"=== {name} FAILED: {e}", flush=True)
            traceback.print_exc()
    print("\nManual-access datasets (request access, then run):")
    for name, cmd in MANUAL.items():
        print(f"  {name:14s} {cmd}")
    if failed:
        print(f"\nfailed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
