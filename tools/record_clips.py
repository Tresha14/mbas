"""Record test clips for ml.speech.asr.eval_clips from your microphone.

    pip install sounddevice
    python -m tools.record_clips my_clips                  # read 20 interview-style sentences aloud
    python -m tools.record_clips my_clips --free          # answer questions in your own words, then type what you said
    python -m tools.record_clips --list-devices

For each clip: press Enter to start, speak, press Enter to stop. The folder gets name.wav (16 kHz mono) and
name.txt (the correct text). Scripted mode writes the text for you; free mode asks you to type it, which is a more
honest test of natural speech (ums, hedges, run-on sentences). Record in the same setup you use for interviews
(same mic, browser and room), and try a few clips with a bit of background noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SR = 16000

SENTENCES = [
    "Hello, my name is Aryan and I am a final year computer science student.",
    "I worked on a machine learning project where we trained a model from scratch.",
    "Our team of four built a web application using Python and React.",
    "I think the biggest challenge was handling noisy data from the microphone.",
    "We improved the accuracy from seventy percent to eighty five percent.",
    "In my last internship I wrote unit tests and fixed bugs in the backend.",
    "I am not sure, but I believe the latency was around two hundred milliseconds.",
    "The system records the interview, transcribes the speech and scores each answer.",
    "I prefer working in a team because we can share ideas and review each other's code.",
    "My strengths are problem solving, communication and learning new tools quickly.",
    "Basically, we used a database to store the sessions and a queue to pass the messages.",
    "I would like to work on products that use artificial intelligence responsibly.",
    "When there was a deadline, I planned the tasks and we finished two days early.",
    "The model was trained on public datasets and our own recordings.",
    "Could you please repeat the question? I did not catch the last part.",
    "I handled the deployment with Docker and monitored the logs every day.",
    "One mistake I made was not testing the audio pipeline with real users.",
    "Thank you for your time, I really enjoyed talking about the role.",
    "The candidate looked at the camera, spoke clearly and answered each question in order.",
    "Maybe we could improve the transcript by collecting more data from different accents.",
]

QUESTIONS = [
    "Tell me about yourself.",
    "Describe a project you are proud of and what your role was.",
    "What was the hardest technical problem you solved, and how?",
    "Tell me about a time you worked in a team and disagreed with someone.",
    "Why do you want this role?",
    "What are your strengths and weaknesses?",
    "How do you handle tight deadlines?",
    "Where do you see yourself in three years?",
]


def record(sd, device, seconds_hint: str) -> np.ndarray:
    chunks: list[np.ndarray] = []

    def cb(indata, frames, t, status):
        chunks.append(indata[:, 0].copy())

    input(f"  Press Enter to start recording ({seconds_hint}) ...")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=device, callback=cb):
        input("  Recording... press Enter to stop.")
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def check(x: np.ndarray) -> str | None:
    if len(x) < SR * 1.0:
        return "too short (under 1 s)"
    peak = float(np.abs(x).max())
    if peak < 0.02:
        return f"very quiet (peak {peak:.3f}); move closer to the mic or raise the input volume"
    if peak > 0.99:
        return "clipping (too loud); lower the input volume"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?")
    ap.add_argument("--free", action="store_true", help="answer questions freely and type the transcript yourself")
    ap.add_argument("--device", help="input device index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="first clip number")
    args = ap.parse_args(argv)
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Install the microphone library first:  pip install sounddevice")
        return 1
    if args.list_devices:
        print(sd.query_devices())
        return 0
    if args.folder is None:
        ap.error("folder is required")
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    args.folder.mkdir(parents=True, exist_ok=True)
    prompts = QUESTIONS if args.free else SENTENCES
    print(f"{len(prompts)} prompts. Ctrl+C any time; finished clips are kept.\n")
    n = args.start
    try:
        for p in prompts:
            name = f"clip{n:02d}"
            if (args.folder / f"{name}.wav").exists():
                n += 1
                continue
            print(f"[{name}] {'Question' if args.free else 'Read aloud'}: {p}")
            while True:
                x = record(sd, device, "speak naturally" if args.free else "read the sentence")
                problem = check(x)
                if problem:
                    print(f"  Problem: {problem}.")
                    if input("  Try again? [Y/n] ").strip().lower() != "n":
                        continue
                break
            if args.free:
                ref = input("  Type exactly what you said (Enter to skip this clip): ").strip()
                if not ref:
                    continue
            else:
                ref = p
            sf.write(str(args.folder / f"{name}.wav"), x, SR, subtype="PCM_16")
            (args.folder / f"{name}.txt").write_text(ref + "\n", encoding="utf-8")
            print(f"  saved {name}.wav ({len(x) / SR:.1f} s)\n")
            n += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"\nDone. Evaluate with:\n  python -m ml.speech.asr.eval_clips {args.folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
