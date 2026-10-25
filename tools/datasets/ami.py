"""AMI Meeting Corpus -> speaker turns + per-participant headset audio (meeting mode, Stage 8).

Downloads the manual annotations (ami_public_manual_1.6.2.zip, 23 MB) and, per
meeting, the individual headset WAVs (one per participant, like our separate
WebRTC tracks) and the headset mix.

Turns come from segments/<meeting>.<A-D>.segments.xml (transcriber start/end)
with the words from words/<meeting>.<A-D>.words.xml; <disfmarker/> and
filled-pause words are counted per turn. Speaker letters map to headset
channels and global speaker IDs through corpusResources/meetings.xml.

    python -m tools.datasets.ami                        # every annotated meeting (~170, ~35 GB audio)
    python -m tools.datasets.ami --max-meetings 10
    python -m tools.datasets.ami --audio none           # annotations only
Output: data/processed/public/ami/{manifest.parquet (one row per meeting+speaker), turns.parquet, meetings.parquet}
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from tools.datasets.common import dirs, download, extract, rel, write_outputs

NAME = "ami"
ANN_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANN_SIZE = 22887865
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{kind}.wav"
NITE = "{http://nite.sourceforge.net/}"
FILLED = {"um", "uh", "mm", "hmm", "mm-hmm", "uh-huh", "erm", "er"}


def speakers(meetings_xml: Path) -> dict[str, dict[str, dict]]:
    """meeting -> letter -> {channel, global_id, role}"""
    out: dict[str, dict[str, dict]] = {}
    for m in ET.parse(meetings_xml).getroot().iter("meeting"):
        obs = m.get("observation")
        for s in m.iter("speaker"):
            out.setdefault(obs, {})[s.get("nxt_agent")] = {
                "channel": s.get("channel"), "global_id": s.get("global_name"), "role": s.get("role")}
    return out


def read_words(path: Path) -> list[dict]:
    words = []
    for el in ET.parse(path).getroot():
        tag = el.tag.split("}")[-1]
        st, en = el.get("starttime"), el.get("endtime")
        if tag == "w" and st and en:
            words.append({"start": float(st), "end": float(en), "text": (el.text or "").strip(),
                          "punc": el.get("punc") == "true", "disf": False})
        elif tag == "disfmarker" and st:
            words.append({"start": float(st), "end": float(en or st), "text": "", "punc": False, "disf": True})
    return sorted(words, key=lambda w: w["start"])


def turns_for(meeting: str, letter: str, ann: Path) -> list[dict]:
    seg_f = ann / "segments" / f"{meeting}.{letter}.segments.xml"
    words_f = ann / "words" / f"{meeting}.{letter}.words.xml"
    if not seg_f.exists() or not words_f.exists():
        return []
    words = read_words(words_f)
    out = []
    for seg in ET.parse(seg_f).getroot().iter("segment"):
        st, en = seg.get("transcriber_start"), seg.get("transcriber_end")
        if st is None or en is None:
            continue
        s, e = float(st), float(en)
        ws = [w for w in words if w["start"] >= s - 0.01 and w["end"] <= e + 0.01]
        toks = [w["text"] for w in ws if w["text"] and not w["punc"]]
        out.append({"meeting_id": meeting, "speaker": letter, "start_s": s, "end_s": e,
                    "duration_s": round(e - s, 3), "text": " ".join(toks), "n_words": len(toks),
                    "n_filled_pauses": sum(t.lower() in FILLED for t in toks),
                    "n_disfmarkers": sum(w["disf"] for w in ws)})
    return out


def convert(ann: Path, proc_audio: Path, meetings: list[str] | None, audio: str) -> tuple[pd.DataFrame, ...]:
    spk = speakers(next(ann.rglob("meetings.xml")))
    annotated = sorted({re.match(r"(\w+)\.[A-E]\.segments", f.name).group(1)
                        for f in (ann / "segments").glob("*.segments.xml")})
    todo = [m for m in annotated if m in spk and (meetings is None or m in meetings)]
    rows, turns, meets = [], [], []
    for m in todo:
        t = []
        for letter, s in sorted(spk[m].items()):
            tt = turns_for(m, letter, ann)
            for x in tt:
                x["subject_id"] = s["global_id"]
            t += tt
            wav = None
            if audio in ("headsets", "all") and s["channel"] is not None:
                wav = download(AUDIO_URL.format(m=m, kind=f"Headset-{s['channel']}"),
                               proc_audio / m / f"{m}.Headset-{s['channel']}.wav", retries=3)
            rows.append({"item_id": f"{m}.{letter}", "subject_id": s["global_id"], "meeting_id": m,
                         "speaker": letter, "role": s["role"], "headset_channel": s["channel"],
                         "split_official": "all", "path": rel(wav) if wav else "",
                         "talk_time_s": round(sum(x["duration_s"] for x in tt), 2), "n_turns": len(tt),
                         "n_words": sum(x["n_words"] for x in tt)})
        mix = None
        if audio in ("mix", "all"):
            mix = download(AUDIO_URL.format(m=m, kind="Mix-Headset"), proc_audio / m / f"{m}.Mix-Headset.wav", retries=3)
        dur = max((x["end_s"] for x in t), default=0.0)
        total = sum(r["talk_time_s"] for r in rows if r["meeting_id"] == m) or 1.0
        for r in rows:
            if r["meeting_id"] == m:
                r["talk_share"] = round(r["talk_time_s"] / total, 4)
        meets.append({"meeting_id": m, "n_speakers": len(spk[m]), "duration_s": round(dur, 2),
                      "n_turns": len(t), "mix_path": rel(mix) if mix else ""})
        turns += t
        print(f"  {m}: {len(spk[m])} speakers, {len(t)} turns")
    turns_df = pd.DataFrame(turns).sort_values(["meeting_id", "start_s"]).reset_index(drop=True)
    return pd.DataFrame(rows), turns_df, pd.DataFrame(meets)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", nargs="*", help="only these meeting IDs (e.g. ES2002a IS1000a)")
    ap.add_argument("--max-meetings", type=int)
    ap.add_argument("--audio", choices=["all", "headsets", "mix", "none"], default="all")
    args = ap.parse_args(argv)
    downloads, extracted, proc = dirs(NAME)
    extract(download(ANN_URL, downloads / "ami_public_manual_1.6.2.zip", ANN_SIZE), extracted)
    ann = next(p.parent for p in extracted.rglob("segments") if p.is_dir())
    meetings = args.meetings
    if args.max_meetings:
        all_m = sorted({f.name.split(".")[0] for f in (ann / "segments").glob("*.segments.xml")})
        meetings = [m for m in all_m if meetings is None or m in meetings][: args.max_meetings]
    man, turns, meets = convert(ann, proc / "audio", meetings, args.audio)
    meets_dur = meets.set_index("meeting_id").duration_s
    man["meeting_duration_s"] = man.meeting_id.map(meets_dur)
    write_outputs(NAME, man, {
        "title": "AMI Meeting Corpus", "used_for": "meeting mode: talk balance, engagement (Stage 8)",
        "license": "CC BY 4.0", "source": "https://groups.inf.ed.ac.uk/ami/corpus/", "audio": args.audio,
        "n_meetings": int(len(meets)), "n_turns": int(len(turns)),
        "hours": round(float(meets.duration_s.sum()) / 3600, 2),
        "note": "hours = meeting hours; one manifest row per meeting x speaker; subject_id = AMI global speaker id",
    }, extra={"turns.parquet": turns, "meetings.parquet": meets})
    return 0


if __name__ == "__main__":
    sys.exit(main())
