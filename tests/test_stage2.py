"""Stage 2 tools: session index, Label Studio import/export, kappa, splits, dataset converters."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import soundfile as sf

from tools import agreement, export_labels, import_to_labelstudio as imp, make_splits, session_index
from tools.check_session import _ffmpeg

FIXTURES = Path(__file__).parent / "fixtures"
T0 = 1_760_000_000_000.0


# ---------------------------------------------------------------- session folders

def _session(tmp: Path, dur_s: float = 12.0, media: bool = False, pid: str = "P001", gender: str = "female") -> Path:
    d = tmp / f"R-{pid}-{int(dur_s)}"
    d.mkdir(parents=True)
    meta = {"session_id": d.name, "participants": {
        "candidate": {"participant_id": pid, "gender": gender, "age_band": "18-24", "accent": "hindi",
                      "glasses": False, "consent": {"agreed": True}},
        "interviewer": {"participant_id": "P900", "consent": {"agreed": True}}}}
    tracks = {"candidate_webcam": {"file": "candidate_webcam.webm", "kind": "video", "start_ms": T0 + 200,
                                   "end_ms": T0 + dur_s * 1000},
              "candidate_mic": {"file": "candidate_mic.wav", "kind": "audio", "start_ms": T0 + 100,
                                "end_ms": T0 + dur_s * 1000}}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "timeline.json").write_text(json.dumps({"recording": {"start_ms": T0, "end_ms": T0 + dur_s * 1000},
                                                 "tracks": tracks}))
    ev = [{"type": "ask", "question_id": "Q01", "t_ms": T0 + 1000, "question_text": "Tell me about yourself.",
           "category": "hr"},
          {"type": "answer_end", "question_id": "Q01", "t_ms": T0 + 6000},
          {"type": "ask", "question_id": "Q02", "t_ms": T0 + 7000, "question_text": "Why us?", "category": "hr"},
          {"type": "answer_end", "question_id": "Q02", "t_ms": T0 + 10500}]
    (d / "question_events.json").write_text(json.dumps({"events": ev}))
    if media:
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=size=640x360:rate=15:duration={dur_s - 0.2}", "-c:v", "libvpx", "-b:v", "300k",
                        "-deadline", "realtime", str(d / "candidate_webcam.webm")], check=True)
        t = np.arange(int((dur_s - 0.1) * 16000)) / 16000
        sf.write(str(d / "candidate_mic.wav"), (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 16000,
                 subtype="PCM_16")
    return d


def _duration(path: Path) -> float:
    p = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v", "-c", "copy", "-f", "framemd5", "-"],
                       capture_output=True, text=True)
    pts = [int(l.split(",")[2]) for l in p.stdout.splitlines() if l and not l.startswith("#")]
    dur = [int(l.split(",")[3]) for l in p.stdout.splitlines() if l and not l.startswith("#")]
    return (pts[-1] + dur[-1] - pts[0]) / 1000


def test_windows_and_answers(tmp_path):
    s = session_index.load_session(_session(tmp_path))
    assert [w.start_s for w in s.windows] == [0.0, 2.5, 5.0]  # 12 s: 7.5-12.5 does not fit and is dropped
    assert all(w.end_s - w.start_s == 5.0 for w in s.windows)
    assert [a.answer_id for a in s.answers] == [f"{s.session_id}_A01", f"{s.session_id}_A02"]
    assert s.answers[0].ask_s == 1.0 and s.answers[1].end_s == 10.5
    assert s.windows[0].answer_id == s.answers[0].answer_id      # 0-5 s overlaps A01 by 4 s
    assert s.windows[2].answer_id == s.answers[1].answer_id      # 5-10 overlaps A01 by 1 s, A02 by 3 s
    assert s.track_offsets_s == {"candidate_webcam": 0.2, "candidate_mic": 0.1}


def test_unpaired_events_are_ignored():
    ev = [{"type": "ask", "question_id": "Q1", "t_ms": 0}, {"type": "ask", "question_id": "Q2", "t_ms": 1000},
          {"type": "answer_end", "question_id": "Q2", "t_ms": 3000}, {"type": "answer_end", "question_id": "Q9", "t_ms": 4000}]
    a = session_index.answers_from_events("S", ev, 0)
    assert [(x.question_id, x.ask_s, x.end_s) for x in a] == [("Q2", 1.0, 3.0)]


# ---------------------------------------------------------------- Label Studio import

def test_build_session_clips(tmp_path):
    s = session_index.load_session(_session(tmp_path / "raw", dur_s=12.0, media=True))
    tasks = imp.build_session(s, tmp_path / "ls")
    assert len(tasks["windows"]) == 3 and len(tasks["answers"]) == 2
    media = tmp_path / "ls" / "media" / s.session_id
    for i in range(3):
        assert _duration(media / "windows" / f"W{i:04d}.webm") == pytest.approx(5.0, abs=0.08)
    a1 = sf.info(str(media / "answers" / "A01.wav"))
    assert a1.samplerate == 16000 and a1.duration == pytest.approx(5.5, abs=0.01)   # ask 1.0 -> end 6.0 + 0.5
    w = tasks["windows"][1]["data"]
    assert w["video"] == f"/data/local-files/?d=mbas/media/{s.session_id}/windows/W0001.webm"
    assert w["t0_s"] == 2.5 and w["participant_id"] == "P001"
    a = tasks["answers"][0]["data"]
    assert a["clip_t0_s"] == 1.0 and a["question_text"] == "Tell me about yourself."


def test_cut_audio_pads_when_track_starts_late(tmp_path):
    wav = tmp_path / "m.wav"
    sf.write(str(wav), np.ones(16000, dtype=np.int16), 16000, subtype="PCM_16")
    imp.cut_audio(wav, offset_s=0.5, start_s=0.0, end_s=1.0, dest=tmp_path / "c.wav")
    x, _ = sf.read(str(tmp_path / "c.wav"), dtype="int16")
    assert len(x) == 16000 and (x[:8000] == 0).all() and (x[8000:] == 1).all()


def test_double_assignment_is_stable():
    first = imp.assign_double([f"S{i:02d}" for i in range(10)], {})
    assert sum(first.values()) == 2
    more = imp.assign_double([f"S{i:02d}" for i in range(20)], first)
    assert all(more[k] == v for k, v in first.items())
    assert sum(more.values()) == 4
    assert sum(imp.assign_double(["only"], {}).values()) == 1


# ---------------------------------------------------------------- export + agreement

def test_export_real_labelstudio_json(tmp_path):
    fx = json.loads((FIXTURES / "labelstudio_export.json").read_text())
    win = export_labels.parse_windows(fx["windows"])
    ans, hes = export_labels.parse_answers(fx["answers"])
    assert len(win) == 8 and win.annotator.nunique() == 2         # 4 windows x 2 annotators
    assert set(win.attention_label) <= {"on_screen", "off_screen"}
    assert ans.quality_1to5.tolist() == [3, 4, 3, 4] or sorted(ans.quality_1to5) == [3, 3, 4, 4]
    assert (ans.follow_up_reason == "vague").all() and (ans.follow_up_needed == 1).all()
    clip0 = {t["data"]["answer_id"]: t["data"]["clip_t0_s"] for t in fx["answers"]}
    first = hes[hes.annotation_rank == 0].iloc[0]
    assert first.start_s == pytest.approx(clip0[first.answer_id] + 1.0)   # span is moved to session time
    counts = export_labels.write_all(win, ans, hes, tmp_path)
    assert counts["windows.parquet"] == 4 and counts["answers.parquet"] == 2
    assert counts["hesitation_events.parquet"] == 2
    assert (tmp_path / "windows_annotations.parquet").exists()


def test_cancelled_annotations_are_dropped():
    t = {"data": {"window_id": "S_W0000", "session_id": "S", "t0_s": 0, "t1_s": 5},
         "annotations": [{"id": 1, "completed_by": 1, "was_cancelled": True, "result": []}]}
    assert export_labels.parse_windows([t]).empty


def test_kappa_values():
    assert agreement.kappa(list("aabb"), list("aabb")) == 1.0
    assert agreement.kappa(list("aabb"), list("abab")) == 0.0
    assert agreement.kappa(list("aaaa"), list("aaaa")) is None       # undefined, one class only
    assert agreement.interpret(0.85) == "almost perfect" and agreement.interpret(0.5) == "moderate"


def test_agreement_end_to_end(tmp_path):
    rows = []
    for i in range(20):
        a = "on_screen" if i % 4 else "off_screen"
        b = a if i != 1 else "off_screen"
        for ann, lab in (("A", a), ("B", b)):
            rows.append({"window_id": f"S1_W{i:04d}", "session_id": "S1", "attention_label": lab,
                         "engagement_label": "neutral" if i % 3 else "engaged", "annotator": ann})
    rows.append({"window_id": "S2_W0000", "session_id": "S2", "attention_label": "on_screen",
                 "engagement_label": "neutral", "annotator": "A"})
    pd.DataFrame(rows).to_parquet(tmp_path / "windows_annotations.parquet")
    ans = pd.DataFrame([{"answer_id": f"S1_A0{i}", "session_id": "S1", "ask_s": 10.0 * i, "end_s": 10.0 * i + 8,
                         "quality_1to5": q, "relevance": r, "follow_up_needed": f, "follow_up_reason": fr,
                         "annotator": ann}
                        for i in range(1, 5) for ann in ("A", "B")
                        for q, r, f, fr in [(i if ann == "A" else min(i + 1, 5), 0.2 * i, i % 2,
                                             "vague" if i % 2 else "none")]])
    ans.to_parquet(tmp_path / "answers_annotations.parquet")
    hes = pd.DataFrame([{"answer_id": "S1_A01", "session_id": "S1", "event_type": "filler", "start_s": 12.0,
                         "end_s": 12.5, "annotator": ann} for ann in ("A", "B")])
    hes.to_parquet(tmp_path / "hesitation_annotations.parquet")
    res = agreement.compute(tmp_path)
    w = res["windows"][0]
    assert w["n_items"] == 20 and w["attention_agreement"] == 0.95
    assert 0.8 < w["attention_kappa"] < 1.0
    assert res["double_labelled_sessions"] == 1 and res["double_labelled_fraction"] == 0.5
    a = res["answers"][0]
    assert a["follow_up_kappa"] == 1.0 and a["relevance_mae"] == 0.0
    assert a["quality_kappa_quadratic"] > 0.5
    h = res["hesitation"][0]
    assert h["filler_frame_kappa"] == 1.0 and h["long_pause_frame_kappa"] is None


# ---------------------------------------------------------------- splits

def test_split_people_ratios_and_strata():
    p = pd.DataFrame({"participant_id": [f"P{i:03d}" for i in range(60)], "gender": ["male", "female"] * 30})
    a = make_splits.split_people(p)
    counts = pd.Series(a).value_counts().to_dict()
    assert counts == {"train": 42, "val": 9, "test": 9}
    by = pd.crosstab(p.gender, p.participant_id.map(a))
    assert (by["test"] >= 4).all() and (by["val"] >= 4).all()
    assert make_splits.split_people(p) == a                       # deterministic


def test_recorded_splits_no_leak(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for i in range(12):
        _session(raw, dur_s=10 + i, pid=f"P{i % 6:03d}", gender="male" if i % 2 else "female")
    monkeypatch.setattr(make_splits, "SPLITS", tmp_path / "splits")
    s = make_splits.recorded(raw)
    assert sum(s["participants"].values()) == 6 and sum(s["sessions"].values()) == 12
    assert s["leakage_problems"] == []
    ses = pd.read_csv(tmp_path / "splits" / "recorded" / "sessions.csv")
    assert (ses.groupby("participant_id").split.nunique() == 1).all()


def test_verify_detects_leak():
    df = pd.DataFrame({"session_id": ["a", "b"], "participant_id": ["P1", "P1"], "split": ["train", "test"]})
    assert make_splits.verify(df) == ["participant P1 is in 2 splits"]


def test_synthetic_splits_and_labels(tmp_path, monkeypatch):
    from tools import synthetic_to_labels

    if not synthetic_to_labels.SYN.exists():
        pytest.skip("synthetic dataset not extracted")
    counts = synthetic_to_labels.convert(out=tmp_path / "labels")
    assert counts["windows.parquet"] == 46234 and counts["answers.parquet"] == 3179
    a = pd.read_parquet(tmp_path / "labels" / "answers_annotations.parquet")
    assert set(a.annotator) == {"ANN1", "ANN2"}
    assert not any(c in a.columns for c in ("confidence", "nervousness", "preparedness"))  # HIDDEN columns
    monkeypatch.setattr(make_splits, "SPLITS", tmp_path / "splits")
    s = make_splits.synthetic()
    assert s["participants"] == {"train": 84, "val": 18, "test": 18} and s["leakage_problems"] == []


# ---------------------------------------------------------------- public dataset converters

def test_wider_face_gt_parser(tmp_path):
    from tools.datasets import wider_face

    gt = tmp_path / "gt.txt"
    gt.write_text("0--Parade/a.jpg\n2\n1 2 30 40 0 0 0 0 0 0\n5 6 7 8 2 0 0 1 0 0\n"
                  "1--Handshaking/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n")
    e = wider_face.parse_gt(gt)
    assert [x["name"] for x in e] == ["0--Parade/a.jpg", "1--Handshaking/b.jpg"]
    assert e[0]["boxes"][1] == [5, 6, 7, 8, 2, 0, 0, 1, 0, 0] and e[1]["boxes"] == []


def test_mpiigaze_convert(tmp_path):
    from tools.datasets import mpiigaze

    n = 3
    g = np.array([[0.0, 0.0, -1.0]] * n)               # straight ahead -> pitch 0, yaw 0
    side = {"image": np.zeros((n, 36, 60), np.uint8), "gaze": g, "pose": np.zeros((n, 3))}
    d = tmp_path / "MPIIGaze" / "Data" / "Normalized" / "p00"
    d.mkdir(parents=True)
    scipy.io.savemat(str(d / "day01.mat"), {"data": {"left": side, "right": side}, "filenames": np.array(["a"] * n)})
    m = mpiigaze.convert(tmp_path, tmp_path)
    assert len(m) == 3 and m.subject_id.unique().tolist() == ["p00"]
    assert np.allclose(m.gaze_pitch_deg, 0) and np.allclose(m.gaze_yaw_deg, 0)
    z = np.load(tmp_path / "p00.npz")
    assert z["left_image"].shape == (3, 36, 60)
    ang = mpiigaze.gaze_to_angles(np.array([[0.0, -np.sin(0.2), -np.cos(0.2)]]))
    assert ang[0, 0] == pytest.approx(0.2, abs=1e-6)   # looking up = positive pitch


def test_columbia_names(tmp_path):
    from tools.datasets import columbia_gaze

    assert columbia_gaze.parse_name("0001_2m_-15P_10V_-5H.jpg") == {
        "subject_id": "0001", "distance_m": 2, "head_pose_deg": -15, "gaze_v_deg": 10, "gaze_h_deg": -5}
    import cv2

    src = tmp_path / "src" / "0003"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "0003_2m_0P_0V_0H.jpg"), np.zeros((200, 3000, 3), np.uint8))
    m = columbia_gaze.convert(tmp_path / "src", tmp_path / "out", max_width=1280)
    assert m.looking_at_camera.tolist() == [True]
    assert cv2.imread(str(tmp_path / "out" / "images" / "0003" / "0003_2m_0P_0V_0H.jpg")).shape[1] == 1280


def test_daisee_convert(tmp_path):
    from tools.datasets import daisee

    (tmp_path / "Labels").mkdir()
    (tmp_path / "Labels" / "AllLabels.csv").write_text(
        "ClipID,Boredom,Engagement,Confusion,Frustration \n1100011002.avi,0,2,0,0\n1100011003.avi,1,3,1,0\n")
    for split, clip in (("Train", "1100011002"), ("Test", "1100011003")):
        f = tmp_path / "DataSet" / split / "110001" / clip / f"{clip}.avi"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x")
    m = daisee.convert(tmp_path).set_index("item_id")
    assert m.loc["1100011002", "engagement_label"] == "neutral" and m.loc["1100011003", "engagement_label"] == "engaged"
    assert m.loc["1100011003", "split_official"] == "test" and m.loc["1100011002", "subject_id"] == "110001"


def test_librispeech_convert(tmp_path):
    from tools.datasets import librispeech

    d = tmp_path / "LibriSpeech" / "dev-clean" / "84" / "121123"
    d.mkdir(parents=True)
    sf.write(str(d / "84-121123-0000.flac"), np.zeros(16000, np.float32), 16000)
    (d / "84-121123.trans.txt").write_text("84-121123-0000 GO DO YOU HEAR\n")
    (tmp_path / "LibriSpeech" / "SPEAKERS.TXT").write_text(";ID|SEX|SUBSET\n84 | F | dev-clean\n")
    m = librispeech.convert(tmp_path, ["dev-clean"])
    r = m.iloc[0]
    assert (r.item_id, r.subject_id, r.split_official, r.text, r.gender) == ("84-121123-0000", "84", "val", "GO DO YOU HEAR", "F")
    assert r.duration_s == pytest.approx(1.0)


def test_common_voice_filter_and_convert(tmp_path):
    from tools.datasets import common_voice

    clips = tmp_path / "en" / "clips"
    clips.mkdir(parents=True)
    for n in ("a.mp3", "b.mp3"):
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1",
                        "-ar", "48000", str(clips / n)], check=True)
    tsv = "client_id\tpath\tsentence\taccents\tgender\tage\n" \
          "c1\ta.mp3\thello there\tIndia and South Asia (India, Pakistan, Sri Lanka)\tmale\ttwenties\n" \
          "c2\tb.mp3\tgood day\tUnited States English\tfemale\tthirties\n"
    (tmp_path / "en" / "validated.tsv").write_text(tsv)
    (tmp_path / "en" / "train.tsv").write_text(tsv)
    validated, split, cdir = common_voice.from_dir(tmp_path / "en")
    m = common_voice.convert(validated, split, cdir, tmp_path / "out", workers=1)
    assert m.item_id.tolist() == ["a"] and m.split_official.tolist() == ["train"]
    info = sf.info(str(tmp_path / "out" / "wav" / "a.wav"))
    assert info.samplerate == 16000 and info.channels == 1


def test_sep28k_labels_and_clips(tmp_path):
    from tools.datasets import sep28k

    sf.write(str(tmp_path / "ep.wav"), np.zeros(16000 * 10, np.int16), 16000, subtype="PCM_16")
    lab = pd.DataFrame([{"Show": "HeStutters", "EpId": 0, "ClipId": 1, "Start": 16000, "Stop": 64000, "Unsure": 0,
                         "PoorAudioQuality": 0, "Prolongation": 1, "Block": 2, "SoundRep": 0, "WordRep": 3,
                         "DifficultToUnderstand": 0, "Interjection": 2, "NoStutteredWords": 0, "NaturalPause": 0,
                         "Music": 0, "NoSpeech": 0}])
    rows = sep28k.cut_clips(tmp_path / "ep.wav", lab, tmp_path)
    r = rows[0]
    assert (r["filler"], r["repetition"], r["block"], r["prolongation"]) == (1, 1, 1, 0)
    assert r["duration_s"] == 3.0 and sf.info(str(tmp_path / "HeStutters_0_1.wav")).frames == 48000
    assert sep28k.cache_rows(lab, tmp_path)[0]["item_id"] == "HeStutters_0_1"


def test_mit_interview_convert(tmp_path):
    from tools.datasets import mit_interview

    (tmp_path / "Labels").mkdir()
    (tmp_path / "Labels" / "turker_scores_full_interview.csv").write_text(
        "Worker,Participant,Overall,RecommendHiring,Engaged\nW1,p1,5,5,4\nW2,p1,3,3,2\nAGGR,p1,4.2,4.1,3.3\n"
        "W1,pp1,6,6,6\nAGGR,pp1,5.5,5.0,5.1\n")
    (tmp_path / "interview_transcripts_by_turkers.csv").write_text(
        'p1,"Interviewer:Hi|Interviewee:Hello I am a student"\npp1,"Interviewer:Hi|Interviewee:Hey"\n')
    (tmp_path / "Videos").mkdir()
    (tmp_path / "Videos" / "P1.avi").write_bytes(b"x")
    m = mit_interview.convert(tmp_path).set_index("item_id")
    assert m.loc["P1", "rating_overall"] == 4.2 and m.loc["PP1", "subject_id"] == "P1"
    assert m.loc["P1", "n_turns"] == 2 and m.loc["P1", "path"].endswith("Videos/P1.avi")
    assert m.loc["PP1", "round"] == 2


def test_ami_turns(tmp_path):
    from tools.datasets import ami

    ann = tmp_path / "ann"
    for sub in ("segments", "words", "corpusResources"):
        (ann / sub).mkdir(parents=True)
    nite = 'xmlns:nite="http://nite.sourceforge.net/"'
    (ann / "corpusResources" / "meetings.xml").write_text(
        f'<meetings {nite}><meeting observation="ES2002a"><speaker nxt_agent="A" channel="1" global_name="FEE005" '
        f'role="PM"/><speaker nxt_agent="B" channel="0" global_name="MEE006" role="UI"/></meeting></meetings>')
    for spk, words in (("A", [("0.5", "1.0", "um"), ("1.0", "1.5", "hello"), ("1.5", "1.5", ".")]),
                       ("B", [("2.0", "2.4", "yes")])):
        punc = ' punc="true"'
        w = "".join(f'<w nite:id="w{i}" starttime="{s}" endtime="{e}"{punc if t == "." else ""}>{t}</w>'
                    for i, (s, e, t) in enumerate(words))
        (ann / "words" / f"ES2002a.{spk}.words.xml").write_text(
            f'<nite:root {nite}>{w}<disfmarker nite:id="d1" starttime="1.2" endtime="1.2"/></nite:root>')
        st, en = words[0][0], words[-1][1]
        (ann / "segments" / f"ES2002a.{spk}.segments.xml").write_text(
            f'<nite:root {nite}><segment nite:id="s1" transcriber_start="{st}" transcriber_end="{en}"/></nite:root>')
    man, turns, meets = ami.convert(ann, tmp_path / "audio", None, "none")
    a = turns[turns.speaker == "A"].iloc[0]
    assert a.text == "um hello" and a.n_filled_pauses == 1 and a.n_disfmarkers == 1 and a.subject_id == "FEE005"
    assert man.set_index("speaker").loc["B", "headset_channel"] == "0"
    assert man.talk_share.sum() == pytest.approx(1.0) and meets.n_speakers.tolist() == [2]


# ---------------------------------------------------------------- dataset card

def test_dataset_card_renders():
    from tools import dataset_stats

    stats = {"generated": "2026-01-01", "recorded": {"sessions": 0, "participants": 0, "labels": {}, "agreement": {},
                                                    "splits": {}},
             "synthetic": {}, "public": {"ami": {"status": "converted", "title": "AMI", "n_items": 3}}}
    text = dataset_stats.render(stats)
    assert "# MBAS dataset card" in text and "- [ ] At least 60 sessions" in text and "| AMI |" in text
