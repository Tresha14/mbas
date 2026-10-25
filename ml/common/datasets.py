"""PyTorch Dataset classes for every model. All read person-wise splits from data/splits/.

  FaceDataset            WIDER FACE (+ RetinaFace landmarks)      image 128x128, boxes, 5 key points
  GazeDataset            MPIIGaze normalised eye crops             2 x (1, 36, 60), head pose, gaze
  EngagementClipDataset  DAiSEE clips / our 5 s windows            (16, 3, 112, 112), engagement 3-class
  VADDataset             LibriSpeech mixtures + noise              waveform, 10 ms frame labels
  ASRDataset             LibriSpeech / Common Voice / our answers  waveform, token ids
  DisfluencyDataset      SEP-28k 3 s clips (+ our spans)           waveform, 4 multi-labels
  AnswerDataset          answers (ours / synthetic)                token ids, relevance, quality, STAR ...
  FusionDataset          per-answer modality feature vectors       5 modality tensors, scores, follow-up
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.common.audio_features import SR, add_noise, load_audio, speech_mask, synthetic_noise
from ml.common.config import LABELS, PUBLIC, RAW, SPLITS, SYNTHETIC, abspath
from ml.common.video_features import Augment, letterbox, normalize, sample_clip

SPLIT_NAMES = ("train", "val", "test")


def _hash01(key: str, salt: str = "") -> float:
    return int(hashlib.sha1(f"{salt}{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(source: str) -> dict[str, str]:
    """id -> split from data/splits/<source>/sessions.csv (session_id/item_id column)."""
    p = SPLITS / source / "sessions.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python -m tools.make_splits --source {source}")
    s = pd.read_csv(p, dtype=str)
    return dict(zip(s.session_id, s.split))


def participants_of(source: str, split: str) -> set[str]:
    p = SPLITS / source / f"{split}_participants.csv"
    return set(pd.read_csv(p, dtype=str).participant_id) if p.exists() else set()


def manifest(name: str) -> pd.DataFrame:
    p = PUBLIC / name / "manifest.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python -m tools.datasets.{name}")
    return pd.read_parquet(p)


def pad_1d(seqs: list[torch.Tensor], value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    out = torch.full((len(seqs), int(lens.max()) if len(seqs) else 0), value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lens


# ---------------------------------------------------------------------------------------------- vision

N_KP = 5  # left eye, right eye, nose, left mouth corner, right mouth corner (RetinaFace order)
FLIP_KP = [1, 0, 2, 4, 3]


class FaceDataset(Dataset):
    """WIDER FACE images letterboxed to `size`.

    Returns image (3, size, size), boxes (N, 4) normalised x1 y1 x2 y2, keypoints (N, 5, 2) normalised and
    kp_mask (N,) = 1 where the face has RetinaFace landmarks (train images only).
    Splits: WIDER train -> train (90%) / val (10%, by image hash); WIDER val -> test.
    Faces smaller than `min_face` px in the original image and 'invalid' faces are dropped.
    """

    def __init__(self, split: str, size: int = 128, augment: bool = False, min_face: int = 10, limit: int | None = None):
        import json

        m = manifest("wider_face")
        if split == "test":
            m = m[m.split_official == "val"]
        else:
            m = m[m.split_official == "train"]
            is_val = m.item_id.map(lambda k: _hash01(k, "face") < 0.1)
            m = m[is_val] if split == "val" else m[~is_val]
        self.items = []
        has_lm = "landmarks" in m.columns
        for r in m.itertuples(index=False):
            lm = json.loads(r.landmarks) if has_lm else []
            if lm:   # RetinaFace boxes + landmarks (same faces as the official train boxes)
                # RetinaFace writes -1 for a missing landmark: such a face keeps its box but gets no landmark target
                faces = np.array([f[:14] + [f[14] if min(f[4:14]) >= 0 else 0.0] for f in lm
                                  if min(f[2], f[3]) >= min_face], np.float32)
            else:
                faces = np.array([b[:4] + [0.0] * 10 + [0.0] for b in json.loads(r.boxes)
                                  if b[7] == 0 and min(b[2], b[3]) >= min_face], np.float32)
            faces = faces.reshape(-1, 15)
            if len(faces) or split != "train":
                self.items.append((r.path, faces))
        if limit:
            self.items = self.items[:limit]
        self.size, self.augment = size, augment
        self.aug = Augment(rotate_deg=0.0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import cv2

        path, faces = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(abspath(path))), cv2.COLOR_BGR2RGB)
        f = faces.copy()
        if self.augment:
            img, f = self._random_crop(img, f)
            img = self.aug.photometric(img)
            if self.aug.flip():
                w = img.shape[1]
                img = np.ascontiguousarray(img[:, ::-1])
                f[:, 0] = w - f[:, 0] - f[:, 2]
                kp = f[:, 4:14].reshape(-1, 5, 2)
                kp[:, :, 0] = w - kp[:, :, 0]
                f[:, 4:14] = kp[:, FLIP_KP].reshape(-1, 10)
        lb, s, (px, py) = letterbox(img, self.size)
        b = f[:, :4]
        xyxy = np.stack([b[:, 0] * s + px, b[:, 1] * s + py, (b[:, 0] + b[:, 2]) * s + px,
                         (b[:, 1] + b[:, 3]) * s + py], 1) / self.size
        kp = f[:, 4:14].reshape(-1, 5, 2).copy()
        kp[:, :, 0] = (kp[:, :, 0] * s + px) / self.size
        kp[:, :, 1] = (kp[:, :, 1] * s + py) / self.size
        mask = f[:, 14].copy()
        xyxy = np.clip(xyxy, 0, 1).reshape(-1, 4)
        keep = ((xyxy[:, 2] - xyxy[:, 0]) * self.size >= 2) & ((xyxy[:, 3] - xyxy[:, 1]) * self.size >= 2)
        mask[(kp < 0).any(axis=(1, 2)) | (kp > 1).any(axis=(1, 2))] = 0   # landmark cut off by the crop
        return (torch.from_numpy(normalize(lb)), torch.from_numpy(xyxy[keep].astype(np.float32)),
                torch.from_numpy(kp[keep].astype(np.float32)), torch.from_numpy(mask[keep].astype(np.float32)))

    def _random_crop(self, img: np.ndarray, faces: np.ndarray, tries: int = 10):
        """Square crop around a random face (scale jitter) so webcam-sized faces dominate training."""
        h, w = img.shape[:2]
        rng = self.aug.rng
        if not len(faces):
            return img, faces
        for _ in range(tries):
            b = faces[rng.integers(len(faces))]
            side = min(max(b[2], b[3]) * rng.uniform(1.4, 6.0), w, h)
            cx = b[0] + b[2] / 2 + rng.uniform(-0.3, 0.3) * side
            cy = b[1] + b[3] / 2 + rng.uniform(-0.3, 0.3) * side
            x0 = int(np.clip(cx - side / 2, 0, w - side))
            y0 = int(np.clip(cy - side / 2, 0, h - side))
            x1, y1 = x0 + int(side), y0 + int(side)
            cxs, cys = faces[:, 0] + faces[:, 2] / 2, faces[:, 1] + faces[:, 3] / 2
            inside = (cxs > x0) & (cxs < x1) & (cys > y0) & (cys < y1)
            if inside.any():
                nf = faces[inside].copy()
                bx0, by0 = np.maximum(nf[:, 0] - x0, 0), np.maximum(nf[:, 1] - y0, 0)
                bx1 = np.minimum(nf[:, 0] + nf[:, 2] - x0, x1 - x0)
                by1 = np.minimum(nf[:, 1] + nf[:, 3] - y0, y1 - y0)
                nf[:, 0], nf[:, 1], nf[:, 2], nf[:, 3] = bx0, by0, bx1 - bx0, by1 - by0
                nf[:, 4:14:2] -= x0
                nf[:, 5:14:2] -= y0
                return img[y0:y1, x0:x1], nf
        return img, faces


def face_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]


class GazeDataset(Dataset):
    """MPIIGaze: left + right 36x60 eye crops, head pose and gaze (pitch, yaw in radians).

    Person-wise split from data/splits/public/mpiigaze (falls back to p00-p10 / p11-p12 / p13-p14).
    MPIIGaze has no off-screen samples, so `onscreen_mask` is 0 (the on/off head is trained on our data).
    """

    def __init__(self, split: str, augment: bool = False, persons: list[str] | None = None, limit: int | None = None):
        root = PUBLIC / "mpiigaze"
        all_p = sorted(p.stem for p in root.glob("p*.npz"))
        if persons is None:
            try:
                sp = participants_of("public/mpiigaze", split)
                persons = sorted(sp) if sp else None
            except FileNotFoundError:
                persons = None
        if persons is None:
            default = {"train": all_p[:11], "val": all_p[11:13], "test": all_p[13:]}
            persons = default[split]
        li, ri, g, h, pid = [], [], [], [], []
        for p in persons:
            z = np.load(root / f"{p}.npz")
            li.append(z["left_image"])
            ri.append(z["right_image"])
            g.append((z["left_gaze"] + z["right_gaze"]) / 2)
            h.append((z["left_head"] + z["right_head"]) / 2)
            pid += [p] * len(z["left_image"])
        self.left, self.right = np.concatenate(li), np.concatenate(ri)
        self.gaze, self.head = np.concatenate(g).astype(np.float32), np.concatenate(h).astype(np.float32)
        self.person = np.array(pid)
        if limit:
            sel = np.random.default_rng(0).permutation(len(self.left))[:limit]
            self.left, self.right, self.gaze, self.head, self.person = (
                self.left[sel], self.right[sel], self.gaze[sel], self.head[sel], self.person[sel])
        self.augment = augment
        self.aug = Augment(rotate_deg=5.0, blur_p=0.1)

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, i: int):
        l, r, g, h = self.left[i], self.right[i], self.gaze[i].copy(), self.head[i].copy()
        if self.augment:
            l, r = self.aug.photometric(l), self.aug.photometric(r)
            if self.aug.flip():  # mirror: swap eyes, flip each crop, negate yaw
                l, r = np.ascontiguousarray(r[:, ::-1]), np.ascontiguousarray(l[:, ::-1])
                g[1], h[1] = -g[1], -h[1]
        f = lambda x: torch.from_numpy(x.astype(np.float32)[None] / 127.5 - 1.0)  # noqa: E731
        return {"left": f(l), "right": f(r), "head": torch.from_numpy(h), "gaze": torch.from_numpy(g),
                "onscreen": torch.tensor(1.0), "onscreen_mask": torch.tensor(0.0)}


ENGAGEMENT_CLASSES = ("disengaged", "neutral", "engaged")


class EngagementClipDataset(Dataset):
    """16 frames x 112x112 per clip -> engagement (0 disengaged, 1 neutral, 2 engaged) + boredom/confusion 0-3.

    source='daisee'   DAiSEE manifest (person-wise split by subject from data/splits/public/daisee)
    source='recorded' our labelled 5 s windows (webcam video, split from data/splits/recorded)
    """

    def __init__(self, split: str, source: str = "daisee", n_frames: int = 16, size: int = 112,
                 augment: bool = False, limit: int | None = None):
        if source == "daisee":
            m = manifest("daisee")
            try:
                sp = split_of("public/daisee")
                m = m[m.item_id.map(sp) == split]
            except FileNotFoundError:
                m = m[m.split_official == split]
            self.items = [(r.path, 0.0, None, ENGAGEMENT_CLASSES.index(r.engagement_label), r.boredom_level,
                           r.confusion_level) for r in m.itertuples(index=False)]
        else:
            from ml.common.windowing import recorded_windows
            from tools.session_index import load_session

            w = recorded_windows()
            w = w[(w.split == split) & w.engagement_label.notna()]
            offs = {}
            self.items = []
            for r in w.itertuples(index=False):
                if r.session_id not in offs:
                    offs[r.session_id] = load_session(RAW / r.session_id).track_offsets_s.get("candidate_webcam", 0.0)
                path = (RAW / r.session_id / "candidate_webcam.webm").as_posix()
                self.items.append((path, max(r.start_s - offs[r.session_id], 0.0), 5.0,
                                   ENGAGEMENT_CLASSES.index(r.engagement_label), -1, -1))
        if limit:
            self.items = self.items[:limit]
        self.n_frames, self.size, self.augment = n_frames, size, augment
        self.aug = Augment(rotate_deg=6.0)

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.array([it[3] for it in self.items])

    def __getitem__(self, i: int):
        path, start, dur, eng, bored, conf = self.items[i]
        clip = sample_clip(abspath(path), self.n_frames, (self.size, self.size), start, dur)
        if self.augment:
            clip = self.aug.frames(clip)
        return {"frames": torch.from_numpy(normalize(clip)), "engagement": torch.tensor(eng),
                "boredom": torch.tensor(bored), "confusion": torch.tensor(conf)}


# ---------------------------------------------------------------------------------------------- speech

def _speech_manifest(split: str, sources: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    if "librispeech" in sources and (PUBLIC / "librispeech" / "manifest.parquet").exists():
        m = manifest("librispeech")
        parts.append(m[m.split_official == split][["item_id", "subject_id", "path", "duration_s", "text"]]
                     .assign(source="librispeech"))
    if "common_voice_in" in sources and (PUBLIC / "common_voice_in" / "manifest.parquet").exists():
        m = manifest("common_voice_in")
        try:  # person-wise split by speaker (client_id)
            sp = split_of("public/common_voice_in")
            m = m[m.item_id.map(sp) == split]
        except FileNotFoundError:
            m = m[m.split_official == split]
        parts.append(m[["item_id", "subject_id", "path", "duration_s", "text"]].assign(source="common_voice_in"))
    if "recorded" in sources:
        parts.append(recorded_transcripts(split))
    if not parts:
        raise FileNotFoundError("no speech data: run python -m tools.datasets.librispeech")
    return pd.concat(parts, ignore_index=True)


def recorded_transcripts(split: str) -> pd.DataFrame:
    """Our manually transcribed answers (audio = the answer clip cut for Label Studio)."""
    cols = ["item_id", "subject_id", "path", "duration_s", "text"]
    p = LABELS / "recorded" / "answers.parquet"
    if not p.exists():
        return pd.DataFrame(columns=cols + ["source"])
    a = pd.read_parquet(p)
    a = a[a.transcript.fillna("").str.strip() != ""]
    try:
        sp = split_of("recorded")
        a = a[a.session_id.map(sp) == split]
    except FileNotFoundError:
        return pd.DataFrame(columns=cols + ["source"])
    media = Path("data/processed/labelstudio/media")
    rows = [{"item_id": r.answer_id, "subject_id": r.participant_id,
             "path": (media / r.session_id / "answers" / f"A{int(r.answer_id.rsplit('_A', 1)[1]):02d}.wav").as_posix(),
             "duration_s": float(r.end_s - r.ask_s + 0.5), "text": r.transcript} for r in a.itertuples(index=False)]
    return pd.DataFrame(rows, columns=cols).assign(source="recorded")


class ASRDataset(Dataset):
    """Waveform (float32, 16 kHz) + SentencePiece ids of the normalised transcript.

    Log-mel is computed on the GPU inside the model (same front end in ONNX). Speed perturbation
    0.9 / 1.0 / 1.1 when augment=True. Utterances longer than max_s are skipped.
    """

    def __init__(self, split: str, tokenizer, sources: tuple[str, ...] = ("librispeech", "common_voice_in", "recorded"),
                 max_s: float = 17.0, augment: bool = False, limit: int | None = None):
        from ml.common.text import normalize_asr

        m = _speech_manifest(split, sources)
        m = m[(m.duration_s <= max_s) & (m.duration_s > 0.3)].sort_values("duration_s").reset_index(drop=True)
        if limit:
            m = m.iloc[np.linspace(0, len(m) - 1, min(limit, len(m))).astype(int)].reset_index(drop=True)
        m["norm"] = m.text.map(normalize_asr)
        self.m = m[m.norm.str.len() > 0].reset_index(drop=True)
        self.tok, self.augment = tokenizer, augment
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.m)

    def durations(self) -> np.ndarray:
        return self.m.duration_s.to_numpy()

    def __getitem__(self, i: int):
        from ml.common.audio_features import speed_perturb

        r = self.m.iloc[i]
        x = load_audio(abspath(r.path))
        if self.augment:
            x = speed_perturb(x, float(self.rng.choice([0.9, 1.0, 1.1])))
        return {"wav": torch.from_numpy(x), "tokens": torch.tensor(self.tok.encode(r.norm), dtype=torch.long),
                "text": r.norm, "item_id": r.item_id, "source": r.source}


def asr_collate(batch):
    wav, wav_len = pad_1d([b["wav"] for b in batch])
    tok, tok_len = pad_1d([b["tokens"] for b in batch], 0)
    return {"wav": wav, "wav_len": wav_len, "tokens": tok, "tok_len": tok_len,
            "text": [b["text"] for b in batch], "item_id": [b["item_id"] for b in batch],
            "source": [b["source"] for b in batch]}


class VADDataset(Dataset):
    """Synthetic conversations: LibriSpeech utterances separated by silences, mixed with noise.

    Frame targets (10 ms) come from the clean utterances' energy mask. val/test items are
    deterministic (seeded by index). Noise: data/raw/noise/*.wav if present (our recorded
    lab/hostel noise), else generated room-like noise.
    """

    def __init__(self, split: str, seconds: float = 8.0, n_items: int | None = None, snr_db: tuple[float, float] = (0, 25),
                 limit: int | None = None):
        m = _speech_manifest(split, ("librispeech", "common_voice_in"))
        self.paths = m.path.tolist()
        self.seconds, self.snr = seconds, snr_db
        self.n = limit or n_items or {"train": 20000, "val": 1000, "test": 2000}[split]
        self.split = split
        self.noise = [load_audio(p) for p in sorted((RAW / "noise").glob("*.wav"))] if (RAW / "noise").exists() else []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(None if self.split == "train" else 1000 + i)
        n = int(self.seconds * SR)
        x = np.zeros(n, np.float32)
        lab = np.zeros(n // 160 + 1, bool)
        pos = int(rng.uniform(0, 1.5) * SR)
        while pos < n - SR // 2:
            u = load_audio(abspath(self.paths[int(rng.integers(len(self.paths)))]))
            u = u[: n - pos] * rng.uniform(0.3, 1.0) / (np.abs(u).max() + 1e-6)
            x[pos:pos + len(u)] += u
            mask = speech_mask(u)
            f0 = pos // 160
            lab[f0:f0 + len(mask)] |= mask[: len(lab) - f0]
            pos += len(u) + int(rng.uniform(0.2, 2.5) * SR)
        noise = self.noise[int(rng.integers(len(self.noise)))] if self.noise and rng.random() < 0.5 \
            else synthetic_noise(n, rng)
        x = add_noise(x, noise, float(rng.uniform(*self.snr)), rng) if rng.random() < 0.9 else x
        frames = n // 160 + 1
        return {"wav": torch.from_numpy(x), "labels": torch.from_numpy(lab[:frames].astype(np.float32))}


DISFLUENCY_LABELS = ("filler", "repetition", "prolongation", "block")


class DisfluencyDataset(Dataset):
    """3 s clips -> 4 binary labels (filler, repetition, prolongation, block).

    SEP-28k clips (unusable audio dropped), split by episode via data/splits/public/sep28k.
    """

    def __init__(self, split: str, limit: int | None = None, augment: bool = False):
        m = manifest("sep28k")
        m = m[m.unusable == 0]
        try:
            sp = split_of("public/sep28k")
            m = m[m.item_id.map(sp) == split]
        except FileNotFoundError:
            m = m[m.item_id.map(lambda k: _hash01(k.rsplit("_", 1)[0], "sep")).between(
                *{"train": (0, 0.8), "val": (0.8, 0.9), "test": (0.9, 1.01)}[split])]
        self.m = m.reset_index(drop=True)
        if limit:
            self.m = self.m.iloc[:limit]
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.m)

    def targets(self) -> np.ndarray:
        return self.m[list(DISFLUENCY_LABELS)].to_numpy(np.float32)

    def __getitem__(self, i: int):
        r = self.m.iloc[i]
        x = load_audio(abspath(r.path))
        x = np.pad(x, (0, max(0, 3 * SR - len(x))))[: 3 * SR]
        if self.augment and self.rng.random() < 0.5:
            x = add_noise(x, synthetic_noise(len(x), self.rng), float(self.rng.uniform(5, 30)), self.rng)
        y = torch.tensor([float(getattr(r, k)) for k in DISFLUENCY_LABELS])
        return {"wav": torch.from_numpy(x), "labels": y}


# ---------------------------------------------------------------------------------------------- response + fusion

FOLLOW_UP_REASONS = ("none", "vague", "low_confidence", "attention_drop", "topic_not_covered", "contradiction")
SPEAKING_PATTERNS = ("confident", "balanced", "hesitant", "rushed", "monotone")


def answers_table(source: str = "synthetic") -> pd.DataFrame:
    """One row per answer: question text, transcript, labels, split. Synthetic adds STAR + completeness."""
    a = pd.read_parquet(LABELS / source / "answers.parquet")
    if source == "synthetic":
        extra = pd.read_csv(SYNTHETIC / "interview" / "answers.csv",
                            usecols=["answer_id", "completeness", "star_situation", "star_task", "star_action",
                                     "star_result", "question_category"])
        a = a.merge(extra, on="answer_id", how="left")
    sp = split_of(source)
    return a.assign(split=a.session_id.map(sp))


class AnswerDataset(Dataset):
    """[question] <sep> [answer] token ids (answer tokenizer) -> relevance, quality, STAR, completeness, follow-up."""

    def __init__(self, split: str, tokenizer, source: str = "synthetic", max_len: int = 256,
                 table: pd.DataFrame | None = None):
        t = table if table is not None else answers_table(source)
        self.t = t[t.split == split].reset_index(drop=True)
        self.tok, self.max_len = tokenizer, max_len
        self.sep = len(tokenizer)          # extra id appended after the SentencePiece vocab
        self.cls = len(tokenizer) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.tok) + 2

    def __len__(self) -> int:
        return len(self.t)

    def encode(self, question: str, answer: str) -> list[int]:
        q = self.tok.encode(str(question))[:48]
        a = self.tok.encode(str(answer))
        return ([self.cls] + q + [self.sep] + a)[: self.max_len]

    def __getitem__(self, i: int):
        r = self.t.iloc[i]
        star = [getattr(r, c, np.nan) for c in ("star_situation", "star_task", "star_action", "star_result")]
        star = np.array([np.nan if v is None else float(v) for v in star], np.float32)
        comp = getattr(r, "completeness", np.nan)
        return {"ids": torch.tensor(self.encode(r.question_text, r.transcript), dtype=torch.long),
                "relevance": torch.tensor(float(r.relevance), dtype=torch.float32),
                "quality": torch.tensor(int(r.quality_1to5) - 1),
                "star": torch.from_numpy(np.nan_to_num(star, nan=0.0)),
                "star_mask": torch.from_numpy((~np.isnan(star)).astype(np.float32)),
                "completeness": torch.tensor(0.0 if comp is None or np.isnan(comp) else float(comp)),
                "completeness_mask": torch.tensor(0.0 if comp is None or np.isnan(comp) else 1.0),
                "follow_up": torch.tensor(float(r.follow_up_needed)),
                "reason": torch.tensor(FOLLOW_UP_REASONS.index(r.follow_up_reason)
                                       if r.follow_up_reason in FOLLOW_UP_REASONS else 0)}


def answer_collate(batch):
    ids, lens = pad_1d([b["ids"] for b in batch], 0)
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "ids"}
    out.update(ids=ids, lens=lens, mask=ids != 0)
    return out


# Per-answer features by modality - only features our live pipeline computes:
#   vision        Stage 4-5 window signals aggregated over the answer
#   prosody       ml/behaviour/prosody.py (+ MFCC means)
#   hesitation    pause model + text rules + disfluency CRNN
#   content       answer-quality transformer outputs (out-of-fold for training) + transcript statistics
#   conversation  conversation analyser
# Simulator-internal columns (key_points_covered, vague/off-topic sentence counts, contradiction flag, jitter,
# shimmer, confusion level) are deliberately NOT used.
FUSION_FEATURES: dict[str, list[str]] = {
    "vision": ["mean_on_screen_ratio", "mean_attention_score", "engaged_window_fraction", "disengaged_window_fraction",
               "mean_blink_rate_per_min", "total_glance_away", "mean_gaze_yaw_std_deg", "mean_head_yaw_std_deg",
               "reading_pattern_detected"],
    "prosody": ["speaking_rate_wpm", "articulation_rate_wpm", "pitch_mean_hz", "pitch_std_hz", "pitch_range_hz",
                "monotony_index", "energy_mean_db", "energy_std_db", "volume_consistency", "speech_rate_variability",
                "snr_db"] + [f"mfcc_{i}_mean" for i in range(1, 14)],
    "hesitation": ["response_latency_s", "long_pause_count", "long_pause_total_s", "pause_ratio",
                   "mean_pause_duration_s", "filler_count", "filler_rate_per_100w", "repetition_count",
                   "restart_count", "hedge_count", "prolongation_count", "block_count", "disfluency_rate_per_100w"],
    "content": ["answer_duration_s", "word_count", "keyword_overlap", "asr_confidence_mean", "question_difficulty",
                "ans_relevance", "ans_quality", "ans_completeness", "ans_star_count", "ans_follow_up_p",
                "ans_reason_vague", "ans_reason_low_confidence", "ans_reason_attention_drop",
                "ans_reason_topic_not_covered", "ans_reason_contradiction"],
    "conversation": ["interrupted_by_interviewer", "overlap_s"],
}
FUSION_TARGETS = ["communication_effectiveness", "clarity_score", "confidence_score", "engagement_score",
                  "relevance_score", "fluency_score"]


def fusion_table(source: str = "synthetic") -> pd.DataFrame:
    """Per-answer feature table + labels. Answer-model features are merged when they have been computed
    (python -m ml.fusion.answer_features); missing ones stay NaN (-> 0 after standardisation)."""
    feats = LABELS.parent / "features"
    if source != "synthetic":
        p = feats / f"answers_{source}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the Stage 8 feature assembler")
        t = pd.read_parquet(p)
    else:
        t = pd.read_csv(SYNTHETIC / "interview" / "answers.csv")
    am = feats / f"answer_model_{source}.parquet"
    if am.exists() and "ans_relevance" not in t:
        t = t.merge(pd.read_parquet(am), on="answer_id", how="left")
    for cols in FUSION_FEATURES.values():
        for c in cols:
            if c not in t:
                t[c] = np.nan
    sp = split_of(source)
    return t.assign(split=t.session_id.map(sp))


class FusionDataset(Dataset):
    """Modality feature vectors (standardised with train statistics, NaN -> 0 + mask) -> scores + follow-up."""

    def __init__(self, split: str, source: str = "synthetic", table: pd.DataFrame | None = None,
                 stats: dict | None = None):
        t = table if table is not None else fusion_table(source)
        train = t[t.split == "train"]
        self.stats = stats or {m: (train[c].astype(float).mean().to_numpy(), train[c].astype(float).std().to_numpy() + 1e-6)
                               for m, c in FUSION_FEATURES.items()}
        self.t = t[t.split == split].reset_index(drop=True)
        self.x = {}
        for m, cols in FUSION_FEATURES.items():
            v = self.t[cols].astype(float).to_numpy()
            mu, sd = self.stats[m]
            self.x[m] = np.nan_to_num((v - mu) / sd, nan=0.0).astype(np.float32)
        self.y = self.t[FUSION_TARGETS].astype(float).to_numpy(np.float32) / 100.0
        self.fu = self.t.follow_up_needed.astype(float).to_numpy(np.float32)
        self.reason = self.t.follow_up_reason.map(lambda r: FOLLOW_UP_REASONS.index(r) if r in FOLLOW_UP_REASONS else 0
                                                  ).to_numpy()

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int):
        out = {m: torch.from_numpy(self.x[m][i]) for m in FUSION_FEATURES}
        out.update(scores=torch.from_numpy(self.y[i]), follow_up=torch.tensor(self.fu[i]),
                   reason=torch.tensor(int(self.reason[i])))
        return out
