"""Evaluate the face detector: AP@0.5 / recall on WIDER val (our test split), landmark NME, CPU FPS.

Two test views of WIDER val:
  full     whole images letterboxed to 128 px; faces >= `large_face` of the input are scored,
           smaller faces are 'ignore' (too small for a 128 px webcam model; not counted as misses or FPs)
  webcam   deterministic square crops around each face (1.5-4x face size), like a webcam frame
Landmark NME (normalised by inter-ocular distance) on the val split, which has RetinaFace landmarks.

    python -m ml.vision.face_detector.eval
Writes docs/metrics/face_detector.json, docs/metrics/face_detector_pr.png, docs/model_cards/face_detector.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ml.common.config import CHECKPOINTS, METRICS, load_config, seed_everything
from ml.common.datasets import FaceDataset
from ml.common.train_utils import benchmark, count_params, model_card, plot_history, write_metrics
from ml.vision.face_detector.model import FaceDetector, box_iou, postprocess

HERE = Path(__file__).resolve().parent


def load_model(ckpt: Path | None = None, dev="cpu") -> FaceDetector:
    st = torch.load(ckpt or CHECKPOINTS / "face_detector" / "best.pt", map_location="cpu", weights_only=False)
    m = FaceDetector(st["cfg"]["model"]["size"], st["cfg"]["model"]["width"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval()


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    if n_gt == 0 or len(scores) == 0:
        return 0.0, np.zeros(1), np.zeros(1)
    o = np.argsort(-scores)
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), rec, prec


@torch.no_grad()
def evaluate(model, ds, dev, large: float, score_thr: float, iou_thr: float = 0.5, batch: int = 64) -> dict:
    all_scores, all_tp, n_gt, hits_at_thr = [], [], 0, 0
    nme, n_kp = [], 0
    for s in range(0, len(ds), batch):
        items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
        x = torch.stack([it[0] for it in items]).to(dev)
        outs = postprocess(*model(x), model.anchors, score_thr=0.05, iou_thr=0.3, max_det=50)
        for (img, gt, kp, km), (b, sc, k) in zip(items, outs):
            gt, b, sc, k = gt.to(dev), b.to(dev), sc.to(dev), k.to(dev)
            size = (gt[:, 2:] - gt[:, :2]).max(1).values if len(gt) else torch.zeros(0, device=dev)
            big = size >= large
            n_gt += int(big.sum())
            matched = torch.zeros(len(gt), dtype=torch.bool, device=dev)
            iou = box_iou(b, gt) if len(gt) and len(b) else torch.zeros(len(b), len(gt), device=dev)
            for j in range(len(b)):          # detections are sorted by score
                if len(gt):
                    v, g = (iou[j] * (~matched)).max(0)
                    if v >= iou_thr:
                        matched[g] = True
                        if big[g]:
                            all_scores.append(float(sc[j]))
                            all_tp.append(1)
                            hits_at_thr += int(sc[j] >= score_thr)
                            if km[g] > 0:
                                kg = kp[g].to(dev)
                                iod = torch.linalg.norm(kg[0] - kg[1]).clamp(min=1e-6)
                                nme.append(float(torch.linalg.norm(k[j] - kg, dim=-1).mean() / iod))
                        continue                 # matches to small faces are ignored
                    if iou[j].max() >= iou_thr:  # duplicate on an already matched face
                        all_scores.append(float(sc[j]))
                        all_tp.append(0)
                        continue
                all_scores.append(float(sc[j]))
                all_tp.append(0)
            n_kp += int((km > 0).sum())
    sc, tp = np.array(all_scores), np.array(all_tp)
    ap, rec, prec = average_precision(sc, tp, n_gt)
    keep = sc >= score_thr
    return {"images": len(ds), "faces_scored": n_gt, "AP@0.5": round(ap, 4),
            f"recall@score{score_thr}": round(hits_at_thr / max(n_gt, 1), 4),
            f"precision@score{score_thr}": round(float(tp[keep].mean()) if keep.any() else 0.0, 4),
            "landmark_NME": round(float(np.mean(nme)), 4) if nme else None, "landmark_faces": len(nme),
            "_pr": (rec.tolist(), prec.tolist())}


def plot_pr(curves: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (r, p, ap) in curves.items():
        ax.plot(r, p, label=f"{name} (AP {ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Face detector, IoU 0.5")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dev=dev)
    ec = cfg["eval"]
    full = FaceDataset("test", cfg["model"]["size"], augment=False, min_face=4, limit=args.limit)
    crops = FaceDataset("test", cfg["model"]["size"], augment=True, min_face=20, limit=args.limit)
    crops.aug.rng = np.random.default_rng(123)
    crops.aug.brightness = crops.aug.contrast = 0.0
    crops.aug.blur_p = 0.0
    crops.aug.flip_p = 0.0
    crops_fixed = [crops[i] for i in range(len(crops))]      # freeze the random crops once
    lm = FaceDataset("val", cfg["model"]["size"], augment=False, min_face=20, limit=args.limit)

    res = {"full_image": evaluate(model, full, dev, ec["large_face"], ec["score_thr"]),
           "webcam_crops": evaluate(model, crops_fixed, dev, ec["large_face"], ec["score_thr"]),
           "landmarks_val": evaluate(model, lm, dev, ec["large_face"], ec["score_thr"])}
    curves = {k: (*v.pop("_pr"), v["AP@0.5"]) for k, v in res.items()}
    plot_pr({k: curves[k] for k in ("full_image", "webcam_crops")}, METRICS / "face_detector_pr.png")

    cpu = load_model().cpu()
    torch.set_num_threads(4)
    x = torch.zeros(1, 3, cfg["model"]["size"], cfg["model"]["size"])
    with torch.no_grad():
        speed = benchmark(lambda: cpu(x))
    res["cpu_latency_torch_4threads"] = speed
    res["params"] = count_params(cpu)
    write_metrics("face_detector", res)
    plot_history("face_detector", ["train_loss", "val_loss", "val_cls", "val_box", "val_kp"])
    fw, wc = res["full_image"], res["webcam_crops"]
    model_card("face_detector", "Face + landmark detector (BlazeFace-style)", {
        "Task": "Detect faces and 5 landmarks (eyes, nose, mouth corners) in a webcam frame; eye landmarks feed the "
                "eye-crop extractor for the gaze/iris CNN.",
        "Architecture": {"type": "single-shot, BlazeBlocks (depthwise 5x5 + pointwise, residual)",
                         "input": f"{cfg['model']['size']}x{cfg['model']['size']} RGB, letterboxed",
                         "anchors": "896 (16x16 x 2 + 8x8 x 6), square", "parameters": f"{res['params']:,}",
                         "heads": "face logit, box (cx, cy, w, h), 5 key points"},
        "Training data": "WIDER FACE train (90%) with RetinaFace 5-point landmarks; random face-centred square "
                         "crops (1.4-6x face size), photometric jitter, horizontal flip with landmark swap.",
        "Training": {"epochs": cfg["train"]["epochs"], "batch": cfg["train"]["batch_size"],
                     "optimiser": f"AdamW lr {cfg['train']['lr']}, cosine, warm-up {cfg['train']['warmup_steps']}",
                     "loss": "focal (alpha .25, gamma 2) + smooth-L1 box + 0.5 x smooth-L1 landmarks",
                     "curves": "docs/metrics/face_detector_curves.png", "logs": "runs/face_detector (TensorBoard)"},
        "Results (WIDER val = our test split)": {
            "webcam-style crops AP@0.5": wc["AP@0.5"], f"webcam-style crops recall@{ec['score_thr']}":
                wc[f"recall@score{ec['score_thr']}"],
            "full images AP@0.5 (faces >= 8% of input)": fw["AP@0.5"],
            "landmark NME (val, inter-ocular)": res["landmarks_val"]["landmark_NME"],
            "CPU latency (torch, 4 threads)": f"{speed['median_ms']} ms ({speed['fps']} fps)",
            "PR curve": "docs/metrics/face_detector_pr.png"},
        "Limitations": ["Faces smaller than ~10 px at 128 px input are not detected (not needed for webcams).",
                        "Not yet fine-tuned on our own frames: the 1,500-frame key-point set from our recordings "
                        "is the next step once sessions exist (FaceDataset accepts the same format).",
                        "WIDER is crowd/event photography; webcam lighting (backlit, dim) is covered only by "
                        "photometric augmentation until our recordings are added."],
    })
    print({k: v for k, v in res.items() if k != "cpu_latency_torch_4threads"}, speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
