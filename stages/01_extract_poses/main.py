"""Stage 1 — extract DW Pose skeletons.

Reads `input.pose_source` (mp4 or directory of frames), samples it down to
`generation.fps_generate`, runs DW Pose on each sampled frame, and writes:

    outputs/<run_id>/poses/0001.png ... NNNN.png   (skeletons on black bg)
    outputs/<run_id>/poses/skeleton.json           (per-frame metadata)
    outputs/<run_id>/poses/face_bboxes.json        (per-frame face bbox in
                                                    output (W, H) space; fed
                                                    to stage 3's detailer pass)

Face landmarks are computed but NOT drawn on the skeleton — at full-body
framing the 68 face keypoints cluster sub-pixel, which forces ControlNet
to render the face in a region too small for SD1.5's 1/8 latent space.
The bbox is preserved separately so the detailer can fix faces post-hoc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image

from video_pipeline_common import REPO_ROOT, Paths, load_config, make_argparser, num_frames_for, setup_logging


def iter_source_frames(src: Path):
    """Yield (idx, np.uint8 HxWx3 RGB) over the input source at native rate.

    Supports a single mp4/mov file or a directory of image frames.
    """
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for i, p in enumerate(files):
            yield i, np.array(Image.open(p).convert("RGB"))
        return
    with iio.imopen(src, "r", plugin="pyav") as f:
        for i, frame in enumerate(f.iter()):
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            yield i, frame


def source_native_fps(src: Path) -> float:
    if src.is_dir():
        return 30.0
    meta = iio.immeta(src, plugin="pyav")
    return float(meta.get("fps", 30.0))


def _letterbox_fit(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[float, int, int]:
    """Aspect-preserving fit. Returns (scale, x_offset, y_offset) where
    src→out is `(x*scale + x_off, y*scale + y_off)`. The unused axis becomes
    a black letterbox band on the output canvas.
    """
    scale = min(out_w / src_w, out_h / src_h)
    x_off = (out_w - int(round(src_w * scale))) // 2
    y_off = (out_h - int(round(src_h * scale))) // 2
    return scale, x_off, y_off


def _face_bbox_in_output_space(
    keypoints: np.ndarray, scores: np.ndarray, src_w: int, src_h: int, out_w: int, out_h: int
) -> list[int] | None:
    """Tight face bbox per detected subject from rtmlib's 68 face keypoints,
    mapped from source pixel space to the SAME letterboxed (out_w, out_h)
    canvas the skeleton is drawn on (uniform scale + center offset). Returns
    the largest-area face's [x1, y1, x2, y2], or None if nothing met the
    confidence threshold. Box is enlarged 1.5× (max dim, square) so the
    detailer has skin/hair context around the face for a believable inpaint.
    """
    if keypoints.ndim < 3 or keypoints.shape[1] < 92:
        return None
    face_kpts = keypoints[:, 24:92, :]   # (N, 68, 2)
    face_scores = scores[:, 24:92]        # (N, 68)
    valid = face_scores > 0.3
    scale, x_off, y_off = _letterbox_fit(src_w, src_h, out_w, out_h)

    best_area = 0
    best: list[int] | None = None
    for i in range(face_kpts.shape[0]):
        if not valid[i].any():
            continue
        pts = face_kpts[i][valid[i]]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        area = (x2 - x1) * (y2 - y1)
        if area <= best_area:
            continue
        best_area = area
        cx = (x1 + x2) / 2 * scale + x_off
        cy = (y1 + y2) / 2 * scale + y_off
        w, h = (x2 - x1) * scale, (y2 - y1) * scale
        side = max(w, h) * 1.5  # square crop with context for inpaint
        x1o = max(0, int(cx - side / 2))
        y1o = max(0, int(cy - side / 2))
        x2o = min(out_w, int(cx + side / 2))
        y2o = min(out_h, int(cy + side / 2))
        if x2o > x1o and y2o > y1o:
            best = [x1o, y1o, x2o, y2o]
    return best


def main():
    args = make_argparser("01_extract_poses").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)

    paths.poses.mkdir(parents=True, exist_ok=True)

    src = Path(cfg["input"]["pose_source"])
    if not src.is_absolute():
        src = REPO_ROOT / src
    if not src.exists():
        log.error(f"pose_source does not exist: {src}")
        sys.exit(1)

    n_target = num_frames_for(cfg)
    target_fps = cfg["generation"]["fps_generate"]
    native_fps = source_native_fps(src)
    stride = max(1, round(native_fps / target_fps))
    log.info(f"native_fps={native_fps:.2f} target_fps={target_fps} stride={stride} target_frames={n_target}")

    from rtmlib import Wholebody, draw_skeleton

    log.info("loading DW Pose detector (rtmlib.Wholebody, onnxruntime, cuda)…")
    detector = Wholebody(to_openpose=True, mode="balanced", backend="onnxruntime", device="cuda")

    W = cfg["generation"]["width"]
    H = cfg["generation"]["height"]

    saved: list[dict] = []
    bboxes: list[dict] = []
    out_idx = 0
    for src_idx, frame in iter_source_frames(src):
        if src_idx % stride != 0:
            continue
        if out_idx >= n_target:
            break

        # rtmlib expects BGR; iter_source_frames yields RGB.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        src_h, src_w = bgr.shape[:2]
        keypoints, scores = detector(bgr)

        # Compute face bbox from face keypoints BEFORE we zero them — stage 3's
        # detailer needs to know where to inpaint. Use the largest detected
        # subject; bbox is in source-image pixel space, then mapped to (W, H)
        # output space. None when no high-confidence face is found.
        bbox = _face_bbox_in_output_space(keypoints, scores, src_w, src_h, W, H)

        # Drop face keypoints. With full-body framing the face occupies ~50 px
        # in a 512-tall frame, so the 68 facial landmarks cluster sub-pixel.
        # ControlNet then forces SD to draw eyes/nose/mouth at exact pixel
        # locations it can't render coherently → smeared, distorted faces.
        # Body+hands+feet are kept; the face gets free reign for IP-Adapter
        # FaceID identity transfer to actually drive what shows up there.
        # rtmlib OpenPose layout (134 kpts): 0-17 body, 18-23 feet, 24-91 face,
        # 92-133 hands. Zeroing scores below kpt_thr drops them at draw time.
        if scores.shape[-1] >= 92:
            scores[..., 24:92] = 0.0
        canvas = np.zeros_like(bgr)
        canvas = draw_skeleton(canvas, keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)
        # Aspect-preserving fit onto the (W, H) target. Force-resizing a 16:9
        # source skeleton to a 2:3 portrait squishes shoulders horizontally —
        # the model then reads "narrow shoulders, body in profile" and renders
        # a side view even when the skeleton is clearly frontal. Letterbox
        # instead so geometry survives.
        scale, x_off, y_off = _letterbox_fit(src_w, src_h, W, H)
        scaled_w, scaled_h = int(round(src_w * scale)), int(round(src_h * scale))
        skeleton_src = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize(
            (scaled_w, scaled_h), Image.LANCZOS
        )
        skeleton = Image.new("RGB", (W, H), "black")
        skeleton.paste(skeleton_src, (x_off, y_off))
        out_path = paths.poses / f"{out_idx + 1:04d}.png"
        skeleton.save(out_path)

        saved.append({"frame": out_idx + 1, "src_idx": src_idx, "path": str(out_path.name)})
        bboxes.append({"frame": out_idx + 1, "bbox": bbox})
        out_idx += 1
        if out_idx % 8 == 0:
            log.info(f"  …{out_idx}/{n_target} pose frames")

    if out_idx < n_target:
        log.warning(f"source ran out at {out_idx} frames (wanted {n_target}). Will pad by repeating last.")
        if out_idx == 0:
            log.error("no frames produced — aborting")
            sys.exit(1)
        last = paths.poses / f"{out_idx:04d}.png"
        last_bbox = bboxes[-1]["bbox"] if bboxes else None
        for i in range(out_idx, n_target):
            (paths.poses / f"{i + 1:04d}.png").write_bytes(last.read_bytes())
            saved.append({"frame": i + 1, "src_idx": -1, "path": last.name, "padded": True})
            bboxes.append({"frame": i + 1, "bbox": last_bbox, "padded": True})

    (paths.poses / "skeleton.json").write_text(json.dumps(saved, indent=2))
    (paths.poses / "face_bboxes.json").write_text(json.dumps(bboxes, indent=2))
    n_with_face = sum(1 for b in bboxes if b["bbox"] is not None)
    log.info(f"wrote {n_target} pose frames to {paths.poses} ({n_with_face} with face bbox)")


if __name__ == "__main__":
    main()
