"""Stage 1 — extract DW Pose skeletons.

Reads `input.pose_source` (mp4 or directory of frames), samples it down to
`generation.fps_generate`, runs DW Pose on each sampled frame, and writes:

    outputs/<run_id>/poses/0001.png ... NNNN.png   (skeletons on black bg)
    outputs/<run_id>/poses/skeleton.json           (keypoints, for reference)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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

    from controlnet_aux import DWposeDetector

    log.info("loading DW Pose detector (controlnet_aux.DWposeDetector)…")
    detector = DWposeDetector()

    W = cfg["generation"]["width"]
    H = cfg["generation"]["height"]

    saved: list[dict] = []
    out_idx = 0
    for src_idx, frame in iter_source_frames(src):
        if src_idx % stride != 0:
            continue
        if out_idx >= n_target:
            break

        skeleton = detector(frame, output_type="pil", include_face=True, include_hand=True)
        skeleton = skeleton.resize((W, H), Image.LANCZOS)
        out_path = paths.poses / f"{out_idx + 1:04d}.png"
        skeleton.save(out_path)

        saved.append({"frame": out_idx + 1, "src_idx": src_idx, "path": str(out_path.name)})
        out_idx += 1
        if out_idx % 8 == 0:
            log.info(f"  …{out_idx}/{n_target} pose frames")

    if out_idx < n_target:
        log.warning(f"source ran out at {out_idx} frames (wanted {n_target}). Will pad by repeating last.")
        if out_idx == 0:
            log.error("no frames produced — aborting")
            sys.exit(1)
        last = paths.poses / f"{out_idx:04d}.png"
        for i in range(out_idx, n_target):
            (paths.poses / f"{i + 1:04d}.png").write_bytes(last.read_bytes())
            saved.append({"frame": i + 1, "src_idx": -1, "path": last.name, "padded": True})

    (paths.poses / "skeleton.json").write_text(json.dumps(saved, indent=2))
    log.info(f"wrote {n_target} pose frames to {paths.poses}")


if __name__ == "__main__":
    main()
