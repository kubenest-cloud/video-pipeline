"""Stage 1 — Wan 2.2 Animate, pose-driven photoreal character video.

Takes one reference image (the identity) plus a driving video (the motion)
and renders the reference person performing the driving video's motion.
Pose extraction is internal to the model; no separate skeleton step.

Outputs (consumed by stage 2):
    outputs/<run_id>/raw_frames/0001.png ... NNNN.png
    outputs/<run_id>/raw_frames/_contactsheet.jpg
    outputs/<run_id>/raw_frames/_preview.mp4
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from video_pipeline_common import REPO_ROOT, Paths, load_config, make_argparser, num_frames_for, setup_logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Wan 2.2 was trained in bf16. fp16 underflows the rotary embedding scales
# during inference and produces black/blocky outputs — this is documented in
# the Wan repo. Stay in bf16 on GPU; CPU fallback gets fp32.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _load_first_frame_size(video_path: Path) -> tuple[int, int]:
    """Return (W, H) of the first frame — used for sanity-logging the driving
    video's resolution alongside the output target."""
    with iio.imopen(str(video_path), "r", plugin="pyav") as f:
        for frame in f.iter():
            return int(frame.shape[1]), int(frame.shape[0])
    raise RuntimeError(f"no frames in {video_path}")


def _load_driving_clip(video_path: Path, n_frames: int, target_fps: int, log) -> list[Image.Image]:
    """Sample the driving video at target_fps to get n_frames PIL frames.
    Wan ingests the full driving video as a list of PIL images and runs its
    own pose extractor; we just have to pick the right cadence so output
    duration matches generation.duration_sec.
    """
    meta = iio.immeta(str(video_path), plugin="pyav")
    src_fps = float(meta.get("fps", 30.0))
    stride = max(1, round(src_fps / target_fps))
    log.info(f"driving video: src_fps={src_fps:.2f} target_fps={target_fps} stride={stride}")

    out: list[Image.Image] = []
    with iio.imopen(str(video_path), "r", plugin="pyav") as f:
        for i, frame in enumerate(f.iter()):
            if i % stride != 0:
                continue
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            out.append(Image.fromarray(frame))
            if len(out) >= n_frames:
                break
    if len(out) < n_frames:
        log.warning(f"driving video ran out at {len(out)}/{n_frames} frames; padding with last")
        if not out:
            raise RuntimeError("driving video yielded zero frames")
        out.extend([out[-1]] * (n_frames - len(out)))
    return out


def _center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop to the target aspect, then resize. Reference image and
    driving frames both pass through this so identity and motion arrive in
    the model's coordinate system without letterbox bands.
    """
    src_w, src_h = img.size
    src_ratio, out_ratio = src_w / src_h, target_w / target_h
    if src_ratio > out_ratio:
        crop_w = int(round(src_h * out_ratio))
        crop_h = src_h
    else:
        crop_w = src_w
        crop_h = int(round(src_w / out_ratio))
    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2
    return img.crop((x, y, x + crop_w, y + crop_h)).resize((target_w, target_h), Image.LANCZOS)


def _face_bbox_from_keypoints(
    keypoints: "np.ndarray", scores: "np.ndarray", src_w: int, src_h: int, pad: float = 1.6
) -> tuple[int, int, int, int] | None:
    """Tight square face bbox from rtmlib's 68 face keypoints (indices 24:92
    in the 134-kpt OpenPose-converted layout). Returns (x1, y1, x2, y2) in
    source pixel space, or None if no high-confidence face was found.
    """
    if keypoints.ndim < 3 or keypoints.shape[1] < 92:
        return None
    face_kpts = keypoints[:, 24:92, :]
    face_scores = scores[:, 24:92]
    valid = face_scores > 0.3

    best_area = 0
    best: tuple[int, int, int, int] | None = None
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
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * pad
        best = (
            max(0, int(cx - side / 2)),
            max(0, int(cy - side / 2)),
            min(src_w, int(cx + side / 2)),
            min(src_h, int(cy + side / 2)),
        )
    return best


def _prepare_animate_conditioning(
    driving: list[Image.Image], W: int, H: int, log
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Run DWPose on each driving frame; return (pose_video, face_video).

    pose_video: rendered skeleton on black background, sized to (W, H) — what
                Wan-Animate's pose encoder was trained on. Without this, it
                ignores motion entirely and animates freely from the reference.
    face_video: face crop per frame, padded ~1.6× and resized to 512×512 (Wan
                auto-reshapes to square anyway, and the model's face encoder
                expects a square crop centered on the face).

    First call downloads ~50 MB of DW Pose weights into ~/.cache/rtmlib —
    mount that cache (or HF_HOME) to persist across docker runs.
    """
    import cv2
    from rtmlib import Wholebody, draw_skeleton

    log.info("loading DW Pose detector for Wan-Animate preprocessing…")
    detector = Wholebody(to_openpose=True, mode="balanced", backend="onnxruntime", device="cuda")

    # Pre-derive a placeholder face crop (small center region, neutral gray) so
    # frames with no detection don't break the list — Wan needs face_video and
    # pose_video to have the same length as the driving clip.
    placeholder_face = Image.new("RGB", (512, 512), (128, 128, 128))

    pose_video: list[Image.Image] = []
    face_video: list[Image.Image] = []
    n_no_face = 0
    for idx, frame in enumerate(driving):
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        src_h, src_w = bgr.shape[:2]
        keypoints, scores = detector(bgr)

        # Pose: render full skeleton (body + face + hands) on a black canvas
        # at source size, then resize to the output (W, H). Wan expects RGB.
        canvas = np.zeros_like(bgr)
        canvas = draw_skeleton(canvas, keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)
        pose_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize(
            (W, H), Image.LANCZOS
        )
        pose_video.append(pose_pil)

        # Face: crop the source frame to the face bbox, square-pad, resize 512.
        bbox = _face_bbox_from_keypoints(keypoints, scores, src_w, src_h, pad=1.6)
        if bbox is None:
            n_no_face += 1
            face_video.append(placeholder_face if not face_video else face_video[-1])
        else:
            face_video.append(frame.crop(bbox).resize((512, 512), Image.LANCZOS))

        if (idx + 1) % 16 == 0:
            log.info(f"  …preprocessed {idx + 1}/{len(driving)} frames")

    if n_no_face:
        log.warning(f"no face detected in {n_no_face}/{len(driving)} frames; reused previous face crop")
    log.info(f"preprocessing done: {len(pose_video)} pose frames @ {W}x{H}, {len(face_video)} face crops @ 512x512")
    return pose_video, face_video


def _coerce_to_pil_list(raw, log) -> list[Image.Image]:
    """Normalize whatever the pipeline returned into list[PIL.Image].

    Wan-Animate's diffusers integration is in flux: `out.frames[0]` is
    sometimes list[PIL] (when output_type='pil' is honored), sometimes a
    numpy array of shape (T, H, W, 3) or (T, 3, H, W), sometimes a torch
    tensor in [0, 1]. Don't assume — coerce.
    """
    # Already a list of PIL.
    if isinstance(raw, list) and raw and isinstance(raw[0], Image.Image):
        return raw

    # torch tensor → numpy
    if hasattr(raw, "detach") and hasattr(raw, "cpu"):
        raw = raw.detach().cpu().float().numpy()

    arr = np.asarray(raw)
    log.info(f"coercing pipeline output: shape={arr.shape}, dtype={arr.dtype}")

    # (T, 3, H, W) → (T, H, W, 3)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise RuntimeError(
            f"unexpected pipeline output shape {arr.shape}; "
            f"expected (T, H, W, 3). Raw is dumped to _raw_output.npz."
        )

    # Normalize value range. uint8 stays as-is; floats in [0,1] scale to 255;
    # floats in [-1,1] (rare) get re-centered first.
    if arr.dtype != np.uint8:
        if arr.min() < -0.05:  # pretty clearly [-1, 1]
            arr = (arr + 1.0) * 0.5
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.5 else \
              np.clip(arr,         0, 255).astype(np.uint8)

    return [Image.fromarray(frame) for frame in arr]


def write_contact_sheet(frames: list[Image.Image], out_path: Path, every: int = 8) -> None:
    sample = frames[::every] or frames[:1]
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    w, h = sample[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for i, im in enumerate(sample):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * w, r * h))
    sheet.save(out_path, quality=88)


def build_pipeline(cfg: dict, log):
    """Load WanAnimatePipeline. Fails loud if it isn't in the installed
    diffusers build — there's no I2V fallback because I2V doesn't take a
    driving video, and without pose conditioning the output is not what
    this pipeline is for. Fix is documented in the error message.
    """
    g = cfg["generation"]
    model_id = g["model"]

    try:
        from diffusers import WanAnimatePipeline
    except ImportError:
        log.error(
            "WanAnimatePipeline is not in the installed diffusers build. The "
            "released wheels (0.34, 0.35) only ship WanImageToVideoPipeline, "
            "which has no pose conditioning and is useless for this task.\n\n"
            "Fix: install diffusers from git main, then re-sync:\n"
            "    uv add --project stages/01_generate "
            "'diffusers @ git+https://github.com/huggingface/diffusers.git@main'\n"
            "    uv sync --project stages/01_generate --reinstall-package diffusers\n"
        )
        sys.exit(1)

    log.info(f"loading WanAnimatePipeline: {model_id}")
    pipe = WanAnimatePipeline.from_pretrained(model_id, torch_dtype=DTYPE)
    # Wan 14B is ~28 GB in bf16; even on a 48 GB A6000 we want headroom for
    # T5 + VAE + activations across 80 frames. cpu-offload streams transformer
    # blocks GPU↔CPU per forward and keeps peak well under 24 GB.
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        log.info("vae tiling enabled (decoder fits 720p without OOM)")
    log.info("pipeline ready")
    return pipe


def main():
    args = make_argparser("01_generate").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)

    paths.raw_frames.mkdir(parents=True, exist_ok=True)

    g = cfg["generation"]
    n = num_frames_for(cfg)
    W, H = g["width"], g["height"]

    ref_path = Path(cfg["input"]["face_ref"])
    if not ref_path.is_absolute():
        ref_path = REPO_ROOT / ref_path
    if not ref_path.exists():
        log.error(f"face_ref does not exist: {ref_path}")
        sys.exit(1)
    reference = _center_crop_to_aspect(Image.open(ref_path).convert("RGB"), W, H)
    log.info(f"reference image: {ref_path.name} → {W}x{H}")

    src_path = Path(cfg["input"]["pose_source"])
    if not src_path.is_absolute():
        src_path = REPO_ROOT / src_path
    if not src_path.exists():
        log.error(f"pose_source does not exist: {src_path}")
        sys.exit(1)
    src_w, src_h = _load_first_frame_size(src_path)
    log.info(f"driving video: {src_path.name} ({src_w}x{src_h})")

    pipe = build_pipeline(cfg, log)

    driving = [_center_crop_to_aspect(f, W, H) for f in _load_driving_clip(src_path, n, g["fps_generate"], log)]
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    # Wan-Animate's diffusers signature splits the driving signal into two
    # required conditioning videos — `pose_video` (rendered skeleton frames)
    # and `face_video` (face crops) — plus the `image` reference. Wan's own
    # inference repo ships a preprocess script that produces both from a raw
    # RGB driving clip; diffusers exposes that as WanAnimateProcessor (or a
    # `prepare_inputs` helper) on recent commits. Try the processor first;
    # if it isn't there, fall back to feeding the same raw RGB driving frames
    # to both slots (lossy: the body pose encoder expects rendered skeletons,
    # not raw RGB, so quality will suffer until we wire real preprocessing).
    sig = inspect.signature(pipe.__call__)
    params = set(sig.parameters.keys())
    log.info(f"WanAnimatePipeline.__call__ params: {sorted(params)}")

    pose_video, face_video = _prepare_animate_conditioning(driving, W, H, log)

    # Dump preprocessing outputs for inspection — if pose/face look wrong here,
    # the 13-min generation will produce garbage, so check before paying.
    debug_dir = paths.raw_frames.parent / "preprocess"
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_contact_sheet(pose_video, debug_dir / "_pose_contactsheet.jpg", every=8)
    write_contact_sheet(face_video, debug_dir / "_face_contactsheet.jpg", every=8)
    log.info(f"wrote preprocess contact sheets → {debug_dir}/_{{pose,face}}_contactsheet.jpg")

    log.info(f"running Wan-Animate: {len(pose_video)} frames @ {W}x{H}, mode=animation, steps={g['steps']}…")
    call_kwargs = {
        "image": reference,
        "pose_video": pose_video,
        "face_video": face_video,
        "prompt": g["prompt"],
        "negative_prompt": g["negative_prompt"],
        "height": H,
        "width": W,
        "num_inference_steps": g["steps"],
        "guidance_scale": g["guidance_scale"],
        "generator": generator,
    }
    if "mode" in params:
        # Accepted values are "animate" (render reference person doing the
        # driving motion) and "replace" (swap the person in the driving video
        # for the reference; needs mask_video + background_video).
        call_kwargs["mode"] = "animate"
    if "output_type" in params:
        call_kwargs["output_type"] = "pil"
    out = pipe(**call_kwargs)

    raw_out = out.frames[0]

    # Dump the raw output BEFORE any post-processing — Wan-Animate runs ~13
    # min per pass on this clip, and a save-loop bug shouldn't be able to
    # waste that compute. Recover frames from this with `np.load(...)['arr_0']`
    # and re-run any of the post steps below independently.
    raw_dump = paths.raw_frames / "_raw_output.npz"
    raw_arr = np.asarray([np.asarray(f) for f in raw_out]) if not isinstance(raw_out, np.ndarray) else raw_out
    np.savez_compressed(raw_dump, arr_0=raw_arr)
    log.info(f"dumped raw output ({raw_arr.shape}, dtype={raw_arr.dtype}) → {raw_dump}")

    frames = _coerce_to_pil_list(raw_out, log)
    log.info(f"writing {len(frames)} frames to {paths.raw_frames}")
    for i, fr in enumerate(frames, start=1):
        fr.save(paths.raw_frames / f"{i:04d}.png")

    write_contact_sheet(frames, paths.raw_frames / "_contactsheet.jpg", every=8)
    log.info(f"wrote contact sheet → {paths.raw_frames / '_contactsheet.jpg'}")

    preview_path = paths.raw_frames / "_preview.mp4"
    iio.imwrite(
        preview_path,
        np.stack([np.array(f) for f in frames]),
        plugin="pyav",
        fps=g["fps_generate"],
        codec="libx264",
    )
    log.info(f"wrote raw-fps preview → {preview_path}")
    log.info("INSPECT contact sheet + preview before running stage 2 (interpolate).")


if __name__ == "__main__":
    main()
