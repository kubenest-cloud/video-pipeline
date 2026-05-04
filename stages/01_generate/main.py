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


def _prepare_animate_conditioning(
    pipe, driving: list[Image.Image], log
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Produce (pose_video, face_video) from the raw driving frames.

    Wan-Animate's diffusers integration expects pre-rendered pose skeletons +
    face crops, not raw RGB. The official Wan repo ships a preprocess utility
    that runs DWPose + face detection over the driving clip. Diffusers usually
    wraps that as `WanAnimateProcessor` (or similar); try the standard import
    locations and fall back to raw frames if nothing's there.

    The fallback is lossy — the pose encoder was trained on rendered skeleton
    PNGs, not natural images, so it'll mis-read what the figure is doing.
    Output will look photoreal but motion will drift from the driving clip
    until we wire real preprocessing. Logs make the situation explicit.
    """
    processor = None
    for path in (
        "diffusers.WanAnimateProcessor",
        "diffusers.pipelines.wan.WanAnimateProcessor",
        "diffusers.pipelines.wan.processor_wan_animate.WanAnimateProcessor",
        "diffusers.pipelines.wan.pipeline_wan_animate.WanAnimateProcessor",
    ):
        try:
            mod_name, _, cls_name = path.rpartition(".")
            mod = __import__(mod_name, fromlist=[cls_name])
            processor = getattr(mod, cls_name)()
            log.info(f"using diffusers preprocessor: {path}")
            break
        except (ImportError, AttributeError):
            continue

    if processor is not None:
        # API not yet stable — try the most likely method names. If none match,
        # log the processor's public methods so we can wire the right one.
        for method_name in ("preprocess", "prepare", "process", "__call__"):
            fn = getattr(processor, method_name, None)
            if fn is None:
                continue
            try:
                result = fn(driving)
            except Exception as e:
                log.warning(f"processor.{method_name}(driving) raised {type(e).__name__}: {e}")
                continue
            if isinstance(result, dict) and "pose_video" in result and "face_video" in result:
                return result["pose_video"], result["face_video"]
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
            log.warning(f"processor.{method_name} returned {type(result).__name__}; expected dict or 2-tuple")
        public = [n for n in dir(processor) if not n.startswith("_")]
        log.error(f"could not call WanAnimateProcessor — public attrs: {public}")
        sys.exit(1)

    log.warning(
        "No WanAnimateProcessor found in this diffusers build. Falling back to "
        "raw RGB frames for pose_video AND face_video — the pose encoder will "
        "mis-read this (it expects rendered skeletons), so motion adherence "
        "will be poor. Wire DWPose + face-crop preprocessing for real output."
    )
    return driving, driving


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

    pose_video, face_video = _prepare_animate_conditioning(pipe, driving, log)

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
    out = pipe(**call_kwargs)

    frames: list[Image.Image] = out.frames[0]
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
