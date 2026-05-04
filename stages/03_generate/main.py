"""Stage 3 — AnimateDiff generation with ControlNet (poses) + IPAdapter FaceID.

Reads pose PNGs from stage 1 and the face embedding from stage 2, then
generates raw frames at `fps_generate` using SD1.5 + AnimateDiff v3 +
ControlNet OpenPose + IPAdapter FaceID. FreeNoise context scheduling is
enabled so we can do >16 frames seamlessly. After generation, an ADetailer-
style face inpaint pass crops each frame to the face bbox (from stage 1's
face_bboxes.json), runs SD1.5 + IP-Adapter img2img at native resolution to
restore facial detail, and pastes back. This is what fixes the smeared faces
that full-body framing at 512×768 produces.

Outputs:
    outputs/<run_id>/raw_frames/0001.png ... NNNN.png   (post-detailer; what stage 4 reads)
    outputs/<run_id>/raw_frames/_contactsheet_pre.jpg   (pre-detailer, for diffing)
    outputs/<run_id>/raw_frames/_contactsheet.jpg       (post-detailer)
    outputs/<run_id>/raw_frames/_preview.mp4            (raw fps_generate playback)
"""
from __future__ import annotations

import gc
import json
import sys
import types
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageFilter

from video_pipeline_common import Paths, load_config, make_argparser, num_frames_for, setup_logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def _patch_unet_lora_for_motion_model(unet) -> None:
    """diffusers' `_load_ip_adapter_loras` enumerates `attn_processors.keys()`
    positionally to map the FaceID LoRA's index N → unet attn processor at
    position N. UNetMotionModel inserts motion-module attentions into that
    iteration order BEFORE spatial attentions at the same depth, so the LoRA's
    index 0 ends up targeting `down_blocks.0.motion_modules.0.attn1` (a
    320-dim motion attn) instead of `down_blocks.1.attentions.0.attn1` (a
    640-dim spatial attn). PEFT then crashes on the shape mismatch.

    Patch the method to enumerate ONLY spatial attns, restoring the SD1.5-base
    ordering the FaceID LoRA was trained against.
    """
    def _load_spatial_only(self, state_dicts):
        lora_dicts = {}
        spatial_names = [n for n in self.attn_processors.keys() if "motion_modules" not in n]
        for key_id, name in enumerate(spatial_names):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" not in state_dict["ip_adapter"]:
                    continue
                lora_dicts.setdefault(i, {})
                for proj in ("to_k", "to_q", "to_v", "to_out"):
                    for direction in ("down", "up"):
                        lora_dicts[i][f"unet.{name}.{proj}_lora.{direction}.weight"] = (
                            state_dict["ip_adapter"][f"{key_id}.{proj}_lora.{direction}.weight"]
                        )
        return lora_dicts

    unet._load_ip_adapter_loras = types.MethodType(_load_spatial_only, unet)


def load_pose_frames(pose_dir: Path, n: int) -> list[Image.Image]:
    frames = []
    for i in range(1, n + 1):
        p = pose_dir / f"{i:04d}.png"
        if not p.exists():
            raise FileNotFoundError(f"missing pose frame: {p}")
        frames.append(Image.open(p).convert("RGB"))
    return frames


def build_pipeline(cfg: dict, log):
    from diffusers import (
        AnimateDiffControlNetPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
        MotionAdapter,
    )

    g = cfg["generation"]
    log.info(f"loading motion adapter: {g['motion_adapter']}")
    adapter = MotionAdapter.from_pretrained(g["motion_adapter"], torch_dtype=DTYPE)
    log.info(f"loading controlnet: {g['controlnet']}")
    controlnet = ControlNetModel.from_pretrained(g["controlnet"], torch_dtype=DTYPE)
    log.info(f"loading base model:  {g['base_model']}")
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        g["base_model"],
        motion_adapter=adapter,
        controlnet=controlnet,
        torch_dtype=DTYPE,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )

    log.info(f"loading IP-Adapter FaceID: {g['ip_adapter_repo']} / {g['ip_adapter_weight']}")
    _patch_unet_lora_for_motion_model(pipe.unet)
    pipe.load_ip_adapter(
        g["ip_adapter_repo"],
        subfolder=None,
        weight_name=g["ip_adapter_weight"],
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(g["ip_adapter_scale"])

    # 80 frames × FreeNoise + motion adapter + ControlNet + IPAdapter Plus v2
    # + FaceID LoRA blows past 10 GB on a 3080 if everything stays GPU-resident
    # (the prior run thrashed at 9.5/10 GB, GPU at 100%, no step ever finished).
    # Memory savers, in increasing pain order:
    #   1. enable_vae_slicing       — VAE decodes one frame at a time. Free,
    #      eliminates the decode-time VRAM peak.
    #   2. enable_model_cpu_offload — sub-models live on CPU and stream onto
    #      GPU only when called. Big VRAM win, ~10–20% slower per step.
    # CPU offload manages device placement internally — do NOT call
    # pipe.to(DEVICE) when offload is on, it'd undo the offload.
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    log.info("memory savers on: vae_slicing + model_cpu_offload")

    if hasattr(pipe, "enable_free_noise"):
        pipe.enable_free_noise(
            context_length=g["context_frames"],
            context_stride=g["context_overlap"],
        )
        log.info(f"FreeNoise enabled: ctx={g['context_frames']} stride={g['context_overlap']}")
    else:
        log.warning("pipeline has no enable_free_noise — your diffusers version may be too old (need >=0.31)")

    return pipe


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


def _feather_mask(w: int, h: int, feather: int = 8) -> Image.Image:
    """Soft alpha mask that's opaque in the center and falls off at the edges
    so paste-backs from the face detailer don't show seams."""
    mask = Image.new("L", (w, h), 0)
    iw, ih = max(1, w - 2 * feather), max(1, h - 2 * feather)
    mask.paste(Image.new("L", (iw, ih), 255), (feather, feather))
    return mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))


def run_face_detailer(
    cfg: dict,
    paths,
    id_embeds: torch.Tensor,
    clip_embeds: torch.Tensor,
    log,
) -> None:
    """Per-frame face inpaint with SD1.5 img2img + IP-Adapter FaceID.

    Reads each frame's face bbox from stage 1's face_bboxes.json, crops with
    1.5× pad (computed in stage 1), img2img at 512×512 with strength 0.55 so
    the FaceID identity drives the new face but the rest of the crop stays put,
    then feathered-paste back. Uses the same id_embeds/clip_embeds the main
    AnimateDiff run did, so identity is consistent across the clip.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    bbox_path = paths.poses / "face_bboxes.json"
    if not bbox_path.exists():
        log.warning(
            "face_bboxes.json missing — skipping detailer. Re-run stage 1 to generate it."
        )
        return
    bboxes = json.loads(bbox_path.read_text())
    bbox_by_frame = {b["frame"]: b["bbox"] for b in bboxes}
    n_with = sum(1 for v in bbox_by_frame.values() if v is not None)
    if n_with == 0:
        log.warning("no face bboxes available — skipping detailer")
        return

    g = cfg["generation"]
    log.info(f"loading img2img pipeline for face detailer ({n_with}/{len(bbox_by_frame)} frames have a face bbox)")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        g["base_model"], torch_dtype=DTYPE, safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )
    pipe.load_ip_adapter(
        g["ip_adapter_repo"],
        subfolder=None,
        weight_name=g["ip_adapter_weight"],
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(g["ip_adapter_scale"])
    pipe.to(DEVICE)

    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds
    proj.shortcut = True
    proj.shortcut_scale = 1.0

    pipe.set_progress_bar_config(disable=True)

    n_done = 0
    for frame_idx in sorted(bbox_by_frame.keys()):
        bbox = bbox_by_frame[frame_idx]
        if bbox is None:
            continue
        frame_path = paths.raw_frames / f"{frame_idx:04d}.png"
        if not frame_path.exists():
            continue

        frame = Image.open(frame_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        crop = frame.crop((x1, y1, x2, y2))
        crop_in = crop.resize((512, 512), Image.LANCZOS)
        out = pipe(
            prompt=f"portrait, close-up, {g['prompt']}, sharp focus on face, detailed skin",
            negative_prompt=g["negative_prompt"],
            image=crop_in,
            strength=0.55,
            num_inference_steps=g["steps"],
            guidance_scale=g["guidance_scale"],
            ip_adapter_image_embeds=[id_embeds],
            generator=torch.Generator(device=DEVICE).manual_seed(int(cfg["seed"]) + frame_idx),
        )
        new_face = out.images[0].resize((x2 - x1, y2 - y1), Image.LANCZOS)
        frame.paste(new_face, (x1, y1), _feather_mask(x2 - x1, y2 - y1, feather=8))
        frame.save(frame_path)
        n_done += 1
        if n_done % 8 == 0:
            log.info(f"  …detailer {n_done}/{n_with}")

    log.info(f"detailer done: {n_done}/{n_with} frames detailed")
    del pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    args = make_argparser("03_generate").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)

    paths.raw_frames.mkdir(parents=True, exist_ok=True)

    n = num_frames_for(cfg)
    log.info(f"loading {n} pose frames…")
    pose_frames = load_pose_frames(paths.poses, n)

    emb_path = paths.reference / "face_embedding.pt"
    clip_path = paths.reference / "face_clip_embeds.pt"
    for p in (emb_path, clip_path):
        if not p.exists():
            log.error(f"{p.name} missing — run stage 2 first ({p})")
            sys.exit(1)
    id_embeds = torch.load(emb_path, map_location="cpu").to(device=DEVICE, dtype=DTYPE)
    clip_embeds = torch.load(clip_path, map_location="cpu").to(device=DEVICE, dtype=DTYPE)

    pipe = build_pipeline(cfg, log)
    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds
    proj.shortcut = True
    proj.shortcut_scale = 1.0

    g = cfg["generation"]
    # FreeNoise uses torch.randperm on this generator inside
    # `_prepare_latents_free_noise`, and randperm rejects CUDA generators —
    # so the generator MUST live on CPU (the pipeline still produces
    # CUDA-resident latents; only sampling uses the generator).
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    log.info(f"generating {n} frames at {g['width']}x{g['height']}, steps={g['steps']}…")
    out = pipe(
        prompt=g["prompt"],
        negative_prompt=g["negative_prompt"],
        num_frames=n,
        width=g["width"],
        height=g["height"],
        num_inference_steps=g["steps"],
        guidance_scale=g["guidance_scale"],
        conditioning_frames=pose_frames,
        controlnet_conditioning_scale=g["controlnet_scale"],
        ip_adapter_image_embeds=[id_embeds],
        generator=generator,
    )
    frames: list[Image.Image] = out.frames[0]

    log.info(f"writing {len(frames)} raw frames to {paths.raw_frames}")
    for i, fr in enumerate(frames, start=1):
        fr.save(paths.raw_frames / f"{i:04d}.png")

    write_contact_sheet(frames, paths.raw_frames / "_contactsheet_pre.jpg", every=8)
    log.info(f"wrote pre-detailer contact sheet → {paths.raw_frames / '_contactsheet_pre.jpg'}")

    preview_path = paths.raw_frames / "_preview.mp4"
    iio.imwrite(
        preview_path,
        np.stack([np.array(f) for f in frames]),
        plugin="pyav",
        fps=g["fps_generate"],
        codec="libx264",
    )
    log.info(f"wrote raw-fps preview → {preview_path}")

    # Free AnimateDiff pipeline before loading the img2img detailer — keeps
    # peak VRAM under one full pipeline at a time. Drop frames too; the
    # detailer reads from disk so we don't need them in RAM either.
    del pipe, frames, out
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    run_face_detailer(cfg, paths, id_embeds, clip_embeds, log)

    detailed_frames = [
        Image.open(paths.raw_frames / f"{i:04d}.png").convert("RGB")
        for i in range(1, n + 1)
    ]
    write_contact_sheet(detailed_frames, paths.raw_frames / "_contactsheet.jpg", every=8)
    log.info(f"wrote post-detailer contact sheet → {paths.raw_frames / '_contactsheet.jpg'}")
    log.info("INSPECT contact sheet + preview before stage 4. Compare _contactsheet.jpg vs _contactsheet_pre.jpg.")


if __name__ == "__main__":
    main()
