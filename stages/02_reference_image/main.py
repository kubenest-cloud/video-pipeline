"""Stage 2 — face reference → IPAdapter FaceID Plus v2 embeddings + preview grid.

Reads `input.face_ref`, runs InsightFace to extract a 512-d ID embedding and
the keypoint-aligned face crop, runs that crop through CLIP-H to get the
penultimate hidden states Plus v2 needs, then generates a 2x2 preview grid
using the SD1.5 + IPAdapter FaceID Plus v2 pipeline so you can EYEBALL
identity transfer before spending GPU on the full run.

Outputs:
    outputs/<run_id>/reference/face_embedding.pt   (2, 1, 512)   — InsightFace
    outputs/<run_id>/reference/face_clip_embeds.pt (2, 1, 257, 1280) — CLIP-H
    outputs/<run_id>/reference/face_crop.png       — bbox crop (audit)
    outputs/<run_id>/reference/face_aligned.png    — 224×224 aligned (CLIP input)
    outputs/<run_id>/reference/preview.png         — 2×2 grid
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from video_pipeline_common import REPO_ROOT, Paths, load_config, make_argparser, setup_logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def _flatten_antelopev2_if_nested(log) -> None:
    """insightface 0.7.3 auto-downloads antelopev2.zip but unzips into a nested
    `models/antelopev2/antelopev2/` because the zip's top-level entry is itself
    a folder. The loader then can't find the .onnx files. Flatten on detection.
    """
    base = Path.home() / ".insightface" / "models" / "antelopev2"
    nested = base / "antelopev2"
    if nested.is_dir() and any(nested.glob("*.onnx")):
        log.info(f"flattening nested {nested} → {base}")
        for f in nested.iterdir():
            f.rename(base / f.name)
        nested.rmdir()


def extract_face_embedding(image_path: Path, log) -> tuple[torch.Tensor, Image.Image, Image.Image]:
    """Returns (id_embedding, bbox_crop_for_audit, aligned_face_for_clip).

    - id_embedding: (2, 1, 512) tensor with neg=zeros and pos=normed_embedding,
      pre-packed for CFG (the pipeline chunks dim 0 into neg/pos halves).
    - bbox_crop_for_audit: PIL crop of the detected face bbox (debug aid).
    - aligned_face_for_clip: 224×224 PIL aligned by insightface keypoints,
      the input the FaceID Plus v2 CLIP-H encoder expects.
    """
    import cv2
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    # IP-Adapter FaceID was trained on antelopev2 (glintr100) embeddings, NOT
    # buffalo_l (w600k_r50). The two recognition models produce 512-d embeddings
    # in different spaces — feeding buffalo_l embeddings into the trained
    # projection lands on a near-random but consistent identity (e.g. flips
    # gender). Switching here is the actual fix to the gender-flip bug.
    _flatten_antelopev2_if_nested(log)
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    faces = app.get(bgr)
    if not faces:
        raise RuntimeError(f"No face detected in {image_path}. Use a clear frontal portrait.")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    log.info(f"face detected, det_score={face.det_score:.3f}, bbox={face.bbox.tolist()}")

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    bbox_crop = bgr[max(0, y1):y2, max(0, x1):x2][:, :, ::-1]
    bbox_crop_pil = Image.fromarray(bbox_crop)

    aligned_bgr = face_align.norm_crop(bgr, landmark=face.kps, image_size=224)
    aligned_pil = Image.fromarray(aligned_bgr[:, :, ::-1])

    pos = torch.from_numpy(face.normed_embedding).view(1, 1, -1)
    neg = torch.zeros_like(pos)
    emb = torch.cat([neg, pos], dim=0)
    return emb, bbox_crop_pil, aligned_pil


def build_pipeline(cfg: dict, log):
    """Plain SD1.5 + IPAdapter FaceID Plus v2. No motion adapter — stage 2 only
    renders stills, and the FaceID LoRA weights have SD1.5 attention dims
    (320/768/1280) that don't match AnimateDiff's motion modules, so loading
    both together triggers a size-mismatch error.

    `image_encoder_folder=None` skips diffusers' auto-loaded CLIP from the
    IP-Adapter repo. Plus v2 needs CLIP-H penultimate hidden states wired
    manually into the projection layer's `clip_embeds`, which `attach_clip_embeds`
    does after this returns.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    g = cfg["generation"]
    log.info(f"loading base model: {g['base_model']}")
    pipe = StableDiffusionPipeline.from_pretrained(g["base_model"], torch_dtype=DTYPE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )

    log.info(f"loading IP-Adapter FaceID: {g['ip_adapter_repo']} / {g['ip_adapter_weight']}")
    pipe.load_ip_adapter(
        g["ip_adapter_repo"],
        subfolder=None,
        weight_name=g["ip_adapter_weight"],
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(g["ip_adapter_scale"])
    pipe.to(DEVICE)

    try:
        active = pipe.get_active_adapters()
    except Exception as e:
        active = f"<get_active_adapters failed: {e!r}>"
    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    log.info(
        f"ip-adapter wired: scale={g['ip_adapter_scale']}, projection={type(proj).__name__}, "
        f"num_tokens={getattr(proj, 'num_tokens', '?')}, active_adapters={active}"
    )
    return pipe


def encode_clip_embeds(aligned_face: Image.Image, encoder_repo: str, log) -> torch.Tensor:
    """Encode the aligned face with CLIP-H, return penultimate hidden states
    pre-packed for CFG: shape (2, 1, 257, 1280) where dim 0 is [neg, pos].

    Plus v2's projection reads `self.clip_embeds` directly (no kwarg path).
    The pipeline's CFG plumbing chunks `ip_adapter_image_embeds` along dim 0
    but does NOT touch `clip_embeds` — so we pack neg/pos here and the
    projection's `proj_in` reshape collapses (2, 1, 257, 1280) → (2, 257, 1280).
    """
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    log.info(f"loading CLIP-H image encoder for Plus v2: {encoder_repo}")
    processor = CLIPImageProcessor.from_pretrained(encoder_repo)
    encoder = CLIPVisionModelWithProjection.from_pretrained(encoder_repo, torch_dtype=DTYPE).to(DEVICE)
    encoder.eval()

    pixel_values = processor(images=aligned_face, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        pos = encoder(pixel_values, output_hidden_states=True).hidden_states[-2]
        # CRITICAL: pass a zero IMAGE through CLIP for the unconditional, not
        # zero embeddings. CLIP's positional embeds + LN make the "blank-image"
        # representation non-trivial — feeding raw zeros downstream cancels the
        # residual identity signal in Plus v2's shortcut path.
        neg = encoder(torch.zeros_like(pixel_values), output_hidden_states=True).hidden_states[-2]
    clip_embeds = torch.stack([neg, pos], dim=0)

    del encoder, processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return clip_embeds.cpu()


def attach_clip_embeds(pipe, clip_embeds: torch.Tensor) -> None:
    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds.to(device=DEVICE, dtype=DTYPE)
    proj.shortcut = True
    proj.shortcut_scale = 1.0


def make_grid(imgs: list[Image.Image], cols: int = 2) -> Image.Image:
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "black")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def main():
    args = make_argparser("02_reference_image").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)

    paths.reference.mkdir(parents=True, exist_ok=True)

    ref_path = Path(cfg["input"]["face_ref"])
    if not ref_path.is_absolute():
        ref_path = REPO_ROOT / ref_path
    if not ref_path.exists():
        log.error(f"face_ref does not exist: {ref_path}")
        sys.exit(1)

    log.info("extracting face embedding (insightface antelopev2 — what FaceID was trained on)…")
    emb, bbox_crop, aligned_face = extract_face_embedding(ref_path, log)
    torch.save(emb, paths.reference / "face_embedding.pt")
    bbox_crop.save(paths.reference / "face_crop.png")
    aligned_face.save(paths.reference / "face_aligned.png")
    log.info(f"saved embedding shape={tuple(emb.shape)} → {paths.reference / 'face_embedding.pt'}")

    g = cfg["generation"]
    clip_embeds = encode_clip_embeds(aligned_face, g["ip_adapter_image_encoder"], log)
    torch.save(clip_embeds, paths.reference / "face_clip_embeds.pt")
    log.info(f"saved clip embeds shape={tuple(clip_embeds.shape)} → {paths.reference / 'face_clip_embeds.pt'}")

    log.info("building preview pipeline (single image, no motion adapter use)…")
    pipe = build_pipeline(cfg, log)
    attach_clip_embeds(pipe, clip_embeds)
    id_embeds = emb.to(device=DEVICE, dtype=DTYPE)

    log.info("generating 4 preview stills (sanity check identity)…")
    previews: list[Image.Image] = []
    for i in range(4):
        out = pipe(
            prompt=g["prompt"],
            negative_prompt=g["negative_prompt"],
            width=g["width"],
            height=g["height"],
            num_inference_steps=g["steps"],
            guidance_scale=g["guidance_scale"],
            ip_adapter_image_embeds=[id_embeds],
            generator=torch.Generator(device=DEVICE).manual_seed(int(cfg["seed"]) + i),
        )
        previews.append(out.images[0])

    grid = make_grid(previews, cols=2)
    grid.save(paths.reference / "preview.png")
    log.info(f"wrote preview grid → {paths.reference / 'preview.png'}")
    log.info("INSPECT THIS GRID before running stage 3. If identity is off, fix the reference image.")


if __name__ == "__main__":
    main()
