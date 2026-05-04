"""Stage 5 — encode interpolated frames to final.mp4 (h264) and optionally h265 + gif.

Outputs:
    outputs/<run_id>/final.mp4
    outputs/<run_id>/final_h265.mp4   (if encode.also_h265)
    outputs/<run_id>/preview.gif      (if encode.preview_gif, 3s loop)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from video_pipeline_common import Paths, load_config, make_argparser, setup_logging


def require_ffmpeg(log) -> None:
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg not found on PATH — install it (`apt install ffmpeg`)")
        sys.exit(1)


def encode_video(frames_glob: str, out_path: Path, fps: int, codec: str, crf: int, pix_fmt: str, log) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_glob,
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
        "-movflags", "+faststart",
        str(out_path),
    ]
    log.info("encoding: " + " ".join(cmd))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def encode_gif(in_mp4: Path, out_gif: Path, max_seconds: int, log) -> None:
    palette = out_gif.with_suffix(".palette.png")
    subprocess.check_call(
        ["ffmpeg", "-y", "-t", str(max_seconds), "-i", str(in_mp4),
         "-vf", "fps=12,scale=480:-1:flags=lanczos,palettegen", str(palette)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["ffmpeg", "-y", "-t", str(max_seconds), "-i", str(in_mp4), "-i", str(palette),
         "-lavfi", "fps=12,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse", str(out_gif)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)
    log.info(f"wrote preview gif → {out_gif}")


def main():
    args = make_argparser("05_encode").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)
    require_ffmpeg(log)

    fps = cfg["interpolation"]["target_fps"]
    enc = cfg["encode"]

    frames_glob = str(paths.interpolated / "%04d.png")
    if not list(paths.interpolated.glob("[0-9]*.png")):
        log.error(f"no frames in {paths.interpolated} — run stage 4 first")
        sys.exit(1)

    encode_video(frames_glob, paths.final_mp4, fps, enc["codec"], enc["crf"], enc["pix_fmt"], log)

    if enc.get("also_h265"):
        h265_path = paths.run_dir / "final_h265.mp4"
        encode_video(frames_glob, h265_path, fps, "libx265", enc["crf"], enc["pix_fmt"], log)

    if enc.get("preview_gif"):
        encode_gif(paths.final_mp4, paths.run_dir / "preview.gif", max_seconds=3, log=log)

    log.info(f"DONE — final clip: {paths.final_mp4}")


if __name__ == "__main__":
    main()
