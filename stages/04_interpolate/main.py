"""Stage 4 — RIFE frame interpolation: fps_generate → target_fps.

We shell out to Practical-RIFE rather than reimplementing it. On first run,
this stage clones the repo into `./third_party/Practical-RIFE` (relative to
this file) and expects the RIFE model checkpoints to be downloaded there per
its README. Set the model version via `interpolation.rife_model` in config.yaml.

We build an intermediate raw mp4 from raw_frames, run RIFE on that to upsample,
then split back into PNG frames in `interpolated/`.

Outputs:
    outputs/<run_id>/interpolated/0001.png ... NNNN.png
    outputs/<run_id>/interpolated/_compare.mp4   (raw vs interpolated, side-by-side)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from video_pipeline_common import Paths, load_config, make_argparser, num_frames_for, setup_logging

RIFE_DIR = Path(__file__).resolve().parent / "third_party" / "Practical-RIFE"


def ensure_rife(cfg: dict, log) -> Path:
    if RIFE_DIR.exists():
        log.info(f"RIFE present at {RIFE_DIR}")
        if not (RIFE_DIR / "train_log").is_dir() or not any((RIFE_DIR / "train_log").iterdir()):
            log.error(
                f"RIFE checkpoints missing in {RIFE_DIR / 'train_log'}. "
                f"Download v{cfg['interpolation']['rife_model']} weights per "
                f"{RIFE_DIR}/README.md before re-running stage 4."
            )
            sys.exit(1)
        return RIFE_DIR
    RIFE_DIR.parent.mkdir(parents=True, exist_ok=True)
    repo = cfg["interpolation"]["rife_repo"]
    log.info(f"cloning {repo} → {RIFE_DIR}")
    subprocess.check_call(["git", "clone", "--depth", "1", repo, str(RIFE_DIR)])
    log.warning(
        "RIFE cloned, but model checkpoints are NOT auto-downloaded. "
        f"Follow {RIFE_DIR}/README.md to place the v{cfg['interpolation']['rife_model']} weights "
        f"under {RIFE_DIR}/train_log/ before re-running stage 4."
    )
    sys.exit(1)


def frames_to_mp4(frames_dir: Path, out_mp4: Path, fps: int, log) -> None:
    files = sorted(frames_dir.glob("[0-9]*.png"))
    if not files:
        raise FileNotFoundError(f"no PNGs in {frames_dir}")
    arr = np.stack([np.array(iio.imread(p)) for p in files])
    iio.imwrite(out_mp4, arr, plugin="pyav", fps=fps, codec="libx264")
    log.info(f"packed {len(files)} frames → {out_mp4} @ {fps}fps")


def run_rife(rife_dir: Path, in_mp4: Path, out_mp4: Path, target_fps: int, log) -> None:
    """Invoke Practical-RIFE inference_video.py to upsample to target_fps."""
    cmd = [
        sys.executable, "inference_video.py",
        "--video", str(in_mp4.resolve()),
        "--output", str(out_mp4.resolve()),
        "--fps", str(target_fps),
    ]
    log.info("running RIFE: " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(rife_dir))


def mp4_to_frames(in_mp4: Path, frames_dir: Path, log) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("*.png"):
        p.unlink()
    n = 0
    for i, frame in enumerate(iio.imiter(in_mp4, plugin="pyav"), start=1):
        iio.imwrite(frames_dir / f"{i:04d}.png", frame)
        n = i
    log.info(f"split {in_mp4.name} → {n} PNGs in {frames_dir}")
    return n


def make_compare(raw_mp4: Path, interp_mp4: Path, out_mp4: Path, log) -> None:
    """ffmpeg side-by-side. Raw is slowed to match real time of interp for fair compare."""
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found, skipping _compare.mp4")
        return
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_mp4),
        "-i", str(interp_mp4),
        "-filter_complex",
        "[0:v]scale=iw:ih,setpts=PTS-STARTPTS[a];"
        "[1:v]scale=iw:ih,setpts=PTS-STARTPTS[b];"
        "[a][b]hstack=inputs=2",
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info(f"wrote side-by-side compare → {out_mp4}")


def main():
    args = make_argparser("04_interpolate").parse_args()
    cfg = load_config(args.config)
    paths = Paths.for_run(cfg["run_id"])
    log = setup_logging(paths)

    n = num_frames_for(cfg)
    raw_count = len(list(paths.raw_frames.glob("[0-9]*.png")))
    if raw_count < n:
        log.error(f"expected {n} raw frames, found {raw_count} — run stage 3 first")
        sys.exit(1)

    rife_dir = ensure_rife(cfg, log)

    raw_mp4 = paths.run_dir / "_raw.mp4"
    interp_mp4 = paths.run_dir / "_interp.mp4"

    frames_to_mp4(paths.raw_frames, raw_mp4, cfg["generation"]["fps_generate"], log)
    run_rife(rife_dir, raw_mp4, interp_mp4, cfg["interpolation"]["target_fps"], log)
    n_out = mp4_to_frames(interp_mp4, paths.interpolated, log)

    make_compare(raw_mp4, interp_mp4, paths.interpolated / "_compare.mp4", log)
    log.info(f"interpolation complete: {raw_count} → {n_out} frames")


if __name__ == "__main__":
    main()
