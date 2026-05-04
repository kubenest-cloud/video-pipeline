"""Run all 5 stages end-to-end. Each stage runs in its own uv-managed venv.

Prefer running stages individually while debugging:
    uv run --project stages/03_generate python main.py --config ../../config.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("stages/01_extract_poses",   "01 extract_poses"),
    ("stages/02_reference_image", "02 reference_image"),
    ("stages/03_generate",        "03 generate"),
    ("stages/04_interpolate",     "04 interpolate"),
    ("stages/05_encode",          "05 encode"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--from-stage", type=int, default=1, help="1-indexed stage to start at")
    ap.add_argument("--to-stage", type=int, default=len(STAGES))
    args = ap.parse_args()

    uv = shutil.which("uv")
    if not uv:
        sys.exit("uv not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")

    env = os.environ.copy()
    # Stale VIRTUAL_ENV would make uv warn about every subprocess.
    env.pop("VIRTUAL_ENV", None)
    env["VIDEO_PIPELINE_ROOT"] = str(ROOT)
    # Share HF cache so stages 2 and 3 don't re-download multi-GB weights.
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

    cfg_abs = str(Path(args.config).resolve())

    for i, (stage_dir, label) in enumerate(STAGES, start=1):
        if i < args.from_stage or i > args.to_stage:
            continue
        print(f"\n========== Stage {i}: {label} ==========")
        t0 = time.time()
        try:
            subprocess.check_call(
                [uv, "run", "--project", str(ROOT / stage_dir),
                 "python", "main.py", "--config", cfg_abs],
                cwd=str(ROOT / stage_dir),
                env=env,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n[pipeline] stage {i} failed (exit {e.returncode}). Stopping.", file=sys.stderr)
            sys.exit(e.returncode)
        print(f"[pipeline] stage {i} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
