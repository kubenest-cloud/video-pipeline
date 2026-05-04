"""Shared helpers for stage scripts. Kept tiny on purpose."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


def _repo_root() -> Path:
    v = os.environ.get("VIDEO_PIPELINE_ROOT")
    if v:
        return Path(v).resolve()
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "config.yaml").exists():
            return cand
    raise RuntimeError(
        "Cannot locate video-pipeline repo root. "
        "Set VIDEO_PIPELINE_ROOT or run from a directory under the repo."
    )


REPO_ROOT = _repo_root()


@dataclass
class Paths:
    run_dir: Path
    raw_frames: Path
    interpolated: Path
    final_mp4: Path
    log: Path

    @classmethod
    def for_run(cls, run_id: str) -> "Paths":
        run_dir = REPO_ROOT / "outputs" / run_id
        return cls(
            run_dir=run_dir,
            raw_frames=run_dir / "raw_frames",
            interpolated=run_dir / "interpolated",
            final_mp4=run_dir / "final.mp4",
            log=run_dir / "run.log",
        )

    def ensure(self) -> None:
        """Create only the run dir (where run.log lives). Each stage mkdirs its own output subdir."""
        self.run_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_argparser(stage_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Stage: {stage_name}")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    return p


def setup_logging(paths: Paths) -> logging.Logger:
    paths.ensure()
    logger = logging.getLogger("video-pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(paths.log, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def num_frames_for(cfg: dict) -> int:
    g = cfg["generation"]
    return int(g["fps_generate"] * g["duration_sec"])
