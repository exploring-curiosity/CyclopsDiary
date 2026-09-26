"""Settings, from the environment and the project's .env (a variable already set wins).

The memory is MongoDB: MONGODB_URI (the Atlas connection string, which holds
a database user's password -- it lives in .env, which git ignores) and
MONGODB_DB. The read defaults are the setting the example-search numbers were
measured at: two frames a step at 2 fps (a 1-s step), about 384x512 pixels a
frame with the source's aspect kept, the last 8192 tokens of video in the
model's memory, each step's 512 best words kept for the index.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STACK = ROOT / "data" / "stacks" / "d40"          # transformers >= 5 for Cosmos3-Edge (data/stacks/README.md)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str | None
    db: str
    model: str
    rate: float
    n_frames: int
    frame_px: int
    window: int
    top_k: int
    union_k: int

    @property
    def atom_s(self) -> float:
        return self.n_frames / self.rate


def get() -> Settings:
    load_env()
    e = os.environ.get
    return Settings(
        mongodb_uri=e("MONGODB_URI") or None,
        db=e("MONGODB_DB") or "cyclopsdiary",
        model=e("CYCLOPSDIARY_MODEL") or "nvidia/Cosmos3-Edge",
        rate=float(e("CYCLOPSDIARY_RATE") or 2.0),
        n_frames=int(e("CYCLOPSDIARY_FRAMES") or 2),
        frame_px=int(e("CYCLOPSDIARY_FRAME_PX") or 384 * 512),
        window=int(e("CYCLOPSDIARY_WINDOW") or 8192),
        top_k=int(e("CYCLOPSDIARY_TOP_K") or 512),
        union_k=int(e("CYCLOPSDIARY_UNION_K") or 128),
    )
