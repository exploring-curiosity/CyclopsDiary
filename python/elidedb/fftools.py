"""Locate the ffmpeg/ffprobe binaries robustly.

Why this exists: a macOS .app (or anything launched from Finder/launchd) runs
with a minimal PATH — typically `/usr/bin:/bin:/usr/sbin:/sbin` — which does
NOT include Homebrew's `/opt/homebrew/bin`. A bare `["ffmpeg", ...]` then
fails with `FileNotFoundError` even though ffmpeg is installed. We resolve the
absolute path once, checking (in order): an explicit env override, the current
PATH, and the common install locations on macOS/Linux.
"""
from __future__ import annotations

import functools
import os
import shutil

_COMMON_DIRS = [
    "/opt/homebrew/bin",   # Apple Silicon Homebrew
    "/usr/local/bin",      # Intel Homebrew / manual installs
    "/opt/local/bin",      # MacPorts
    "/usr/bin",
]


@functools.lru_cache(maxsize=None)
def find(tool: str) -> str:
    """Absolute path to `ffmpeg`/`ffprobe`, or raise with a fixable message."""
    env = os.environ.get(f"ELIDEDB_{tool.upper()}")
    if env and os.path.exists(env):
        return env
    found = shutil.which(tool)
    if found:
        return found
    for d in _COMMON_DIRS:
        cand = os.path.join(d, tool)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"{tool} not found. Install it with `brew install ffmpeg`, or set "
        f"ELIDEDB_{tool.upper()}=/path/to/{tool}. (Apps launched from Finder "
        f"get a minimal PATH, so Homebrew's bin dir may not be visible.)")
