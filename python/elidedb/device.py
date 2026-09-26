"""One device policy for every model loader.

The store format is portable; the model loaders must be too. Every
loader asks this module instead of probing torch itself, so a cloud
CPU box, an Apple laptop, and a CUDA node run the same code with the
right placement:

  ELIDEDB_DEVICE  force a device ("cpu", "mps", "cuda"); default is
                  mps when available, else cuda, else cpu
  ELIDEDB_DTYPE   force a torch dtype name; default float16 on
                  mps/cuda and bfloat16 on cpu (halves resident
                  memory against fp32, and fp16 matmuls are not a
                  real CPU option)
"""
from __future__ import annotations

import os


def text_only():
    """Serving deployments set ELIDEDB_TEXT_ONLY=1: the query path
    encodes TEXT only (every frame vector is precomputed at ingest),
    so the loaders drop their vision towers after load and roughly
    halve resident memory. Ingest machines leave it unset."""
    return os.environ.get("ELIDEDB_TEXT_ONLY", "") == "1"


def strip_vision(model, *attrs):
    """Release the named submodules when serving text-only."""
    if not text_only():
        return model
    import gc
    for a in attrs:
        if hasattr(model, a):
            setattr(model, a, None)
    gc.collect()
    return model


#: Devices with a real half-precision matmul path. Autocast anywhere else
#: is pure cost: the casts are work and the kernels they feed are not
#: faster. Measured on one 8-frame exposure, seconds per video second:
#:
#:             224p fp32  224p fp16  224p bf16  336p fp32  336p fp16  336p bf16
#:   aarch64        1.66      12.98      16.78       3.79      47.95      51.27
#:   mps               -       0.16          -       0.99          -          -
HALF_DEVICES = ("mps", "cuda", "xpu")

_AUTOCAST = {"fp16": "float16", "float16": "float16",
             "bf16": "bfloat16", "bfloat16": "bfloat16"}


def autocast_for(requested, device: str | None = None):
    """The autocast dtype to actually run under, given the one asked for.

    A store's pin records the dtype its trace was built with, and the
    store then travels to machines the ingesting one never saw. Applying
    it everywhere is how a 224p/fp16 pilot store came to cost 12.98 s per
    video second on a CPU box where the same tower in fp32 costs 1.66.

    Dropping it is safe because autocast never touches the weights: the
    two vectors for one clip agree to cos 1.000000, and pairwise
    similarities to 1.1e-5. The dtype is a performance note about the
    machine that wrote the trace, not part of the space it lives in.
    """
    name = _AUTOCAST.get(str(requested or "").strip().lower())
    if name is None:
        return None
    dev = device or pick()[0]
    if dev not in HALF_DEVICES:
        return None
    import torch
    return getattr(torch, name)


def pick():
    import torch
    dev = os.environ.get("ELIDEDB_DEVICE", "").strip()
    if not dev:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    name = os.environ.get("ELIDEDB_DTYPE", "").strip()
    if name:
        dtype = getattr(torch, name)
    else:
        dtype = torch.float16 if dev in ("mps", "cuda") \
            else torch.bfloat16
    return dev, dtype
