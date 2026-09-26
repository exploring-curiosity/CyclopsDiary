"""The streaming read with its language model in MLX (`cosmos3_mlx.MLXStream`).

The same bookkeeping as the torch reader (`cosmos3.Cosmos3Stream`: sink, frame
blocks, atoms, 3-D positions, window eviction); only the memory and the
language model's arithmetic move to MLX, which is what takes the read from
about one times real time on MPS to about three. So the check is parity: on a
tiny random Edge model in float32, the MLX memory state after every atom is
the torch reader's, with and without eviction, and the atom's word list (the
head on the atom's video tokens, the log of their mean probability, top-k) is
the one a one-shot read of the same frames gives.

MLX runs on the CPU here, so the check is the math, exact to float32
rounding (measured 1e-6). On the GPU the same math differs from torch by
about 1e-3 relative -- the GPU's float32 matmul, a property of the machine --
which is below the float16 the index stores a log-probability in.

Needs the side stack (transformers >= 5), MLX and the cached Cosmos3-Edge
processor and config; skipped otherwise:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=data/stacks/d40:python \
        python3 -m pytest tests/test_cosmos3_mlx.py -v -p no:django
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np                                    # noqa: E402
import pytest                                         # noqa: E402

MODEL_ID = "nvidia/Cosmos3-Edge"
PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def enc():
    mx = pytest.importorskip("mlx.core")
    was = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield _tiny()
    mx.set_default_device(was)


def _tiny():
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("the Cosmos 3 side stack (transformers >= 5) is not on the path")
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        cfg = AutoConfig.from_pretrained(MODEL_ID)
    except Exception as e:                            # noqa: BLE001
        pytest.skip(f"no cached {MODEL_ID}: {type(e).__name__}")
    cfg.text_config.num_hidden_layers = 2
    cfg.vision_config.num_hidden_layers = 1
    cfg.text_config.vocab_size = 32768                # every id the chat template uses is below 26k
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32).eval()
    from elidedb import cosmos3 as C3

    class Tiny(C3.Cosmos3Embed):
        def _load(self):
            self.proc, self.model = proc, model
    return Tiny(MODEL_ID, device="cpu", dtype=torch.float32)


def _frames(n, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (n, 256, 320, 3), dtype=np.uint8)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_the_mlx_memory_state_is_the_torch_readers_after_every_atom(enc):
    from elidedb import cosmos3_mlx as CM
    F = _frames(6)
    t = enc.stream(states=("memory",))
    t.open()
    m = CM.MLXStream(enc, window=None, dtype="float32")
    m.open()
    for i in (0, 2, 4):
        a = t.push(F[i:i + 2], idx=[i, i + 1], fps=4.0)
        b = m.push(F[i:i + 2], idx=[i, i + 1], fps=4.0)
        assert b.shape == a.shape and abs(np.linalg.norm(b) - 1) < 1e-5
        assert _cos(a, b) > 0.9999, (i, _cos(a, b))
    assert m.cache_len() == t.cache_len()


def test_the_mlx_window_evicts_what_the_torch_window_evicts(enc):
    from elidedb import cosmos3_mlx as CM
    F = _frames(6, seed=2)
    t = enc.stream(window=2 * 88, states=("memory",))
    t.open()
    m = CM.MLXStream(enc, window=2 * 88, dtype="float32")
    m.open()
    for i in (0, 2, 4):
        a = t.push(F[i:i + 2], idx=[i, i + 1], fps=4.0)
        b = m.push(F[i:i + 2], idx=[i, i + 1], fps=4.0)
        assert _cos(a, b) > 0.9999, (i, _cos(a, b))
        assert m.cache_len() == t.cache_len()


def test_the_atoms_word_list_is_the_heads_on_its_video_tokens(enc):
    """top_k: the atom's k best words by the log of their mean probability over
    the atom's video tokens, best first -- the list the index stores."""
    import torch
    from transformers.video_utils import VideoMetadata
    from elidedb import cosmos3_mlx as CM
    F = _frames(4, seed=1)
    m = CM.MLXStream(enc, window=None, dtype="float32", top_k=16)
    m.open()
    m.push(F[:2], idx=[0, 1], fps=4.0)
    m.push(F[2:], idx=[2, 3], fps=4.0)
    ids, lp = m.last_top
    assert ids.shape == (16,) and lp.shape == (16,) and np.all(np.diff(lp) <= 1e-6)
    txt = enc._chat(enc.prompt, True).split(PLACEHOLDER)[0] + PLACEHOLDER
    x = enc.proc(text=[txt], videos=[F], do_sample_frames=False, return_tensors="pt",
                 video_metadata=[VideoMetadata(total_num_frames=4, fps=4.0, duration=1.0, frames_indices=[0, 1, 2, 3])])
    with torch.no_grad():
        hs = enc.model(**x, output_hidden_states=True).hidden_states[-1][0]
        seq = x["input_ids"][0]
        L = m._atoms[-1][1] - m._atoms[-1][0]
        rows = torch.nonzero(seq == enc.model.model.config.video_token_id).flatten()
        rows = rows[rows >= len(seq) - L]
        p = torch.log_softmax(enc.model.lm_head(hs[rows]).float(), dim=-1)
        agg = (torch.logsumexp(p, dim=0) - math.log(len(rows))).numpy()
    np.testing.assert_allclose(lp, agg[ids], atol=1e-4)
    assert lp[-1] >= np.sort(agg)[-16] - 1e-4          # nothing better was left off the list
