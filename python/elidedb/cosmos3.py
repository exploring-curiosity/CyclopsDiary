"""Cosmos 3 Edge as the stream's tower: the model's own summary state. The
same read serves its siblings (Cosmos3-Nano, Cosmos-Reason2: `_load` takes
the checkpoint's own class), so a pre-check of another world model changes
nothing but the model id in the pin.

One frozen world model reads both sides. A video exposure goes through the
model's chat template with a question, and the vector is the last-layer
hidden state at the final position -- the state the model is in as it is
about to answer. A sentence goes through the same shape with the SAME
question, the sentence standing where the video stands, so the two sides
share a space by construction and no text head is fitted (owner rule: no
text head from scratch, no training).

The question is the space. Asked to summarise, a video's state is the
room; asked what CHANGED from the start to the end, it is the event
(BENCHMARKS 2026-09-17: +0.15..+0.27 p@1 across kitchens). And the text
side must be asked the same question, verbatim: a sentence asked to
"summarize the above sentence in one word" lands in another subspace of
the same model, and the words entry read 0.50 / 0.38 at one on bridge 13 /
18 for it; asked the video's question it lands beside the video's states
(0.89 / 0.41 through the same entry, 0.81 / 0.61 as a one-step example).
Paraphrases of the question cost most of that gain (a "the sentence above
describes a video" preface 0.67, the instruction before the sentence 0.57,
a system-turn anchor on the text side alone 0.60), which is the E5-V
finding (the "in one word" answer slot is where the modalities meet) and
the symmetric task-instruction finding of arXiv 2508.00955 reproduced on
this tower. So there is one prompt, recorded in the store's pin, and it
is asked of both sides.

Not centred (`CENTRE = False`): subtracting the stream's mean collapsed
this tower to 0.12-0.17 p@1 (its two sides are already one space), so the
index is the unit vectors as the model left them.

Same interface as streamtext.CosmosEmbed and qwenembed.QwenVLEmbed:
embed_clips(clips, clocks=None) with clips of (T, H, W, 3) uint8 frames,
embed_text(texts), .id, .dev, .autocast; TraceEmbedder hands over the
clock (`wants_clock`) because the processor writes one timestamp per
frame pair into the prompt from the source rate and frame indices --
given frames alone it assumes 24 fps, and a 2-s exposure would read as a
third of a second.

Forwards only, never `generate()`: on MPS the second generate() call of a
process traps the interpreter (exit 133, no traceback).

The streaming read (`Cosmos3Stream`, 2026-09-20) is the same question
asked of a recording read as ONE growing video instead of 2-s exposures:
the frames go in as they arrive, the model's memory (its KV cache) is
carried forward, and after every atom the question is asked on that
memory and cropped again, so the state is the moment seen with
everything before it. The owner's rule against fixed-second chunks and
the field's answer to it (EM-LLM, SelectStream, Gorlo et al. 2026: read
the stream with the memory carried, cut by the model's own surprise)
meet here; see docs/research/2026-09-20-world-model-plan-95.md.

Runs on transformers >= 5 (the checkpoint's own architecture class),
which the main environment does not carry: the encoder service alone is
started with the side stack on its path (`scripts/desk.py --encoder-stack`).
"""
from __future__ import annotations

from collections import deque

import numpy as np

MODEL_ID = "nvidia/Cosmos3-Edge"                        # OpenMDW-1.1
VIDEO_PROMPT = "What changed from the start to the end of the above video? Answer in one word:"
FRAME_HW = (256, 320)                                   # (H, W) the frames go in at: 640x480 kept in aspect
PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"   # where the chat template puts the video


class Cosmos3Embed:
    CENTRE = False
    wants_clock = True
    BATCH = 1            # one exposure per forward: a language model over video tokens, no padding across clips
    TEXT_BATCH = 1

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, dtype=None,
                 prompt: str = VIDEO_PROMPT, text_prompt: str | None = None, frame_hw=FRAME_HW):
        import torch
        from . import device as DV
        self.id = model_id
        self.dev = device or DV.pick()[0]
        self.dtype = dtype or torch.bfloat16        # the checkpoint's own dtype; weights run in it, no autocast
        self.autocast = None                        # accepted for interface parity, unused
        self.prompt = prompt
        self.text_prompt = text_prompt or prompt    # one question, both sides (see the module docstring)
        self.frame_hw = tuple(frame_hw) if frame_hw else None
        self._load()

    def _load(self) -> None:
        # The checkpoint's own class through the auto mapping: Cosmos3-Edge, the
        # omni siblings (Cosmos3-Nano / -Super, whose generator, sound and action
        # experts are dropped at load and only the autoregressive tower stays)
        # and Cosmos-Reason2 (Qwen3-VL architecture) all read the same way.
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.id, dtype=self.dtype).to(self.dev).eval()

    # ---- the model's own input format ----------------------------------------
    def _chat(self, question: str, with_video: bool) -> str:
        content = ([{"type": "video"}] if with_video else []) + [{"type": "text", "text": question}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    def _inputs(self, text: str, px=None, clock: dict | None = None):
        from transformers.video_utils import VideoMetadata
        kw = dict(text=[text], return_tensors="pt")
        if px is not None:
            n = int(px.shape[0])
            clock = clock or dict(fps=1.0, idx=list(range(n)), total=n)
            kw.update(videos=[px], do_sample_frames=False,
                      video_metadata=[VideoMetadata(total_num_frames=int(clock["total"]), fps=float(clock["fps"]),
                                                    duration=float(clock["total"]) / float(clock["fps"]),
                                                    frames_indices=[int(i) for i in clock["idx"]])])
        x = self.proc(**kw)
        return {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in x.items()}

    def _last_state(self, x) -> np.ndarray:
        h = self.model(**x, output_hidden_states=True).hidden_states[-1][0, -1].float()
        return (h / h.norm()).cpu().numpy()

    def _frames(self, c) -> np.ndarray:
        c = np.asarray(c)
        if self.frame_hw and tuple(c.shape[1:3]) != self.frame_hw:
            import cv2
            H, W = self.frame_hw
            c = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in c])
        return np.ascontiguousarray(c)

    # ---- the interface ------------------------------------------------------------
    def embed_clips(self, clips, clocks=None) -> np.ndarray:
        import torch
        clips = list(clips)
        out = []
        with torch.no_grad():
            for k, c in enumerate(clips):
                clock = clocks[k] if clocks is not None else None
                out.append(self._last_state(self._inputs(self._chat(self.prompt, True), self._frames(c), clock)))
        return np.stack(out) if out else np.zeros((0, 1), np.float32)

    def embed_text(self, texts, batch: int = 0, memory: bool = False) -> np.ndarray:
        """The sentence where the video would be, then the tower's question.

        `memory=True` is the sentence read the way the memory state reads a
        video (`Cosmos3Stream`, states="memory"): the user turn holds the
        sentence and nothing else, no question, and the state is the last
        token's -- the model's state after reading it."""
        import torch
        out = []
        with torch.no_grad():
            for t in texts:
                if memory:
                    ids = self.proc.tokenizer(self.prefix() + t, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    h = self.model(input_ids=ids.to(self.dev), output_hidden_states=True).hidden_states[-1][0, -1].float()
                    out.append((h / h.norm()).cpu().numpy())
                else:
                    out.append(self._last_state(self._inputs(self._chat(f"{t}\n{self.text_prompt}", False))))
        return np.stack(out) if out else np.zeros((0, 1), np.float32)

    def prefix(self) -> str:
        """The chat template up to where the video goes."""
        return self._chat(self.prompt, True).split(PLACEHOLDER, 1)[0]

    def stream(self, window: int | None = None, states=("answer",)) -> "Cosmos3Stream":
        """A reader of one recording at a time through this tower's memory."""
        return Cosmos3Stream(self, window=window, states=states)


class Cosmos3Stream:
    """One recording read as one growing video, a state per atom.

    open() starts a recording: the chat template's prefix goes into a
    fresh memory (the KV cache) and is the attention sink. push(frames,
    idx, fps) appends an atom (a frame pair at the tower's rate) as the
    processor itself lays a frame out -- `<t seconds>`, vision start, the
    patches, vision end, the timestamp from the source index and rate --
    with the 3-D positions the model gives the same frames in one prompt,
    then asks the pin's question on the memory, reads the state at the
    answer slot and crops the question's tokens off again. Causal
    attention makes the k-th state identical to the one-shot read of the
    first k atoms with the question at the end (tests/test_cosmos3_stream),
    so nothing about the space changes: the row is the moment seen with
    everything before it instead of a 2-s exposure alone.

    The event probe (`probe_span`): the same question asked over a SPAN of
    atoms of the carried memory -- a view of the memory that holds the sink
    and the span's own tokens, the probe placed right after the span's end,
    nothing kept afterwards. The span's tokens are what the model wrote for
    them when they arrived (with everything before them in view), so this is
    the event as the world model remembered it, not a clip read on its own;
    over the whole recording it is the streaming probe itself. It can hand
    over the answer's word distribution beside the state (top-k ids and
    log-probs through the model's own head).

    Memory: a recording's memory grows with its length. Within the model's
    context (131k positions; 88 tokens a frame, 350 a second, about six
    minutes) it is left whole. `window` is a token budget for recordings
    longer than that (StreamingLLM: keep the sink, drop the oldest frames,
    keep going): a compute budget, not a semantic constant, and the pin
    records it. Positions never restart inside a recording, so the kept
    frames stay at their own times.
    """

    STATES = ("answer", "memory")

    def __init__(self, enc: Cosmos3Embed, window: int | None = None, states=("answer",)):
        self.enc = enc
        # Which states of the moment a row carries, in this order, each half a
        # unit vector over sqrt(len): "answer" is the state at the answer slot
        # after the question, "memory" the last hidden state at the last video
        # token before any question -- the world model's own state of now,
        # given the past. Measured on rollout episode 0 (2026-09-20): the
        # answer state converges to a summary of the recording as the memory
        # grows (last-third consecutive cos .998) while the memory state keeps
        # moving with the moment (.806) and its change marks the annotated
        # step edges (AUC .76 vs .46); the grade decides which half is the row.
        self.states = tuple(states)
        if not self.states or any(x not in self.STATES for x in self.states):
            raise ValueError(f"states={states!r}: each of {self.STATES}, at least one")
        self.core = enc.model.model                       # the model: visual tower, projector, language model
        self.lm = self.core.language_model
        self.proc, self.tok, self.dev = enc.proc, enc.proc.tokenizer, enc.dev
        pre, post = enc._chat(enc.prompt, True).split(PLACEHOLDER, 1)
        self.prefix_ids = list(self.tok(pre, add_special_tokens=False)["input_ids"])
        self.probe_ids = list(self.tok(post, add_special_tokens=False)["input_ids"])
        self.window = int(window) if window else None
        self.merge = int(self.core.config.vision_config.spatial_merge_size)
        self.video_id = int(self.core.config.video_token_id)
        self.cache = None
        self.pos = 0                                      # the next 1-D position (mRoPE's running counter)
        self._prefix_pos = None
        self._blocks: deque = deque()                     # (tokens, positions) per frame after the sink, oldest first
        self._atoms: list = []                            # (token start, token end, position after) per atom since open()
        self._tok_total = 0                               # tokens pushed since open(), the sink excluded
        self._dropped = 0                                 # of those, evicted from the front

    # ---- the memory -------------------------------------------------------------------
    def open(self) -> None:
        import torch
        from transformers import DynamicCache
        self.cache = DynamicCache(config=self.lm.config)
        self.pos = 0
        self._blocks.clear()
        self._atoms, self._tok_total, self._dropped = [], 0, 0
        ids = torch.tensor([self.prefix_ids], device=self.dev)
        pos = self._text_positions(len(self.prefix_ids))
        self._prefix_pos = pos[:, 0, :].cpu()
        with torch.inference_mode():
            self._feed(self.core.get_input_embeddings()(ids), pos)

    def cache_len(self) -> int:
        return int(self.cache.get_seq_length()) if self.cache is not None else 0

    def positions(self):
        """The 3-D positions of what the memory holds, (3, n): the sink and the kept frames."""
        import torch
        return torch.cat([self._prefix_pos] + [p for _, p in self._blocks], dim=1)

    def _text_positions(self, n: int):
        import torch
        p = torch.arange(self.pos, self.pos + n, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        self.pos += n
        return p

    def _feed(self, emb, pos):
        out = self.lm(inputs_embeds=emb, position_ids=pos, past_key_values=self.cache, use_cache=True)
        return out.last_hidden_state

    # ---- an atom ------------------------------------------------------------------------
    def push(self, frames, idx, fps: float) -> np.ndarray:
        """frames (k, H, W, 3) uint8 at source indices `idx` of a source at `fps`; -> the unit state."""
        import torch
        from transformers.video_utils import VideoMetadata
        if self.cache is None:
            self.open()
        frames = self.enc._frames(frames)
        k = int(frames.shape[0])
        idx = [int(i) for i in idx]
        total = max(idx) + 1
        x = self.proc(text=[PLACEHOLDER], videos=[frames], do_sample_frames=False, return_tensors="pt",
                      video_metadata=[VideoMetadata(total_num_frames=total, fps=float(fps),
                                                    duration=total / float(fps), frames_indices=idx)])
        ids = x["input_ids"][0].to(self.dev)
        grid = x["video_grid_thw"].to(self.dev)
        with torch.inference_mode():
            feats = self.core.get_video_features(x["pixel_values_videos"].to(self.dev), grid).pooler_output
            feats = torch.cat(list(feats), dim=0)
            emb = self.core.get_input_embeddings()(ids[None]).clone()
            emb[0, ids == self.video_id] = feats.to(emb.dtype)
            pos = self._mixed_positions(ids, grid)
            h = self._feed(emb, pos)
            per, L = int(ids.numel()) // k, int(ids.numel())
            if per * k != L:
                raise RuntimeError(f"{L} tokens for {k} frames: the processor's frame layout is not uniform")
            for j in range(k):
                self._blocks.append((per, pos[:, 0, j * per:(j + 1) * per].cpu()))
            self._atoms.append((self._tok_total, self._tok_total + L, int(self.pos)))
            self._tok_total += L
            self._evict(keep=k)
            halves = []
            for which in self.states:
                if which == "memory":
                    m = h[0, -1].float()
                    halves.append((m / m.norm()).cpu().numpy())
                else:
                    halves.append(self._probe())
            return np.concatenate(halves) / np.sqrt(len(halves)) if len(halves) > 1 else halves[0]

    def _mixed_positions(self, ids, grid):
        """Text tokens count 1-D; each frame's patches take the model's own 3-D grid positions
        (`get_vision_position_ids`, as `get_rope_index` lays a video out frame by frame)."""
        import torch
        h, w = int(grid[0, 1]), int(grid[0, 2])
        is_video = (ids == self.video_id).tolist()
        pieces, i, n = [], 0, len(is_video)
        while i < n:
            j = i
            while j < n and is_video[j] == is_video[i]:
                j += 1
            if is_video[i]:
                p = self.core.get_vision_position_ids(self.pos, torch.tensor([1, h, w]), 1, self.merge, 1, device=self.dev)
                pieces.append(p.reshape(3, -1))
                self.pos += max(h, w) // self.merge
            else:
                pieces.append(torch.arange(self.pos, self.pos + (j - i), device=self.dev).view(1, -1).expand(3, -1))
                self.pos += j - i
            i = j
        return torch.cat(pieces, dim=1).unsqueeze(1)          # (3, 1, L)

    def _evict(self, keep: int) -> None:
        """Drop the oldest frames after the sink until the frames fit the window; the atom just
        pushed (`keep` frames) always stays."""
        import torch
        if self.window is None:
            return
        drop = 0
        while len(self._blocks) > keep and sum(t for t, _ in self._blocks) > self.window:
            drop += self._blocks.popleft()[0]
        if not drop:
            return
        self._dropped += drop
        sink = len(self.prefix_ids)
        for layer in self.cache.layers:
            layer.keys = torch.cat([layer.keys[..., :sink, :], layer.keys[..., sink + drop:, :]], dim=-2)
            layer.values = torch.cat([layer.values[..., :sink, :], layer.values[..., sink + drop:, :]], dim=-2)

    # ---- the event probe ------------------------------------------------------------------
    def atom_end(self, j: int) -> int:
        """The 1-D position right after atom j's tokens (where a probe over a span ending at j goes)."""
        return int(self._atoms[j][2])

    def probe_span(self, a: int, b: int, top_k: int = 0):
        """The pin's question over atoms [a, b) of the memory: a view of the memory holding the sink and
        those atoms' tokens, the probe right after atom b-1. -> the unit answer state, or with `top_k`
        (state, ids, log-probs) of the answer's top-k words. The memory is not touched."""
        import torch
        from transformers import DynamicCache
        if not (0 <= a < b <= len(self._atoms)):
            raise ValueError(f"atoms [{a}, {b}) of {len(self._atoms)} pushed")
        if self._atoms[a][0] < self._dropped:
            raise ValueError(f"atom {a} was evicted from the memory (window {self.window})")
        sink = len(self.prefix_ids)
        s, e = sink + self._atoms[a][0] - self._dropped, sink + self._atoms[b - 1][1] - self._dropped
        view = DynamicCache(config=self.lm.config)
        for src, dst in zip(self.cache.layers, view.layers):
            dst.lazy_initialization(src.keys, src.values)
            dst.keys = torch.cat([src.keys[..., :sink, :], src.keys[..., s:e, :]], dim=-2)
            dst.values = torch.cat([src.values[..., :sink, :], src.values[..., s:e, :]], dim=-2)
        n = len(self.probe_ids)
        p0 = self.atom_end(b - 1)
        ids = torch.tensor([self.probe_ids], device=self.dev)
        pos = torch.arange(p0, p0 + n, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        with torch.inference_mode():
            out = self.lm(inputs_embeds=self.core.get_input_embeddings()(ids), position_ids=pos, past_key_values=view,
                          use_cache=True)
            h = out.last_hidden_state[0, -1]
            state = (h.float() / h.float().norm()).cpu().numpy()
            if not top_k:
                return state
            logits = self.enc.model.lm_head(h[None]).float()[0]
            lp = torch.log_softmax(logits, dim=-1)
            top = torch.topk(lp, int(top_k))
            return state, top.indices.cpu().numpy().astype(np.int64), top.values.cpu().numpy().astype(np.float32)

    def _probe(self) -> np.ndarray:
        """The question on the memory, the state at the answer slot, the question gone again."""
        import torch
        n = len(self.probe_ids)
        ids = torch.tensor([self.probe_ids], device=self.dev)
        pos = torch.arange(self.pos, self.pos + n, device=self.dev).view(1, 1, -1).expand(3, 1, -1)   # not kept
        h = self._feed(self.core.get_input_embeddings()(ids), pos)[0, -1].float()
        for layer in self.cache.layers:
            layer.crop(-n)                                # negative: remove that many from the end
        return (h / h.norm()).cpu().numpy()
