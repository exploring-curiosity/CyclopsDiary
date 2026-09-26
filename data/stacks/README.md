# Side stacks (not in git; data/ is ignored)

`d40` — transformers >= 5 for nvidia/Cosmos3-Edge (the checkpoint's own architecture class),
which the main environment (transformers 4.57.6, pinned for Cosmos-Embed1's remote code) cannot load.
Used by the encoder service only: `scripts/desk.py --encoder-stack data/stacks/d40` prepends it to that
one child's PYTHONPATH. Rebuild:

    pip install --no-deps --target data/stacks/d40 diffusers==0.40.0 "huggingface_hub>=1.23,<2" "transformers>=5,<6" "tokenizers>=0.23.1,<0.24"

Everything else (torch, numpy, cv2, av, django, ninja, uvicorn) comes from the main environment.
