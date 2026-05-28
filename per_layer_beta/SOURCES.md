# Source Grounding

These are the implementation facts this scaffold is currently grounded in.

## Muon Definition and Optimizer Split

Source: [Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"](https://kellerjordan.github.io/posts/muon/)

Relevant facts:

- Muon is defined for 2D hidden-layer parameters.
- It takes SGD-momentum updates and applies Newton-Schulz orthogonalization before applying them.
- Scalar/vector parameters and input/output layers should be optimized with a standard optimizer such as AdamW.
- The writeup says AdamW should be used for transformer embeddings and final classifier heads for best performance.
- The update is normalized before Newton-Schulz, so rescaling the pre-orthogonalization matrix does not change the target orthogonalized direction.

Implication for this project:

- The first per-layer beta experiment should vary beta only inside Muon-managed hidden/projection matrices.
- Embeddings, heads, gate weights, scalar parameters, and vector parameters should remain in the AdamW side of the optimizer split unless a later ablation intentionally changes that.

## modded-nanoGPT Integration Shape

Source: [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)

Relevant facts:

- The repository targets NanoGPT/FineWeb speedrunning and uses Muon among its training-speed techniques.
- Current `train_gpt.py` uses per-parameter configuration via a `ParamConfig` table rather than ordinary PyTorch parameter groups.
- Current update code computes Nesterov momentum before orthogonalization, with a per-parameter `momentum` field in the parameter config.
- Current comments say Muon is applied only to attention and MLP projection matrices, not embeddings, scalars, individual weight vectors, bias terms, gate weights, or heads.

Implication for this project:

- The real patch should add or derive a per-parameter Muon momentum value in the param table, not add global optimizer groups.
- In current modded-nanoGPT, the likely minimal change is to let the `ParamConfig.momentum` value differ by parameter label and log that table.

## Kimi / Moonlight Baseline

Source: [Liu et al. 2025, "Muon is Scalable for LLM Training", arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

Relevant facts:

- The paper identifies weight decay and per-parameter update scale adjustment as important for scaling Muon.
- The abstract reports scaling-law experiments where Muon reaches about 2x computational efficiency compared with AdamW under compute-optimal training.
- Figure 1a is a loss-versus-training-FLOPs comparison between Muon and AdamW.

Implication for this project:

- Validation loss versus training FLOPs is the right primary plot.
- Per-layer beta is not claimed by the paper; it is the next hypothesis after their per-parameter scaling work.

## FineWeb-Edu Dataset

Source: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

Relevant facts:

- FineWeb-Edu is a 1.3T-token educational subset filtered from FineWeb.
- It is released on Hugging Face for text-generation training and supports streaming through the `datasets` library.

Implication for this project:

- A 1B-token single-L4 run is a small controlled slice of the dataset, not a full-dataset experiment.
