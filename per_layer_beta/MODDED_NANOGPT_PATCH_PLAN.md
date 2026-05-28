# modded-nanoGPT Patch Plan

This is the concrete patch target once `KellerJordan/modded-nanogpt` is available locally.

## Source Facts To Preserve

- Current modded-nanoGPT uses a per-parameter `ParamConfig` table rather than conventional PyTorch optimizer groups.
- Muon/NorMuon is applied to attention and MLP projection matrices.
- AdamW handles embeddings, heads, scalars, vectors, and gate weights.
- The current Muon-like path applies Nesterov momentum before orthogonalization.

## Minimal MVP Patch

1. Add a beta policy function near the param-table construction:

```python
def shape_beta_for_label(label: str, shape: tuple[int, ...]) -> float:
    rows, cols = shape[-2], shape[-1]
    min_dim = min(rows, cols)
    aspect = max(rows, cols) / max(1, min_dim)
    if min_dim < 64:
        return 0.90
    if aspect >= 4.0:
        return 0.97
    return 0.95
```

2. When creating `ParamConfig` for Muon-managed projection matrices, set:

```python
momentum=shape_beta_for_label(label, tuple(param.shape))
```

3. Leave AdamW configs unchanged.

4. Log the full `label -> momentum` table with the training code snapshot.

## Why This Is The Right Insertion Point

The raw current `train_gpt.py` already has a `ParamConfig.momentum` field. That means the cleanest per-layer beta implementation is not a new optimizer group. It is a different per-parameter value in the existing table.

## Validation Before Long Runs

Run a tiny local sanity check before spending L4 time:

1. Print all parameter labels, optimizer type, shape, and momentum.
2. Assert all AdamW-managed parameters have `momentum is None`.
3. Assert all Muon-managed parameters have non-null momentum.
4. Assert `muon_global_beta_095` assigns exactly one unique Muon beta.
5. Assert `muon_shape_beta` assigns at least two unique Muon betas on the 117M model.

## Long-Run Commands

These commands are placeholders until the local modded-nanoGPT checkout path and launch script are known:

```bash
python train_gpt.py --optimizer adamw --tokens 1000000000 --run-name adamw
python train_gpt.py --optimizer muon --beta-policy global_095 --tokens 1000000000 --run-name muon_global_beta_095
python train_gpt.py --optimizer muon --beta-policy shape --tokens 1000000000 --run-name muon_shape_beta
```

If the actual code is still shell-script driven, thread `BETA_POLICY=global_095|shape` through `run.sh` into `train_gpt.py`.
