# Per-Layer Beta Muon

Active research question: does per-layer momentum beta improve Muon training efficiency for LLM pretraining?

Muon commonly uses one global beta, such as `0.95`, for all matrix parameters. This project tests Rose Yu's hypothesis that different layers benefit from different directional smoothing. Because Newton-Schulz normalizes away the momentum buffer scale, beta should be interpreted mainly as direction memory: high beta follows long-term gradient consensus, while low beta responds faster to recent gradients.

## Target Experiment

Train a 117M Llama-architecture model from scratch on FineWeb-Edu for about 1B tokens on a single L4.

Primary comparison:

| Run | Optimizer | Beta policy |
| --- | --- | --- |
| `adamw` | AdamW | none |
| `muon_global_beta` | Muon | global beta `0.95` |
| `muon_shape_beta` | Muon | static shape-based per-parameter beta |

Follow-up:

| Run | Optimizer | Beta policy |
| --- | --- | --- |
| `muon_gradnorm_beta` | Muon | adaptive beta from per-layer grad-norm history |

## Metric

Main plot: validation loss versus training FLOPs, matching the style of Kimi Figure 1a at small scale.

Log at minimum:

- train loss
- validation loss
- estimated training FLOPs
- tokens processed
- wall-clock time
- optimizer name
- beta policy name
- per-parameter beta summary

## Integration Target

The intended codebase is `modded-nanoGPT`. The expected implementation is small:

1. Add a per-parameter beta source.
2. Pass beta into `muon_update` per parameter instead of reading only a group-level beta.
3. Log the beta policy and beta summary with each run.

This folder currently holds the beta-policy nucleus and experiment plan. The actual modded-nanoGPT tree is not present in this workspace yet.

## Source Grounding

See [SOURCES.md](./SOURCES.md). The key correction from source review is that Muon should initially stay restricted to hidden/projection matrices. Keller Jordan's Muon writeup says scalar/vector parameters plus input and output layers should be optimized by AdamW, and current modded-nanoGPT states the same optimizer split for embeddings, scalars, gate weights, and heads.

## MVP Shape Policy

The MVP policy is deliberately simple and deterministic. The exact beta values below are hypotheses for the first experiment, not values copied from the source papers:

- tall/wide matrices with large aspect ratio: beta `0.97`
- square-ish hidden matrices: beta `0.95`
- small matrices: beta `0.90`
- embeddings and output heads: not Muon-managed in the first run, so leave to AdamW
- gate weights, normalization, bias, scalar, and vector parameters: not Muon-eligible in the source-aligned split, so leave to AdamW or fallback optimizer

This is not a claim that these values are optimal. It gives a defensible first run that tests whether directional smoothing heterogeneity moves the loss/FLOP curve.

## Open Decision

The first hard decision is parameter ownership: should the first run exactly mirror modded-nanoGPT's current optimizer split, or simplify it for a single-L4 reproduction? The recommended first L4 run is to mirror the baseline optimizer split as closely as possible, then only vary beta among Muon-managed hidden/projection matrices.
