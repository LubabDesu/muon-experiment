# Experiment Plan

## Claim Under Test

Per-layer Muon beta improves training efficiency by tuning directional smoothing per hidden/projection matrix.

This project does not claim that beta changes update magnitude. Source review supports treating Muon's orthogonalization step as scale-insensitive: the pre-orthogonalization matrix is normalized, and the orthogonalized target is unchanged by scalar rescaling. The experiment therefore treats beta as a direction-memory parameter.

## Runs

Run all configurations with the same model, data order, token budget, validation cadence, and FLOP accounting.

| ID | Optimizer | Changed variable | Purpose |
| --- | --- | --- | --- |
| `adamw` | AdamW | none | Strong baseline |
| `muon_global_beta_095` | Muon | global beta `0.95` | Source-aligned Muon baseline |
| `muon_shape_beta` | Muon | static per-parameter beta | MVP hypothesis test |
| `muon_gradnorm_beta` | Muon | adaptive beta | v2 only after MVP signal |

## Controls

- Keep AdamW-managed parameters identical across Muon runs.
- Keep Muon learning-rate multipliers and update-scale rules identical across `muon_global_beta_095` and `muon_shape_beta`.
- Do not move embeddings or heads into Muon for the MVP.
- Use the same seed and data shard order for all runs.
- Log enough metadata to reproduce the parameter-to-beta table.

## Primary Output

Plot validation loss versus estimated training FLOPs:

- x-axis: cumulative training FLOPs
- y-axis: validation loss
- curves: `adamw`, `muon_global_beta_095`, `muon_shape_beta`

This mirrors the Kimi/Moonlight paper's Figure 1a comparison style at small scale.

## Stop Rule for First Result

Email Rose only after the three MVP curves exist and the plot includes:

- validation loss/FLOPs curves
- final validation loss table
- notes on whether `muon_shape_beta` changed wall-clock overhead
- beta assignment summary

## Risks

- A single L4 may make 1B tokens slow enough that the first credible result needs a smaller pilot token budget.
- The shape beta values are hypotheses, not literature constants.
- If the AdamW baseline is undertuned, the comparison will not meet the evidence standard argued for in the Muon writeup.
