"""Print MVP beta assignments for representative parameter names.

This is a local sanity script for the policy module. Replace SAMPLE_PARAMS with
real modded-nanoGPT named parameters once that tree is present.
"""

from beta_policy import ParamInfo, assign_shape_betas, beta_summary


SAMPLE_PARAMS = [
    ParamInfo("transformer.wte.weight", (50257, 768)),
    ParamInfo("blocks.0.attn.q_proj.weight", (768, 768)),
    ParamInfo("blocks.0.attn.k_proj.weight", (768, 768)),
    ParamInfo("blocks.0.mlp.up_proj.weight", (3072, 768)),
    ParamInfo("blocks.0.mlp.down_proj.weight", (768, 3072)),
    ParamInfo("blocks.0.router.gate.weight", (8, 768)),
    ParamInfo("blocks.0.norm.weight", (768,)),
    ParamInfo("lm_head.weight", (50257, 768)),
]


def main() -> None:
    assignments = assign_shape_betas(SAMPLE_PARAMS)
    for item in assignments:
        beta = "AdamW" if item.beta is None else f"{item.beta:.2f}"
        print(f"{item.name:<34} shape={item.shape!s:<16} beta={beta:<6} reason={item.reason}")
    print()
    print(beta_summary(assignments))


if __name__ == "__main__":
    main()
