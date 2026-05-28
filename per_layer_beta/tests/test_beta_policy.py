import unittest

from beta_policy import ParamInfo, assign_shape_betas, beta_summary, shape_based_beta


class ShapeBetaPolicyTest(unittest.TestCase):
    def test_non_matrix_is_not_muon_eligible(self):
        assignment = shape_based_beta(ParamInfo("transformer.ln.weight", (768,)))
        self.assertIsNone(assignment.beta)
        self.assertEqual(assignment.reason, "not_matrix")

    def test_embeddings_are_excluded_from_muon_beta(self):
        assignment = shape_based_beta(ParamInfo("transformer.wte.weight", (50257, 768)))
        self.assertIsNone(assignment.beta)
        self.assertEqual(assignment.reason, "adamw_embedding_or_head")

    def test_small_matrix_gets_responsive_beta(self):
        assignment = shape_based_beta(ParamInfo("attn.small_proj.weight", (32, 768)))
        self.assertEqual(assignment.beta, 0.90)
        self.assertEqual(assignment.reason, "small_matrix")

    def test_gate_weight_is_excluded_from_muon_beta(self):
        assignment = shape_based_beta(ParamInfo("blocks.0.router.gate.weight", (8, 768)))
        self.assertIsNone(assignment.beta)
        self.assertEqual(assignment.reason, "adamw_gate")

    def test_high_aspect_matrix_gets_conservative_beta(self):
        assignment = shape_based_beta(ParamInfo("mlp.up_proj.weight", (4096, 768)))
        self.assertEqual(assignment.beta, 0.97)
        self.assertEqual(assignment.reason, "high_aspect_matrix")

    def test_summary_counts_reasons(self):
        assignments = assign_shape_betas([
            ParamInfo("transformer.wte.weight", (50257, 768)),
            ParamInfo("block.attn.weight", (768, 768)),
            ParamInfo("block.ln.weight", (768,)),
        ])
        summary = beta_summary(assignments)
        self.assertEqual(summary["n_params"], 3)
        self.assertEqual(summary["n_muon_params"], 1)
        self.assertEqual(summary["unique_betas"], [0.95])
        self.assertEqual(summary["by_reason"]["not_matrix"], 1)


if __name__ == "__main__":
    unittest.main()
