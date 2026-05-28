import unittest

from mini_pretrain.beta_assign import BankBetaOffsets, beta_for_name, should_use_muon


class TestBankBeta(unittest.TestCase):
    def test_embed_not_muon(self):
        self.assertIsNone(beta_for_name("wte.weight", (50257, 768), "bank", 0.95))
        self.assertFalse(should_use_muon("wpe.weight", (1024, 768)))

    def test_bank_offsets(self):
        base = 0.95
        off = BankBetaOffsets(qk=-0.01, vo=0.0, mlp=0.01)
        self.assertAlmostEqual(
            beta_for_name("blocks.0.attn.q_proj.weight", (768, 768), "bank", base, off),
            0.94,
        )
        self.assertAlmostEqual(
            beta_for_name("blocks.0.attn.v_proj.weight", (768, 768), "bank", base, off),
            0.95,
        )
        self.assertAlmostEqual(
            beta_for_name("blocks.0.mlp.c_fc.weight", (3072, 768), "bank", base, off),
            0.96,
        )

    def test_configurable_delta(self):
        off = BankBetaOffsets(qk=-0.03, vo=0.0, mlp=0.03)
        self.assertAlmostEqual(
            beta_for_name("blocks.0.mlp.c_proj.weight", (768, 3072), "bank", 0.95, off),
            0.98,
        )

    def test_global_uniform(self):
        b = beta_for_name("blocks.0.attn.k_proj.weight", (768, 768), "global", 0.95)
        self.assertAlmostEqual(b, 0.95)
        b2 = beta_for_name("blocks.0.mlp.c_proj.weight", (768, 3072), "global", 0.95)
        self.assertAlmostEqual(b2, 0.95)

    def test_global_vs_bank_differ(self):
        name = "blocks.0.attn.q_proj.weight"
        shape = (768, 768)
        g = beta_for_name(name, shape, "global", 0.95)
        b = beta_for_name(name, shape, "bank", 0.95)
        self.assertNotEqual(g, b)


if __name__ == "__main__":
    unittest.main()
