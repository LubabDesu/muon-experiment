import unittest
import sys
import os
from unittest.mock import MagicMock

# Add modded-nanogpt to path
sys.path.append("/Users/lucasyan/Desktop/UCSD/Spring 26/CSE151B/experiments/muon/modded-nanogpt")

# Setup dummy distributed environment variables to prevent initialization crashes

# Setup dummy distributed environment variables to prevent initialization crashes
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

import torch
import torch.nn as nn

class TestTrainGPTPatch(unittest.TestCase):
    def setUp(self):
        # We can dynamically reload/import train_gpt.py
        # To test different env variables, we can modify os.environ and reload,
        # but let's first test the default imports and get the classes.
        import train_gpt
        self.train_gpt = train_gpt

        # Create a MockModel
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.qk_bank = nn.Parameter(torch.zeros(4, 8, 8))
                self.qk_bank.label = "qk_bank"
                self.qk_bank.reshape = (4, 8, 8)

                self.vo_bank = nn.Parameter(torch.zeros(4, 8, 8))
                self.vo_bank.label = "vo_bank"
                self.vo_bank.reshape = (4, 8, 8)

                self.mlp_bank = nn.Parameter(torch.zeros(4, 8, 8))
                self.mlp_bank.label = "mlp_bank"
                self.mlp_bank.reshape = (4, 8, 8)

                self.scalars = nn.Parameter(torch.zeros(2))
                self.scalars.label = "scalars"

                self.smear_gate = nn.Parameter(torch.zeros(2))
                self.smear_gate.label = "smear_gate"

                self.skip_gate = nn.Parameter(torch.zeros(2))
                self.skip_gate.label = "skip_gate"

                self.attn_gate_bank = nn.Parameter(torch.zeros(2))
                self.attn_gate_bank.label = "attn_gate_bank"

                self.ve_gate_bank = nn.Parameter(torch.zeros(2))
                self.ve_gate_bank.label = "ve_gate_bank"

                self.lm_head = nn.Parameter(torch.zeros(2, 2))
                self.lm_head.label = "lm_head"

                self.bigram_embed = nn.Parameter(torch.zeros(2, 2))
                self.bigram_embed.label = "bigram_embed"

                self.post_lambdas = nn.Parameter(torch.zeros(2))
                self.post_lambdas.label = "post_lambdas"

                self.x0_lambdas = nn.Parameter(torch.zeros(2))
                self.x0_lambdas.label = "x0_lambdas"

                self.bigram_lambdas = nn.Parameter(torch.zeros(2))
                self.bigram_lambdas.label = "bigram_lambdas"

                self.resid_lambdas = nn.Parameter(torch.zeros(2))
                self.resid_lambdas.label = "resid_lambdas"

                self.value_embeds = nn.Parameter(torch.zeros(2, 2))
                self.value_embeds.label = "value_embeds"

                self.embed = nn.Parameter(torch.zeros(2, 2))
                self.embed.label = "embed"

                self.yarn = MagicMock()
                self.yarn_paired_head = MagicMock()

        self.model = MockModel()

    def test_shape_beta_policy_values(self):
        # Test shape beta calculation values
        get_shape_beta = self.train_gpt.get_shape_beta_for_matrix
        
        # Test qk_bank uses base_beta
        self.assertEqual(get_shape_beta("qk_bank", 0, (4, 8, 8), 0.95), 0.95)
        
        # Test mlp_bank handles c_fc (even global index) -> base_beta + 0.01
        self.assertEqual(get_shape_beta("mlp_bank", 0, (4, 8, 8), 0.95), 0.96)
        
        # Test mlp_bank handles c_proj (odd global index) -> base_beta + 0.02
        self.assertEqual(get_shape_beta("mlp_bank", 1, (4, 8, 8), 0.95), 0.97)

        # Test clamp limits
        self.assertEqual(get_shape_beta("mlp_bank", 1, (4, 8, 8), 0.97), 0.98)  # Clamped to 0.98
        self.assertEqual(get_shape_beta("mlp_bank", 0, (4, 8, 8), 0.80), 0.85)  # Clamped to 0.85

    def test_optimizer_mode_defaults(self):
        # Verify globals are initialized
        self.assertIn(self.train_gpt.BETA_POLICY, ["global", "shape"])
        self.assertIn(self.train_gpt.OPTIMIZER_MODE, ["muon", "adamw"])

    def test_adamw_baseline_param_table(self):
        # Force OPTIMIZER_MODE to 'adamw' and check ParamConfig creation
        self.train_gpt.OPTIMIZER_MODE = "adamw"
        
        manager = self.train_gpt.TrainingManager(self.model)
        
        # Assert qk_bank, vo_bank, and mlp_bank are configured for adam
        self.assertEqual(manager.param_table["qk_bank"]["optim"], "adam")
        self.assertEqual(manager.param_table["vo_bank"]["optim"], "adam")
        self.assertEqual(manager.param_table["mlp_bank"]["optim"], "adam")
        
        # Check that we run adam on all steps
        self.assertTrue(manager._is_adam_step(0))
        self.assertTrue(manager._is_adam_step(1))

    def test_shape_beta_policy_assignment(self):
        # Setup shape beta policy
        self.train_gpt.OPTIMIZER_MODE = "muon"
        self.train_gpt.BETA_POLICY = "shape"
        
        manager = self.train_gpt.TrainingManager(self.model)
        
        # Call step_optimizers for step 0
        manager.step_optimizers(0)
        
        # Verify that parameters optimized by normuon have tensor momentum
        for param, p_cfg in manager.optimizer.param_cfgs.items():
            if p_cfg.optim == "normuon":
                self.assertIsInstance(p_cfg.momentum, torch.Tensor)
                self.assertEqual(p_cfg.momentum.shape, (p_cfg.chunk_size, 1, 1))
                
                # Check actual values (for mlp_bank: global index 0 in chunk should be base_beta + 0.01)
                # Note: get_muon_momentum(0) returns momentum_min = 0.85
                if p_cfg.label == "mlp_bank":
                    # global index 0 is c_fc -> 0.85 + 0.01 = 0.86
                    self.assertAlmostEqual(p_cfg.momentum[0].item(), 0.86, places=4)
                    # global index 1 is c_proj -> 0.85 + 0.02 = 0.87
                    self.assertAlmostEqual(p_cfg.momentum[1].item(), 0.87, places=4)

if __name__ == "__main__":
    unittest.main()
