import unittest

import torch

from muon_beta import muon_update


class MuonBetaTest(unittest.TestCase):
    def test_beta_changes_direction_before_orthogonalization(self):
        grad1 = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
        grad2 = torch.tensor([[0.0, 1.0], [0.5, 0.0]])

        high_buf = torch.zeros_like(grad1)
        low_buf = torch.zeros_like(grad1)
        muon_update(grad1, high_buf, beta=0.95)
        muon_update(grad1, low_buf, beta=0.50)
        high_update = muon_update(grad2, high_buf, beta=0.95)
        low_update = muon_update(grad2, low_buf, beta=0.50)

        cosine = torch.nn.functional.cosine_similarity(
            high_update.flatten(),
            low_update.flatten(),
            dim=0,
        )
        self.assertLess(cosine.item(), 0.999)

    def test_newton_schulz_removes_most_scale_information(self):
        grad = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        small = muon_update(grad, torch.zeros_like(grad), beta=0.95)
        large = muon_update(10.0 * grad, torch.zeros_like(grad), beta=0.95)

        cosine = torch.nn.functional.cosine_similarity(small.flatten(), large.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999)


if __name__ == "__main__":
    unittest.main()
