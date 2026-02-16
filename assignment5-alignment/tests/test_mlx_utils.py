import unittest
import numpy as np
import torch
import mlx.core as mx

from cs336_alignment.utils import masked_mean as pt_masked_mean
from cs336_alignment.utils import masked_normalize as pt_masked_normalize
from cs336_alignment.mlx_utils import masked_mean as mx_masked_mean
from cs336_alignment.mlx_utils import masked_normalize as mx_masked_normalize


class TestMLXUtils(unittest.TestCase):
    def test_masked_mean(self):
        # Create random data
        np.random.seed(42)
        data = np.random.randn(2, 5).astype(np.float32)
        mask_np = (np.random.rand(2, 5) > 0.5).astype(np.float32)

        # PyTorch
        pt_data = torch.from_numpy(data)
        pt_mask = torch.from_numpy(mask_np)
        pt_res = pt_masked_mean(pt_data, pt_mask, dim=1)

        # MLX
        mx_data = mx.array(data)
        mx_mask = mx.array(mask_np)
        mx_res = mx_masked_mean(mx_data, mx_mask, axis=1)

        # Compare
        np.testing.assert_allclose(pt_res.numpy(), np.array(mx_res), rtol=1e-5)
        print("test_masked_mean PASSED")

    def test_masked_normalize(self):
        # Create random data
        np.random.seed(42)
        data = np.random.randn(2, 5).astype(np.float32)
        mask_np = (np.random.rand(2, 5) > 0.5).astype(np.float32)
        constant = 2.5

        # PyTorch
        pt_data = torch.from_numpy(data)
        pt_mask = torch.from_numpy(mask_np)
        pt_res = pt_masked_normalize(
            pt_data, pt_mask, normalize_constant=constant, dim=1
        )

        # MLX
        mx_data = mx.array(data)
        mx_mask = mx.array(mask_np)
        mx_res = mx_masked_normalize(
            mx_data, mx_mask, normalize_constant=constant, axis=1
        )

        # Compare
        np.testing.assert_allclose(pt_res.numpy(), np.array(mx_res), rtol=1e-5)
        print("test_masked_normalize PASSED")


if __name__ == "__main__":
    unittest.main()
