"""Tests for EOT module"""

import pytest
import torch

from triadfuse.eot import EOT
from triadfuse.utils.seed import set_seed


def test_eot_initialization():
    """Test EOT can be initialized"""
    eot = EOT(n=2)
    assert eot.n == 2


def test_eot_sample_shape():
    """Test EOT sample preserves shape"""
    set_seed(42)
    eot = EOT(n=2)
    x = torch.rand(1, 3, 256, 256)
    y = eot.sample(x)

    assert y.shape == x.shape
    assert y.min() >= 0
    assert y.max() <= 1


def test_eot_expectation():
    """Test EOT expectation computation"""
    set_seed(42)
    eot = EOT(n=4)
    x = torch.rand(1, 3, 128, 128)

    def dummy_loss(x_t):
        return x_t.mean()

    loss = eot.expectation(x, dummy_loss)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar


def test_eot_deterministic():
    """Test EOT is deterministic with fixed seed"""
    x = torch.rand(1, 3, 64, 64)

    # First run
    set_seed(123)
    eot1 = EOT(n=2, seed=123)
    y1 = eot1.sample(x)

    # Second run with same seed
    set_seed(123)
    eot2 = EOT(n=2, seed=123)
    y2 = eot2.sample(x)

    # Should be identical
    assert torch.allclose(y1, y2, atol=1e-5)


def test_eot_transforms_applied():
    """Test that EOT actually applies transformations"""
    set_seed(42)
    eot = EOT(n=1, jpeg_q=(40, 40))  # Force low quality JPEG
    x = torch.rand(1, 3, 128, 128)
    y = eot.sample(x)

    # Should be different due to transformations
    assert not torch.allclose(x, y, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
