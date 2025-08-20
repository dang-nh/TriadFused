"""Smoke test for attack training"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.utils as vutils


def create_dummy_image(path: Path, size: tuple = (224, 224)):
    """Create a dummy test image"""
    img = torch.rand(1, 3, *size)
    vutils.save_image(img, path, normalize=False)


def test_attack_train_smoke():
    """Test that attack_train.py runs without crashing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy input image
        input_path = tmpdir / "test_input.png"
        create_dummy_image(input_path)

        # Run attack_train.py with minimal steps
        script_path = Path(__file__).parent.parent / "scripts" / "attack_train.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(tmpdir / "output"),
            "--steps",
            "5",  # Just 5 steps for smoke test
            "--eot_samples",
            "2",  # Reduce EOT samples for speed
            "--max_img_size",
            "224",  # Small size for speed
        ]

        # Run the script
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60  # 60 second timeout
        )

        # Check it didn't crash
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            pytest.fail(f"Script failed with return code {result.returncode}")

        # Check output files were created
        output_dir = tmpdir / "output"
        assert output_dir.exists()
        assert (output_dir / "adversarial.png").exists()
        assert (output_dir / "clean.png").exists()
        assert (output_dir / "results.json").exists()


def test_texture_head():
    """Test texture head basic functionality"""
    from triadfuse.heads.texture import TextureHead

    head = TextureHead(img_hw=(224, 224), scale=0.01, lowres=32)
    x = torch.rand(1, 3, 224, 224)
    y = head(x)

    assert y.shape == x.shape
    assert y.min() >= 0
    assert y.max() <= 1

    # Should be close to original (small perturbation)
    diff = (y - x).abs().max()
    assert diff < 0.05  # Max perturbation should be small


def test_constraints():
    """Test constraint functions"""
    from triadfuse.constraints import project_linf, ssim_ok

    x_clean = torch.rand(1, 3, 128, 128)
    x_adv = x_clean + 0.1 * torch.randn_like(x_clean)

    # Test L∞ projection
    eps = 0.03
    x_proj = project_linf(x_adv, x_clean, eps)
    assert (x_proj - x_clean).abs().max() <= eps + 1e-6

    # Test SSIM check
    assert ssim_ok(x_clean, x_clean, threshold=0.99)


def test_losses():
    """Test loss computation"""
    from triadfuse.losses import composite_loss, total_variation

    x_clean = torch.rand(1, 3, 64, 64, requires_grad=True)
    x_adv = x_clean + 0.01 * torch.randn_like(x_clean)
    task_loss = torch.tensor(1.0)

    # Test composite loss
    loss = composite_loss(task_loss, x_adv, x_clean, lam_tv=0.1)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # Test total variation
    tv = total_variation(x_adv - x_clean)
    assert tv >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
