"""Configuration defaults for TriadFuse"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttackConfig:
    """Main attack configuration"""

    # Optimization
    steps: int = 200
    lr: float = 5e-2
    batch_size: int = 1

    # Perturbation bounds
    eps_linf: float = 2 / 255
    ssim_threshold: float = 0.95
    lpips_weight: float = 0.0

    # EOT settings
    eot_samples: int = 4
    jpeg_quality: tuple[int, int] = (40, 85)
    resize_range: tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.5
    gamma_range: tuple[float, float] = (0.9, 1.1)

    # Texture head
    texture_scale: float = 0.01
    texture_lowres: int = 64

    # Model settings
    max_img_size: int = 896
    device: str | None = None

    # Random seed
    seed: int = 1337


@dataclass
class DonutConfig:
    """Donut model configuration"""

    model_name: str = "naver-clova-ix/donut-base"
    max_length: int = 64
    device: str | None = None
    use_8bit: bool = False


# TODO: Add configurations for Glyph and Layout heads in v2
@dataclass
class GlyphConfig:
    """Glyph head configuration (TODO for v2)"""

    enabled: bool = False
    confusable_path: str = "data/confusables.json"
    max_replacements: int = 10
    gumbel_tau: float = 0.5


@dataclass
class LayoutConfig:
    """Layout head configuration (TODO for v2)"""

    enabled: bool = False
    tps_control_points: int = 16
    stamp_area_budget: float = 0.002
    warp_strength: float = 0.01
