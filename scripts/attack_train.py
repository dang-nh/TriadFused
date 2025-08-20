#!/usr/bin/env python
"""
TriadFuse attack training script

Optimizes adversarial perturbations using texture head + EOT
on document images with surrogate models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from rich.console import Console
from rich.progress import track

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from triadfuse.constraints import (
    check_constraints,
    project_all_constraints,
    ssim_ok,
)
from triadfuse.eot import EOT
from triadfuse.heads.texture import TextureHead
from triadfuse.losses import composite_loss
from triadfuse.surrogate.donut import DonutSurrogate
from triadfuse.utils.seed import set_seed

console = Console()


def load_image(path: Path | str, size: int = 896) -> torch.Tensor:
    """Load and preprocess image"""
    img = Image.open(path).convert("RGB")

    # Resize to manageable size (keep aspect ratio)
    w, h = img.size
    if max(w, h) > size:
        if w > h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Ensure dimensions are reasonable for memory
    w, h = img.size
    if w * h > 1024 * 1024:  # Limit to ~1MP for memory safety
        scale = (1024 * 1024 / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to tensor
    transform = T.ToTensor()
    return transform(img).unsqueeze(0)


def save_results(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    output_dir: Path,
    metrics: dict,
    predictions: dict,
):
    """Save attack results and metrics"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    vutils.save_image(x_clean, output_dir / "clean.png", normalize=False)
    vutils.save_image(x_adv, output_dir / "adversarial.png", normalize=False)

    # Save difference map
    diff = (x_adv - x_clean).abs()
    diff_scaled = diff / diff.max() if diff.max() > 0 else diff
    vutils.save_image(diff_scaled, output_dir / "difference.png", normalize=False)

    # Save metrics and predictions
    results = {
        "metrics": metrics,
        "predictions": predictions,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="TriadFuse adversarial attack training"
    )

    # Input/output
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--output", type=str, default="outputs", help="Output directory"
    )

    # Attack parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="<s_cord-v2><s_menu><s_nm>",  # Better Donut format for general document extraction
        help="Task prompt for the model",
    )
    parser.add_argument(
        "--target", type=str, default="9999", help="Target answer to induce"
    )

    # Optimization
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of optimization steps"
    )
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer type",
    )

    # Constraints
    parser.add_argument(
        "--eps", type=float, default=2 / 255, help="Maximum L∞ perturbation"
    )
    parser.add_argument(
        "--ssim_min", type=float, default=0.95, help="Minimum SSIM constraint"
    )
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=0.0,
        help="Weight for LPIPS perceptual loss",
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        default=0.0,
        help="Weight for total variation regularization",
    )

    # EOT parameters
    parser.add_argument(
        "--eot_samples", type=int, default=4, help="Number of EOT samples"
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        nargs=2,
        default=[40, 85],
        help="JPEG quality range",
    )
    parser.add_argument(
        "--resize_range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        help="Random resize range",
    )

    # Texture head parameters
    parser.add_argument(
        "--texture_scale",
        type=float,
        default=0.01,
        help="Maximum texture perturbation",
    )
    parser.add_argument(
        "--texture_lowres",
        type=int,
        default=64,
        help="Low-resolution grid size",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="naver-clova-ix/donut-base",
        help="Surrogate model name",
    )
    parser.add_argument(
        "--max_img_size",
        type=int,
        default=512,  # Reduced default for memory safety
        help="Maximum image dimension (use smaller values to reduce memory usage)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Clear GPU memory if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()

    console.print(f"[bold cyan]TriadFuse Attack Training[/bold cyan]")
    console.print(f"Device: {device}")
    if device.type == "cuda":
        console.print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    console.print(f"Input: {args.input}")
    console.print(f"Target: '{args.target}'")
    console.print()

    # Load image
    console.print("[yellow]Loading image...[/yellow]")
    x_clean = load_image(args.input, args.max_img_size).to(device)
    h, w = x_clean.shape[-2:]
    console.print(f"Image size: {w}x{h}")

    # Initialize components
    console.print("[yellow]Initializing models...[/yellow]")

    # Texture head
    texture_head = TextureHead(
        img_hw=(h, w),
        scale=args.texture_scale,
        lowres=args.texture_lowres,
    ).to(device)

    # EOT sampler
    eot = EOT(
        n=args.eot_samples,
        jpeg_q=tuple(args.jpeg_quality),
        resize=tuple(args.resize_range),
        blur_p=0.5,
        seed=args.seed,
    )

    # Surrogate model
    model = DonutSurrogate(
        model_name=args.model,
        device=device,
        max_length=64,
        use_8bit=True if device.type == "cuda" else False,  # Use 8-bit on GPU to save memory
    )

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(texture_head.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(
            texture_head.parameters(), lr=args.lr, momentum=0.9
        )

    # Get baseline prediction
    console.print("[yellow]Getting baseline prediction...[/yellow]")
    with torch.no_grad():
        clean_pred = model.predict(x_clean, args.prompt)
        console.print(f"Clean prediction: [green]{clean_pred}[/green]")

    # Training loop
    console.print(f"\n[yellow]Starting optimization ({args.steps} steps)...[/yellow]")

    loss_history = []
    best_loss = float("inf")
    best_x_adv = None

    for step in track(range(args.steps), description="Optimizing..."):
        # Define loss function for EOT
        def loss_fn(x_transformed):
            # Apply texture perturbation
            x_perturbed = texture_head(x_transformed)

            # Compute task loss
            task_loss, _ = model.forward_task_loss(
                x_perturbed, args.prompt, args.target
            )

            # Compute composite loss
            total_loss = composite_loss(
                task_loss,
                x_perturbed,
                x_transformed,
                lam_lpips=args.lpips_weight,
                lam_tv=args.tv_weight,
            )

            return total_loss

        # Compute expectation over transformations
        optimizer.zero_grad()
        loss = eot.expectation(x_clean, loss_fn)
        loss.backward()
        optimizer.step()

        # Project to constraints
        with torch.no_grad():
            x_adv = texture_head(x_clean)
            x_adv_proj = project_all_constraints(
                x_adv, x_clean, eps_linf=args.eps, ssim_min=args.ssim_min
            )

            # Update texture head parameters if projection changed things significantly
            if (x_adv_proj - x_adv).abs().max() > 1e-4:
                # Scale down parameters
                texture_head.param.data *= 0.95

        # Track best result
        loss_val = float(loss)
        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_x_adv = x_adv_proj.clone()

        # Periodic logging
        if args.verbose and step % 20 == 0:
            mem_info = ""
            if device.type == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_cached = torch.cuda.memory_reserved() / 1024**3
                mem_info = f" | GPU: {mem_used:.1f}GB/{mem_cached:.1f}GB"
            console.print(f"Step {step}: loss = {loss_val:.4f}{mem_info}")
            
        # Clear cache periodically
        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.empty_cache()

    # Final evaluation
    console.print("\n[yellow]Evaluating attack...[/yellow]")

    x_adv_final = best_x_adv if best_x_adv is not None else texture_head(x_clean)

    with torch.no_grad():
        adv_pred = model.predict(x_adv_final, args.prompt)
        console.print(f"Adversarial prediction: [red]{adv_pred}[/red]")

        # Check constraints
        constraints = check_constraints(
            x_adv_final,
            x_clean,
            eps_linf=args.eps,
            ssim_min=args.ssim_min,
        )

        console.print("\n[bold]Constraint satisfaction:[/bold]")
        for name, satisfied in constraints.items():
            status = "✓" if satisfied else "✗"
            color = "green" if satisfied else "red"
            console.print(f"  [{color}]{status}[/{color}] {name}")

    # Compute final metrics
    from triadfuse.constraints import lpips_loss, ssim

    metrics = {
        "loss_final": float(best_loss),
        "loss_history": loss_history[-10:],  # Last 10 for brevity
        "lpips": float(lpips_loss(x_adv_final, x_clean)),
        "ssim": float(ssim(x_adv_final, x_clean)),
        "l_inf": float((x_adv_final - x_clean).abs().max()),
        "l_2": float((x_adv_final - x_clean).view(-1).norm()),
        "constraints_satisfied": all(constraints.values()),
    }

    predictions = {
        "clean": clean_pred,
        "adversarial": adv_pred,
        "success": adv_pred.lower().strip() == args.target.lower().strip(),
    }

    # Save results
    output_dir = Path(args.output)
    console.print(f"\n[yellow]Saving results to {output_dir}...[/yellow]")
    save_results(x_clean, x_adv_final, output_dir, metrics, predictions)

    # Print summary
    console.print("\n[bold cyan]Attack Summary:[/bold cyan]")
    console.print(f"  Clean → Adversarial: '{clean_pred}' → '{adv_pred}'")
    console.print(f"  Success: {predictions['success']}")
    console.print(f"  SSIM: {metrics['ssim']:.4f}")
    console.print(f"  LPIPS: {metrics['lpips']:.4f}")
    console.print(f"  L∞: {metrics['l_inf']:.4f}")

    console.print("\n[green]✓ Attack complete![/green]")


if __name__ == "__main__":
    main()
