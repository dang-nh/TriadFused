#!/usr/bin/env python
"""
Simple demonstration of TriadFuse attack

This script:
1. Creates a test document image
2. Runs the attack with minimal steps
3. Shows the results
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image
from rich.console import Console

console = Console()


def main():
    console.print("[bold cyan]TriadFuse Demo[/bold cyan]\n")

    # Step 1: Create test image
    console.print("[yellow]Step 1: Creating test document...[/yellow]")
    test_image = Path("demo_document.png")

    if not test_image.exists():
        subprocess.run(
            [sys.executable, "scripts/create_test_image.py", str(test_image)],
            check=True,
        )
        console.print(f"✓ Test document created: {test_image}")
    else:
        console.print(f"✓ Using existing test document: {test_image}")

    # Step 2: Run attack
    console.print("\n[yellow]Step 2: Running adversarial attack...[/yellow]")
    console.print("Target: Change total amount to '$999.99'")
    console.print("This will take about 30 seconds...\n")

    output_dir = Path("demo_output")
    
    cmd = [
        sys.executable,
        "scripts/attack_train.py",
        "--input", str(test_image),
        "--output", str(output_dir),
        "--target", "$999.99",
        "--prompt", "<s_docvqa><s_question> What is the TOTAL? <s_answer>",
        "--steps", "50",  # Quick demo
        "--eot_samples", "2",  # Fewer samples for speed
        "--max_img_size", "512",  # Smaller for speed
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        console.print(f"[red]Attack failed:[/red]")
        console.print(result.stderr)
        sys.exit(1)

    # Step 3: Show results
    console.print("\n[yellow]Step 3: Results[/yellow]")
    
    if (output_dir / "adversarial.png").exists():
        console.print(f"✓ Adversarial image saved to: {output_dir / 'adversarial.png'}")
        console.print(f"✓ Clean image saved to: {output_dir / 'clean.png'}")
        console.print(f"✓ Difference map saved to: {output_dir / 'difference.png'}")
        console.print(f"✓ Metrics saved to: {output_dir / 'results.json'}")
        
        # Try to load and display metrics
        try:
            import json
            with open(output_dir / "results.json") as f:
                results = json.load(f)
            
            console.print("\n[bold]Attack Results:[/bold]")
            if "predictions" in results:
                console.print(f"  Clean prediction: {results['predictions'].get('clean', 'N/A')}")
                console.print(f"  Adversarial prediction: {results['predictions'].get('adversarial', 'N/A')}")
                console.print(f"  Attack success: {results['predictions'].get('success', False)}")
            
            if "metrics" in results:
                console.print("\n[bold]Quality Metrics:[/bold]")
                console.print(f"  SSIM: {results['metrics'].get('ssim', 0):.4f}")
                console.print(f"  L∞: {results['metrics'].get('l_inf', 0):.4f}")
        except Exception as e:
            console.print(f"[yellow]Could not load results: {e}[/yellow]")
    
    console.print("\n[green]✓ Demo complete![/green]")
    console.print("\nTo view the images, open:")
    console.print(f"  - {output_dir / 'clean.png'}")
    console.print(f"  - {output_dir / 'adversarial.png'}")
    console.print(f"  - {output_dir / 'difference.png'}")


if __name__ == "__main__":
    main()
