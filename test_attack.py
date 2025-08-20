#!/usr/bin/env python
"""
Simple test script for TriadFuse with memory-efficient settings
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Create a simple test image if test.png doesn't exist
    if not Path("test.png").exists():
        print("Creating test image...")
        subprocess.run([
            sys.executable, "scripts/create_test_image.py", "test.png"
        ], check=True)
    
    print("Running TriadFuse attack with memory-efficient settings...")
    
    # Run with very conservative memory settings
    cmd = [
        sys.executable, "scripts/attack_train.py",
        "--input", "test.png",
        "--target", "9999",
        "--steps", "20",  # Very few steps for testing
        "--eot_samples", "1",  # Minimal EOT samples
        "--max_img_size", "256",  # Very small image size
        "--verbose",
        "--output", "test_output"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Attack completed successfully!")
        print("Check 'test_output/' directory for results")
    except subprocess.CalledProcessError as e:
        print(f"✗ Attack failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
