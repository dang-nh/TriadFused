#!/usr/bin/env python
"""
Ultra-simple test to isolate the issue
"""

import torch
import sys
from pathlib import Path

# Add to Python path
sys.path.append(str(Path(__file__).parent))

def test_donut():
    """Test just the Donut model loading"""
    from triadfuse.surrogate.donut import DonutSurrogate
    
    print("Testing Donut model loading...")
    
    # First try: CPU only (safest)
    try:
        print("1. Trying CPU...")
        model = DonutSurrogate(device="cpu", use_8bit=False)
        print("✓ CPU loading successful")
        
        # Test basic functionality
        x = torch.rand(1, 3, 256, 256)
        pred = model.predict(x, "<s_cord-v2><s_menu><s_nm>")
        print(f"✓ Prediction: '{pred}'")
        
        del model
        return True
        
    except Exception as e:
        print(f"✗ CPU loading failed: {e}")
        return False

def test_simple_attack():
    """Test minimal attack"""
    from triadfuse.surrogate.donut import DonutSurrogate
    from triadfuse.heads.texture import TextureHead
    
    print("\nTesting simple attack...")
    
    try:
        # Small image
        x = torch.rand(1, 3, 128, 128)
        
        # CPU model
        model = DonutSurrogate(device="cpu", use_8bit=False)
        texture = TextureHead(img_hw=(128, 128), scale=0.01, lowres=16)
        
        # Forward pass
        x_adv = texture(x)
        loss, _ = model.forward_task_loss(x_adv, "<s_cord-v2><s_menu><s_nm>", "test")
        
        print(f"✓ Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple attack failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_donut()
    if success:
        test_simple_attack()
    else:
        print("Basic model loading failed, skipping attack test")
