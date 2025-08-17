#!/usr/bin/env python3
"""
Test CUDA availability for training
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent))

def test_cuda():
    print("=== CUDA TEST ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test tensor operations
        print("\nTesting tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✅ GPU tensor operation successful! Result shape: {z.shape}")
        print(f"   Device: {z.device}")
        
        return True
    else:
        print("❌ CUDA not available")
        return False

def test_training_imports():
    print("\n=== TRAINING IMPORTS TEST ===")
    try:
        from models.tcn_model import create_model
        print("✅ TCN model import successful")
        
        from training.train_tcn import TCNTrainer
        print("✅ TCN trainer import successful")
        
        # Test model creation
        model = create_model()
        print(f"✅ Model creation successful: {model}")
        
        # Test moving model to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model moved to GPU successfully")
            
            # Test a forward pass
            x = torch.randn(1, 24, 80).cuda()
            with torch.no_grad():
                output = model(x)
            print(f"✅ GPU forward pass successful! Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Import/model error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    cuda_ok = test_cuda()
    training_ok = test_training_imports()
    
    print(f"\n=== RESULTS ===")
    print(f"CUDA: {'✅ OK' if cuda_ok else '❌ FAILED'}")
    print(f"Training: {'✅ OK' if training_ok else '❌ FAILED'}")
    
    if cuda_ok and training_ok:
        print("🚀 Ready for GPU training!")
    else:
        print("⚠️ Issues detected")