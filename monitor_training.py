#!/usr/bin/env python3
"""
Monitor Training Progress
Check GPU training progress and display metrics
"""

import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_training_progress():
    """Check current training progress"""
    
    # Check for model files
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ No models directory found")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    checkpoint_files = list(models_dir.glob("checkpoint_*.pth"))
    
    print("🔍 Training Progress Check")
    print("=" * 50)
    
    # Check for best model
    best_model = models_dir / "best_tcn_gpu_model.pth"
    if best_model.exists():
        print(f"✅ Best model found: {best_model}")
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        # Try to get modification time to see how recent
        mod_time = best_model.stat().st_mtime
        print(f"   Last updated: {time.ctime(mod_time)}")
    else:
        print("⏳ Best model not found yet...")
    
    # Check for checkpoints
    if checkpoint_files:
        print(f"\n📁 Found {len(checkpoint_files)} checkpoint(s):")
        for checkpoint in sorted(checkpoint_files):
            print(f"   - {checkpoint.name}")
    
    # Check for training plots
    plot_files = list(Path(".").glob("training_progress_*.png"))
    if plot_files:
        print(f"\n📊 Found {len(plot_files)} training plot(s):")
        for plot in sorted(plot_files):
            print(f"   - {plot.name}")
    
    # Check if training is still running
    print(f"\n🔄 Total model files: {len(model_files)}")
    
def check_gpu_usage():
    """Check GPU usage if nvidia-smi is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"\n🖥️ GPU Status:")
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_util = parts[0].strip()
                        mem_used = parts[1].strip()
                        mem_total = parts[2].strip()
                        print(f"   GPU {i}: {gpu_util}% utilization, {mem_used}/{mem_total} MB memory")
    except:
        print("\n⚠️ Could not check GPU status")

def main():
    """Main monitoring function"""
    print("🚀 Training Monitor")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            check_training_progress()
            check_gpu_usage()
            
            print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            print("Checking again in 30 seconds...\n")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

if __name__ == "__main__":
    main()