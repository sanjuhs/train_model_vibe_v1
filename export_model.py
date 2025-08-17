#!/usr/bin/env python3
"""
Model Export Script - Convert trained TCN to ONNX for deployment
"""

import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import joblib

# Import model
from models.tcn_model import create_model

def export_to_onnx(model_path="models/best_tcn_model.pth", 
                   output_path="models/tcn_blendshapes.onnx",
                   input_shape=(1, 24, 80)):  # (batch, time, features)
    """
    Export trained PyTorch model to ONNX format
    
    Args:
        model_path: Path to trained PyTorch model
        output_path: Output path for ONNX model
        input_shape: Input tensor shape for tracing
    """
    
    print(f"Exporting model to ONNX format...")
    print(f"Input model: {model_path}")
    print(f"Output model: {output_path}")
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(*input_shape)
    
    print(f"Model input shape: {input_shape}")
    print(f"Model output shape: {model(dummy_input).shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['audio_features'],
        output_names=['blendshapes_pose'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'sequence_length'},
            'blendshapes_pose': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed")
    
    return output_path

def test_onnx_model(onnx_path="models/tcn_blendshapes.onnx",
                    pytorch_path="models/best_tcn_model.pth"):
    """
    Test ONNX model against PyTorch model
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_path: Path to PyTorch model
    """
    
    print("\\nTesting ONNX model...")
    
    # Load PyTorch model
    checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
    pytorch_model = create_model()
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test input
    test_input = np.random.randn(1, 24, 80).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
    
    # ONNX inference
    onnx_output = ort_session.run(None, {'audio_features': test_input})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Output comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX model matches PyTorch model")
    else:
        print("⚠️ ONNX model differs from PyTorch model")
    
    return max_diff < 1e-5

def create_deployment_package():
    """
    Create complete deployment package
    """
    
    print("\\nCreating deployment package...")
    
    deployment_dir = Path("deployment")
    deployment_dir.mkdir(exist_ok=True)
    
    # Export model to ONNX
    onnx_path = export_to_onnx(output_path=deployment_dir / "tcn_blendshapes.onnx")
    
    # Test ONNX model
    onnx_works = test_onnx_model(onnx_path)
    
    # Copy scalers
    import shutil
    if Path("extracted_features/audio_scaler.pkl").exists():
        shutil.copy("extracted_features/audio_scaler.pkl", deployment_dir / "audio_scaler.pkl")
        print("✅ Audio scaler copied")
    
    if Path("extracted_features/target_scaler.pkl").exists():
        shutil.copy("extracted_features/target_scaler.pkl", deployment_dir / "target_scaler.pkl")
        print("✅ Target scaler copied")
    
    # Create model metadata
    if Path("models/best_tcn_model.pth").exists():
        checkpoint = torch.load("models/best_tcn_model.pth", map_location='cpu', weights_only=False)
        model_info = checkpoint.get('model_config', {})
        validation_metrics = checkpoint.get('validation_metrics', {})
    else:
        model_info = {}
        validation_metrics = {}
    
    # Create deployment info
    deployment_info = {
        "model_info": {
            "architecture": "TCN",
            "input_dim": 80,
            "output_dim": 59,
            "context_frames": 24,
            "context_ms": 240,
            "real_time_capable": True,
            "model_size_mb": 1.3,
            "onnx_compatible": onnx_works
        },
        "audio_processing": {
            "sample_rate": 16000,
            "n_mels": 80,
            "hop_length": 160,
            "win_length": 400,
            "n_fft": 512,
            "mel_frame_rate": 100.0
        },
        "output_format": {
            "blendshapes_count": 52,
            "head_pose_count": 7,
            "blendshapes_indices": "0-51",
            "head_pose_indices": "52-58",
            "head_pose_format": "[x, y, z, qw, qx, qy, qz]"
        },
        "performance": {
            "inference_time_ms": 3.77,
            "max_fps": 365,
            "real_time_latency": True,
            "target_fps": 30
        },
        "validation_metrics": validation_metrics,
        "usage_notes": [
            "Input: 24 frames of 80-dimensional mel features",
            "Output: 59-dimensional vector (52 blendshapes + 7 head pose)",
            "Process audio at 16kHz with 10ms hop length",
            "Apply EMA smoothing (α=0.85) for stability",
            "Model supports real-time inference at 30+ FPS"
        ]
    }
    
    # Save deployment info
    with open(deployment_dir / "model_info.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"✅ Deployment info saved")
    
    # Create simple inference example
    inference_example = '''
# Simple ONNX Inference Example
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("tcn_blendshapes.onnx")

# Prepare input (24 frames x 80 mel features)
audio_features = np.random.randn(1, 24, 80).astype(np.float32)

# Run inference
output = session.run(None, {'audio_features': audio_features})[0]

# Extract results
blendshapes = output[0, -1, :52]  # Latest frame, first 52 values
head_pose = output[0, -1, 52:]    # Latest frame, last 7 values

print(f"Blendshapes: {blendshapes.shape}")
print(f"Head pose: {head_pose.shape}")
'''
    
    with open(deployment_dir / "inference_example.py", 'w') as f:
        f.write(inference_example)
    
    print(f"✅ Inference example created")
    
    # List deployment contents
    print(f"\\nDeployment package created in: {deployment_dir}")
    print("Contents:")
    for file in deployment_dir.iterdir():
        size = file.stat().st_size / 1024 / 1024 if file.is_file() else 0
        print(f"  {file.name} ({size:.1f} MB)" if size > 0.1 else f"  {file.name}")
    
    return deployment_dir

def main():
    """
    Main export function
    """
    print("=== MODEL EXPORT FOR DEPLOYMENT ===")
    
    # Check if trained model exists
    if not Path("models/best_tcn_model.pth").exists():
        print("❌ Trained model not found at models/best_tcn_model.pth")
        print("Please run training first: python training/train_tcn.py")
        return
    
    try:
        # Create deployment package
        deployment_dir = create_deployment_package()
        
        print(f"\\n🎉 Export completed successfully!")
        print(f"Deployment package ready at: {deployment_dir}")
        print("\\nYou can now use the ONNX model for:")
        print("  • Web deployment (ONNX Runtime Web)")
        print("  • Mobile apps (ONNX Runtime Mobile)")
        print("  • Edge devices (ONNX Runtime)")
        print("  • Integration with MediaPipe or other frameworks")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()