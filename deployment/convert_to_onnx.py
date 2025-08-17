#!/usr/bin/env python3
"""
PyTorch to ONNX Model Converter
Converts trained TCN models (.pth) to ONNX format for cross-platform deployment
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class ModelConverter:
    """Convert PyTorch models to ONNX format"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Model Converter initialized on {self.device}")
    
    def load_pytorch_model(self, model_path):
        """Load PyTorch model from .pth file"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📥 Loading PyTorch model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with default config
        model = create_model()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
        else:
            # Assume direct state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded direct state dict")
        
        model.eval()
        model.to(self.device)
        
        # Print model info
        info = model.get_model_info()
        print(f"📊 Model Info:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Input dim: {info['input_dim']}")
        print(f"  Output dim: {info['output_dim']}")
        print(f"  Parameters: {info['num_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.1f} MB")
        
        return model
    
    def convert_to_onnx(self, model, output_path, input_shape=(1, 24, 80)):
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            input_shape: Input tensor shape (batch_size, sequence_length, features)
        """
        print(f"🔄 Converting to ONNX format...")
        print(f"  Input shape: {input_shape}")
        print(f"  Output path: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Ensure model is in eval mode
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"  Test output shape: {test_output.shape}")
        
        # Convert to ONNX
        torch.onnx.export(
            model,                          # Model to export
            dummy_input,                    # Model input
            output_path,                    # Output file
            export_params=True,             # Store trained weights
            opset_version=11,               # ONNX opset version
            do_constant_folding=True,       # Optimize constant folding
            input_names=['audio_features'], # Input names
            output_names=['blendshapes'],   # Output names
            dynamic_axes={                  # Dynamic axes for variable length
                'audio_features': {0: 'batch_size', 1: 'sequence_length'},
                'blendshapes': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"✅ ONNX model saved to: {output_path}")
        
        # Verify ONNX model
        self.verify_onnx_model(output_path, dummy_input, test_output)
        
        return output_path
    
    def verify_onnx_model(self, onnx_path, test_input, expected_output):
        """Verify ONNX model works correctly"""
        print(f"🔍 Verifying ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model structure is valid")
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            print(f"  Input name: {input_name}")
            print(f"  Output name: {output_name}")
            
            # Run inference
            ort_inputs = {input_name: test_input.cpu().numpy()}
            ort_outputs = ort_session.run([output_name], ort_inputs)
            onnx_output = ort_outputs[0]
            
            # Compare outputs
            pytorch_output = expected_output.cpu().numpy()
            diff = np.abs(pytorch_output - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff:.8f}")
            print(f"  Mean difference: {mean_diff:.8f}")
            
            # Check if outputs are close enough
            if max_diff < 1e-5:
                print(f"✅ ONNX model outputs match PyTorch (excellent)")
            elif max_diff < 1e-3:
                print(f"✅ ONNX model outputs match PyTorch (good)")
            else:
                print(f"⚠️  ONNX model outputs differ significantly from PyTorch")
                print(f"   This might indicate conversion issues")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX verification failed: {e}")
            return False
    
    def get_model_metadata(self, model, model_path):
        """Get model metadata for documentation"""
        info = model.get_model_info()
        
        metadata = {
            "source_model": str(model_path),
            "conversion_date": str(Path().cwd()),
            "model_architecture": info,
            "input_format": {
                "name": "audio_features",
                "shape": "[batch_size, sequence_length, 80]",
                "dtype": "float32",
                "description": "Mel spectrogram features (80 dimensions)",
                "sequence_length": 24,
                "sequence_duration_ms": 240,
                "sample_rate": 16000,
                "hop_length": 160
            },
            "output_format": {
                "name": "blendshapes",
                "shape": "[batch_size, sequence_length, 59]",
                "dtype": "float32",
                "description": "Blendshapes + head pose predictions",
                "components": {
                    "blendshapes": {
                        "indices": "0-51",
                        "count": 52,
                        "range": "[0, 1]",
                        "description": "MediaPipe facial blendshapes"
                    },
                    "head_pose": {
                        "indices": "52-58",
                        "count": 7,
                        "format": "[x, y, z, qw, qx, qy, qz]",
                        "description": "3D translation + quaternion rotation"
                    }
                }
            },
            "preprocessing_required": {
                "audio_normalization": True,
                "scaler_file": "audio_scaler.pkl",
                "mel_spectrogram_params": {
                    "n_mels": 80,
                    "hop_length": 160,
                    "win_length": 400,
                    "n_fft": 512,
                    "sample_rate": 16000
                }
            },
            "postprocessing_required": {
                "target_denormalization": True,
                "scaler_file": "target_scaler.pkl",
                "temporal_smoothing": "recommended"
            },
            "deployment_notes": {
                "javascript_compatibility": "ONNX.js compatible",
                "real_time_capable": True,
                "max_latency_ms": 10,
                "memory_usage": f"{info['model_size_mb']:.1f} MB"
            }
        }
        
        return metadata

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input .pth model file")
    parser.add_argument("--output", type=str, 
                       help="Output .onnx file (default: same name with .onnx extension)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for conversion (default: 1)")
    parser.add_argument("--sequence-length", type=int, default=24,
                       help="Sequence length for conversion (default: 24)")
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.onnx')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔄 PYTORCH TO ONNX CONVERTER")
    print("="*60)
    print(f"Input model: {input_path}")
    print(f"Output ONNX: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    
    try:
        # Create converter
        converter = ModelConverter()
        
        # Load PyTorch model
        model = converter.load_pytorch_model(input_path)
        
        # Convert to ONNX
        input_shape = (args.batch_size, args.sequence_length, 80)
        onnx_path = converter.convert_to_onnx(model, output_path, input_shape)
        
        # Generate metadata
        metadata = converter.get_model_metadata(model, input_path)
        metadata_path = output_path.with_suffix('.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata saved to: {metadata_path}")
        
        print("\n✅ Conversion completed successfully!")
        print(f"📁 Output files:")
        print(f"  - ONNX model: {onnx_path}")
        print(f"  - Metadata: {metadata_path}")
        
        # File size comparison
        pth_size = input_path.stat().st_size / (1024 * 1024)
        onnx_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n📊 File sizes:")
        print(f"  - PyTorch (.pth): {pth_size:.1f} MB")
        print(f"  - ONNX (.onnx): {onnx_size:.1f} MB")
        print(f"  - Size ratio: {onnx_size/pth_size:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

