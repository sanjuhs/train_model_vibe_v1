#!/usr/bin/env python3
"""
PyTorch Model Inference Demo
Demonstrates how to use trained .pth models for audio-to-blendshapes inference
"""

import torch
import numpy as np
import librosa
import joblib
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import time

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class PyTorchInferenceDemo:
    """Demo class for PyTorch model inference"""
    
    def __init__(self, 
                 model_path="models/best_full_model.pth",
                 audio_scaler_path="deployment/audio_scaler.pkl",
                 target_scaler_path="deployment/target_scaler.pkl"):
        """
        Initialize inference demo
        
        Args:
            model_path: Path to .pth model file
            audio_scaler_path: Path to audio feature scaler
            target_scaler_path: Path to target scaler
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 PyTorch Inference Demo on {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Load scalers
        self.audio_scaler = self.load_scaler(audio_scaler_path, "audio")
        self.target_scaler = self.load_scaler(target_scaler_path, "target")
        
        # Audio processing parameters (must match training)
        self.sample_rate = 16000
        self.n_mels = 80
        self.hop_length = 160
        self.win_length = 400
        self.n_fft = 512
        self.context_frames = 24  # 240ms context
        
        # Performance tracking
        self.inference_times = []
        
        print(f"✅ Inference demo initialized")
        print(f"  Context frames: {self.context_frames} ({self.context_frames*10}ms)")
        print(f"  Audio parameters: {self.sample_rate}Hz, {self.n_mels} mels")
    
    def load_model(self, model_path):
        """Load PyTorch model"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📥 Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model
        model = create_model()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
            
            # Print training info if available
            if 'epoch' in checkpoint:
                print(f"  Trained for {checkpoint['epoch']} epochs")
            if 'best_val_loss' in checkpoint:
                print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded direct state dict")
        
        model.eval()
        model.to(self.device)
        
        # Print model info
        info = model.get_model_info()
        print(f"📊 Model Architecture:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return model
    
    def load_scaler(self, scaler_path, scaler_type):
        """Load feature scaler"""
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"✅ Loaded {scaler_type} scaler from: {scaler_path}")
            return scaler
        else:
            print(f"⚠️  {scaler_type} scaler not found: {scaler_path}")
            return None
    
    def load_sample_audio(self, audio_path, duration=3.0):
        """
        Load sample audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load (seconds)
        
        Returns:
            np.ndarray: Audio samples
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"⚠️  Audio file not found: {audio_path}")
            print("🎵 Generating synthetic audio for demo...")
            return self.generate_synthetic_audio(duration)
        
        print(f"🎵 Loading audio from: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
        
        print(f"  Duration: {len(audio)/sr:.2f}s")
        print(f"  Sample rate: {sr}Hz")
        print(f"  Samples: {len(audio):,}")
        
        return audio
    
    def generate_synthetic_audio(self, duration=3.0):
        """Generate synthetic audio for demo"""
        print(f"🎵 Generating {duration}s of synthetic audio...")
        
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Create complex audio with multiple components
        audio = np.zeros_like(t)
        
        # Add fundamental frequencies (speech-like)
        for freq in [110, 220, 330, 440]:  # Harmonics
            audio += 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.5)
        
        # Add some noise and modulation
        audio += 0.1 * np.random.randn(len(t))
        audio *= (1 + 0.3 * np.sin(2 * np.pi * 2 * t))  # AM modulation
        
        # Add envelope
        envelope = np.exp(-t * 0.3) * (1 - np.exp(-t * 5))
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def extract_mel_features(self, audio):
        """
        Extract mel spectrogram features from audio
        
        Args:
            audio: Audio samples
        
        Returns:
            np.ndarray: Mel features (time, n_mels)
        """
        print(f"🎯 Extracting mel features...")
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            fmin=0,
            fmax=self.sample_rate // 2
        )
        
        # Convert to log mel
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, mels)
        mel_features = log_mel_spec.T
        
        print(f"  Mel features shape: {mel_features.shape}")
        print(f"  Time frames: {mel_features.shape[0]} ({mel_features.shape[0]*10}ms)")
        
        return mel_features
    
    def create_context_windows(self, mel_features):
        """
        Create context windows for inference
        
        Args:
            mel_features: Mel features (time, mels)
        
        Returns:
            np.ndarray: Context windows (num_windows, context_frames, mels)
        """
        num_frames = mel_features.shape[0]
        
        if num_frames < self.context_frames:
            print(f"⚠️  Not enough frames ({num_frames}) for context window ({self.context_frames})")
            # Pad with zeros
            padding = np.zeros((self.context_frames - num_frames, self.n_mels))
            mel_features = np.vstack([padding, mel_features])
            num_frames = self.context_frames
        
        # Create sliding windows
        windows = []
        for i in range(num_frames - self.context_frames + 1):
            window = mel_features[i:i + self.context_frames]
            windows.append(window)
        
        context_windows = np.array(windows)
        print(f"  Context windows: {context_windows.shape}")
        
        return context_windows
    
    def predict_blendshapes(self, context_windows):
        """
        Predict blendshapes from context windows
        
        Args:
            context_windows: Context windows (num_windows, context_frames, mels)
        
        Returns:
            tuple: (blendshapes, head_pose, inference_time)
        """
        print(f"🧠 Running inference on {len(context_windows)} windows...")
        
        all_predictions = []
        inference_times = []
        
        with torch.no_grad():
            for i, window in enumerate(context_windows):
                start_time = time.time()
                
                # Normalize input if scaler available
                if self.audio_scaler is not None:
                    window_flat = window.reshape(-1, window.shape[-1])
                    window_normalized = self.audio_scaler.transform(window_flat)
                    window_normalized = window_normalized.reshape(window.shape)
                else:
                    window_normalized = window
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)
                
                # Forward pass
                output_tensor = self.model(input_tensor)
                output = output_tensor.squeeze(0).cpu().numpy()  # (seq_len, 59)
                
                # Get the latest prediction (last time step)
                latest_prediction = output[-1]  # (59,)
                
                # Denormalize if scaler available
                if self.target_scaler is not None:
                    latest_prediction = self.target_scaler.inverse_transform(
                        latest_prediction.reshape(1, -1)
                    )[0]
                
                all_predictions.append(latest_prediction)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(context_windows)} windows")
        
        predictions = np.array(all_predictions)  # (num_windows, 59)
        
        # Split into blendshapes and head pose
        blendshapes = predictions[:, :52]  # (num_windows, 52)
        head_pose = predictions[:, 52:]    # (num_windows, 7)
        
        avg_inference_time = np.mean(inference_times)
        self.inference_times.extend(inference_times)
        
        print(f"✅ Inference completed")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Blendshapes range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
        print(f"  Head pose range: [{head_pose.min():.3f}, {head_pose.max():.3f}]")
        print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms")
        print(f"  FPS capability: {1/avg_inference_time:.1f}")
        
        return blendshapes, head_pose, avg_inference_time
    
    def visualize_results(self, blendshapes, head_pose, save_path=None):
        """
        Visualize prediction results
        
        Args:
            blendshapes: Blendshape predictions (time, 52)
            head_pose: Head pose predictions (time, 7)
            save_path: Optional path to save plot
        """
        print(f"📊 Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PyTorch Model Inference Results', fontsize=16)
        
        time_frames = np.arange(len(blendshapes)) * 10  # Convert to milliseconds
        
        # Plot 1: Key blendshapes
        ax1 = axes[0, 0]
        key_blendshapes = [
            ('jawOpen', 25),  # Approximate index for jaw open
            ('mouthOpen', 19), # Approximate index for mouth open
            ('eyeBlinkLeft', 0),  # Approximate index for left eye blink
            ('eyeBlinkRight', 1), # Approximate index for right eye blink
        ]
        
        for name, idx in key_blendshapes:
            if idx < blendshapes.shape[1]:
                ax1.plot(time_frames, blendshapes[:, idx], label=name, linewidth=2)
        
        ax1.set_title('Key Blendshapes')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Blendshape Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All blendshapes heatmap
        ax2 = axes[0, 1]
        im = ax2.imshow(blendshapes.T, aspect='auto', cmap='viridis', 
                       extent=[0, time_frames[-1], 0, 52])
        ax2.set_title('All Blendshapes (Heatmap)')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Blendshape Index')
        plt.colorbar(im, ax=ax2, label='Value')
        
        # Plot 3: Head pose translation
        ax3 = axes[1, 0]
        pose_labels = ['X', 'Y', 'Z']
        for i, label in enumerate(pose_labels):
            if i < head_pose.shape[1]:
                ax3.plot(time_frames, head_pose[:, i], label=f'Translation {label}', linewidth=2)
        
        ax3.set_title('Head Translation')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Translation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Head pose rotation (quaternion)
        ax4 = axes[1, 1]
        quat_labels = ['qw', 'qx', 'qy', 'qz']
        for i, label in enumerate(quat_labels):
            if i + 3 < head_pose.shape[1]:
                ax4.plot(time_frames, head_pose[:, i + 3], label=label, linewidth=2)
        
        ax4.set_title('Head Rotation (Quaternion)')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Quaternion Component')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, blendshapes, head_pose, output_path, metadata=None):
        """
        Save inference results to file
        
        Args:
            blendshapes: Blendshape predictions
            head_pose: Head pose predictions
            output_path: Output file path
            metadata: Optional metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "metadata": {
                "model_type": "PyTorch TCN",
                "num_frames": len(blendshapes),
                "duration_ms": len(blendshapes) * 10,
                "frame_rate": 100,
                "inference_stats": {
                    "avg_time_ms": np.mean(self.inference_times) * 1000 if self.inference_times else 0,
                    "fps_capability": 1 / np.mean(self.inference_times) if self.inference_times else 0
                }
            },
            "blendshapes": {
                "shape": list(blendshapes.shape),
                "data": blendshapes.tolist(),
                "description": "52 MediaPipe facial blendshapes [0-1]"
            },
            "head_pose": {
                "shape": list(head_pose.shape),
                "data": head_pose.tolist(),
                "format": "[x, y, z, qw, qx, qy, qz]",
                "description": "3D translation + quaternion rotation"
            }
        }
        
        if metadata:
            results["metadata"].update(metadata)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        
        return results
    
    def run_demo(self, audio_path=None, output_dir="inference_results"):
        """
        Run complete inference demo
        
        Args:
            audio_path: Path to audio file (optional)
            output_dir: Output directory for results
        """
        print("\n" + "="*60)
        print("🎬 PYTORCH INFERENCE DEMO")
        print("="*60)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        if audio_path:
            audio = self.load_sample_audio(audio_path)
        else:
            audio = self.generate_synthetic_audio(duration=3.0)
        
        # Extract features
        mel_features = self.extract_mel_features(audio)
        
        # Create context windows
        context_windows = self.create_context_windows(mel_features)
        
        # Run inference
        blendshapes, head_pose, avg_time = self.predict_blendshapes(context_windows)
        
        # Save results
        results_path = output_dir / "pytorch_inference_results.json"
        metadata = {
            "audio_source": str(audio_path) if audio_path else "synthetic",
            "audio_duration_s": len(audio) / self.sample_rate
        }
        self.save_results(blendshapes, head_pose, results_path, metadata)
        
        # Create visualization
        plot_path = output_dir / "pytorch_inference_plot.png"
        self.visualize_results(blendshapes, head_pose, plot_path)
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"  Audio duration: {len(audio)/self.sample_rate:.2f}s")
        print(f"  Inference frames: {len(blendshapes)}")
        print(f"  Avg inference time: {avg_time*1000:.2f}ms per frame")
        print(f"  FPS capability: {1/avg_time:.1f}")
        print(f"  Real-time capable: {'✅' if avg_time < 0.033 else '❌'}")
        
        return blendshapes, head_pose

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="PyTorch model inference demo")
    parser.add_argument("--model", type=str, default="../models/best_full_model.pth",
                       help="Path to .pth model file")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file (optional, will generate synthetic if not provided)")
    parser.add_argument("--audio-scaler", type=str, default="../deployment/audio_scaler.pkl",
                       help="Path to audio scaler")
    parser.add_argument("--target-scaler", type=str, default="../deployment/target_scaler.pkl",
                       help="Path to target scaler")
    parser.add_argument("--output", type=str, default="pytorch_inference_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Create demo instance
        demo = PyTorchInferenceDemo(
            model_path=args.model,
            audio_scaler_path=args.audio_scaler,
            target_scaler_path=args.target_scaler
        )
        
        # Run demo
        blendshapes, head_pose = demo.run_demo(
            audio_path=args.audio,
            output_dir=args.output
        )
        
        print(f"\n✅ PyTorch inference demo completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

