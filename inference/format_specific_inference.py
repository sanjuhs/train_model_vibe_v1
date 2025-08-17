#!/usr/bin/env python3
"""
Format-Specific Inference Script
Generates inference results in the exact JSON format specified by the user
Uses PyTorch only to avoid ONNX Runtime issues
"""

import torch
import numpy as np
import librosa
import joblib
import json
import argparse
from pathlib import Path
import sys
import time
import uuid

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class FormattedInferenceDemo:
    """Demo class that outputs in the specified JSON format"""
    
    def __init__(self, 
                 model_path="../models/best_full_model.pth",
                 audio_scaler_path="../deployment/audio_scaler.pkl",
                 target_scaler_path="../deployment/target_scaler.pkl"):
        """
        Initialize inference demo
        
        Args:
            model_path: Path to .pth model file
            audio_scaler_path: Path to audio feature scaler
            target_scaler_path: Path to target scaler
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Formatted Inference Demo on {self.device}")
        
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
        
        # Target FPS (can be adjusted)
        self.target_fps = 15
        self.frame_interval_ms = 1000 / self.target_fps  # ~66.67ms for 15 FPS
        
        # MediaPipe blendshape names (52 total)
        self.blendshape_names = [
            "_neutral",
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
            "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
            "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            "jawForward", "jawLeft", "jawOpen", "jawRight",
            "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
            "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
            "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight"
        ]
        
        print(f"✅ Formatted inference demo initialized")
        print(f"  Target FPS: {self.target_fps}")
        print(f"  Frame interval: {self.frame_interval_ms:.1f}ms")
        print(f"  Blendshapes: {len(self.blendshape_names)}")
    
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
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded direct state dict")
        
        model.eval()
        model.to(self.device)
        
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
    
    def load_sample_audio(self, audio_path, duration=5.0):
        """
        Load sample audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load (seconds)
        
        Returns:
            np.ndarray: Audio samples
        """
        if audio_path and Path(audio_path).exists():
            print(f"🎵 Loading audio from: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
            print(f"  Duration: {len(audio)/sr:.2f}s")
        else:
            print(f"🎵 Generating synthetic speech-like audio ({duration}s)...")
            audio = self.generate_speech_like_audio(duration)
        
        return audio
    
    def generate_speech_like_audio(self, duration=5.0):
        """Generate more realistic speech-like synthetic audio"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Create speech-like audio with varying intensity
        audio = np.zeros_like(t)
        
        # Fundamental frequency variation (simulating speech prosody)
        f0_base = 120  # Base fundamental frequency
        f0_variation = 50 * np.sin(2 * np.pi * 0.5 * t)  # Slow prosodic variation
        f0 = f0_base + f0_variation
        
        # Add harmonics (like speech formants)
        for harmonic in range(1, 6):
            amplitude = 0.3 / harmonic  # Decreasing amplitude for higher harmonics
            audio += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
        
        # Add formant-like resonances
        formants = [800, 1200, 2400]  # Typical formant frequencies
        for formant in formants:
            formant_signal = 0.1 * np.sin(2 * np.pi * formant * t)
            # Modulate formants with fundamental
            audio += formant_signal * (1 + 0.5 * np.sin(2 * np.pi * f0 * t))
        
        # Add speech-like envelope (pauses and emphasis)
        envelope = np.ones_like(t)
        # Add some pauses
        pause_times = [1.2, 2.8, 4.1]
        for pause_time in pause_times:
            pause_start = int(pause_time * self.sample_rate)
            pause_end = int((pause_time + 0.2) * self.sample_rate)
            if pause_end < len(envelope):
                envelope[pause_start:pause_end] *= 0.1
        
        # Add emphasis points
        emphasis_times = [0.5, 1.8, 3.2, 4.5]
        for emphasis_time in emphasis_times:
            emphasis_idx = int(emphasis_time * self.sample_rate)
            emphasis_width = int(0.3 * self.sample_rate)
            start_idx = max(0, emphasis_idx - emphasis_width // 2)
            end_idx = min(len(envelope), emphasis_idx + emphasis_width // 2)
            envelope[start_idx:end_idx] *= 1.5
        
        audio *= envelope
        
        # Add realistic noise
        noise_level = 0.02
        audio += noise_level * np.random.randn(len(t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio.astype(np.float32)
    
    def extract_mel_features(self, audio):
        """Extract mel spectrogram features"""
        print(f"🎯 Extracting mel features...")
        
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
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_features = log_mel_spec.T  # (time, mels)
        
        print(f"  Mel features shape: {mel_features.shape}")
        return mel_features
    
    def create_context_windows(self, mel_features):
        """Create context windows for inference"""
        num_frames = mel_features.shape[0]
        
        if num_frames < self.context_frames:
            # Pad with zeros
            padding = np.zeros((self.context_frames - num_frames, self.n_mels))
            mel_features = np.vstack([padding, mel_features])
            num_frames = self.context_frames
        
        windows = []
        for i in range(num_frames - self.context_frames + 1):
            window = mel_features[i:i + self.context_frames]
            windows.append(window)
        
        return np.array(windows)
    
    def predict_blendshapes(self, context_windows):
        """Predict blendshapes using PyTorch model"""
        print(f"🧠 Running PyTorch inference on {len(context_windows)} windows...")
        
        all_predictions = []
        
        with torch.no_grad():
            for i, window in enumerate(context_windows):
                # Normalize input if scaler available
                if self.audio_scaler is not None:
                    window_flat = window.reshape(-1, window.shape[-1])
                    window_normalized = self.audio_scaler.transform(window_flat)
                    window_normalized = window_normalized.reshape(window.shape)
                else:
                    window_normalized = window
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)
                
                # Forward pass
                output_tensor = self.model(input_tensor)
                output = output_tensor.squeeze(0).cpu().numpy()  # (seq_len, 59)
                
                # Get latest prediction
                latest_prediction = output[-1]  # (59,)
                
                # Denormalize if scaler available
                if self.target_scaler is not None:
                    latest_prediction = self.target_scaler.inverse_transform(
                        latest_prediction.reshape(1, -1)
                    )[0]
                
                all_predictions.append(latest_prediction)
        
        predictions = np.array(all_predictions)  # (num_windows, 59)
        
        # Split predictions
        blendshapes = predictions[:, :52]  # (num_windows, 52)
        head_pose = predictions[:, 52:]    # (num_windows, 7)
        
        print(f"✅ Inference completed")
        print(f"  Blendshapes shape: {blendshapes.shape}")
        print(f"  Head pose shape: {head_pose.shape}")
        
        return blendshapes, head_pose
    
    def format_output(self, blendshapes, head_pose, audio_duration):
        """Format output in the exact JSON structure specified"""
        
        # Generate session info
        session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
        start_time = int(time.time() * 1000)
        
        # Calculate timing for target FPS
        num_frames = len(blendshapes)
        audio_chunk_count = max(1, int(audio_duration / 0.5))  # Assume 0.5s chunks
        
        frames = []
        
        for i in range(num_frames):
            # Calculate timestamp based on target FPS
            timestamp = start_time + int(i * self.frame_interval_ms)
            
            # Create blendshapes dict
            blendshapes_dict = {}
            for j, name in enumerate(self.blendshape_names):
                if j < len(blendshapes[i]):
                    # Ensure values are in reasonable range [0, 1] for most blendshapes
                    value = float(blendshapes[i][j])
                    # Clamp to reasonable range
                    if name != "_neutral":
                        value = max(0.0, min(1.0, abs(value)))
                    blendshapes_dict[name] = value
                else:
                    blendshapes_dict[name] = 0.0
            
            # Extract head pose (7 values: x, y, z, qw, qx, qy, qz)
            if len(head_pose[i]) >= 7:
                head_position = {
                    "x": float(head_pose[i][0]),
                    "y": float(head_pose[i][1]), 
                    "z": float(head_pose[i][2])
                }
                head_rotation = {
                    "w": float(head_pose[i][3]),
                    "x": float(head_pose[i][4]),
                    "y": float(head_pose[i][5]),
                    "z": float(head_pose[i][6])
                }
            else:
                # Default values if pose data incomplete
                head_position = {"x": 0.0, "y": 0.0, "z": -30.0}
                head_rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
            
            frame = {
                "timestamp": timestamp,
                "sessionId": session_id,
                "blendshapes": blendshapes_dict,
                "headPosition": head_position,
                "headRotation": head_rotation,
                "id": i + 1
            }
            
            frames.append(frame)
        
        # Create final output structure
        output = {
            "sessionInfo": {
                "sessionId": session_id,
                "startTime": start_time,
                "targetFPS": self.target_fps
            },
            "frameCount": num_frames,
            "audioChunkCount": audio_chunk_count,
            "frames": frames
        }
        
        return output
    
    def run_inference(self, audio_path=None, output_path="inference_result.json"):
        """Run complete inference and save formatted output"""
        
        print("\n" + "="*60)
        print("🎬 FORMATTED INFERENCE DEMO")
        print("="*60)
        
        # Load audio
        audio = self.load_sample_audio(audio_path)
        audio_duration = len(audio) / self.sample_rate
        
        # Extract features
        mel_features = self.extract_mel_features(audio)
        
        # Create context windows
        context_windows = self.create_context_windows(mel_features)
        
        # Run inference
        blendshapes, head_pose = self.predict_blendshapes(context_windows)
        
        # Format output
        formatted_output = self.format_output(blendshapes, head_pose, audio_duration)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        
        print(f"\n✅ Inference completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Summary:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Frame count: {formatted_output['frameCount']}")
        print(f"  Audio chunks: {formatted_output['audioChunkCount']}")
        print(f"  Target FPS: {formatted_output['sessionInfo']['targetFPS']}")
        print(f"  Session ID: {formatted_output['sessionInfo']['sessionId']}")
        
        return formatted_output

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Format-specific inference demo")
    parser.add_argument("--model", type=str, default="../models/best_full_model.pth",
                       help="Path to .pth model file")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file (optional)")
    parser.add_argument("--audio-scaler", type=str, default="../deployment/audio_scaler.pkl",
                       help="Path to audio scaler")
    parser.add_argument("--target-scaler", type=str, default="../deployment/target_scaler.pkl",
                       help="Path to target scaler")
    parser.add_argument("--output", type=str, default="formatted_inference_result.json",
                       help="Output JSON file")
    parser.add_argument("--fps", type=int, default=15,
                       help="Target FPS for output")
    
    args = parser.parse_args()
    
    try:
        # Create demo instance
        demo = FormattedInferenceDemo(
            model_path=args.model,
            audio_scaler_path=args.audio_scaler,
            target_scaler_path=args.target_scaler
        )
        
        # Set target FPS
        demo.target_fps = args.fps
        demo.frame_interval_ms = 1000 / args.fps
        
        # Run inference
        result = demo.run_inference(
            audio_path=args.audio,
            output_path=args.output
        )
        
        # Print first frame as example
        if result['frames']:
            print(f"\n📄 First frame example:")
            first_frame = result['frames'][0]
            print(json.dumps(first_frame, indent=2)[:500] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

