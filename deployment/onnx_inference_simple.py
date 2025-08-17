#!/usr/bin/env python3
"""
Simple ONNX Inference Script
Attempts ONNX Runtime inference with graceful fallback handling
"""

import numpy as np
import librosa
import joblib
import json
import argparse
from pathlib import Path
import time
import uuid
import sys

# Try to import ONNX Runtime with graceful fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime successfully imported")
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"⚠️  ONNX Runtime not available: {e}")
    print("Will provide alternative instructions for ONNX usage")

class SimpleONNXInference:
    """Simple ONNX inference with graceful error handling"""
    
    def __init__(self, 
                 onnx_model_path="best_full_model.onnx",
                 audio_scaler_path="audio_scaler.pkl",
                 target_scaler_path="target_scaler.pkl"):
        
        self.model_path = onnx_model_path
        self.onnx_available = ONNX_AVAILABLE
        
        # Load scalers
        self.audio_scaler = self.load_scaler(audio_scaler_path, "audio")
        self.target_scaler = self.load_scaler(target_scaler_path, "target")
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.n_mels = 80
        self.hop_length = 160
        self.win_length = 400
        self.n_fft = 512
        self.context_frames = 24
        
        # Target FPS
        self.target_fps = 15
        self.frame_interval_ms = 1000 / self.target_fps
        
        # MediaPipe blendshape names
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
        
        # Try to load ONNX session
        if self.onnx_available:
            self.session = self.load_onnx_session()
        else:
            self.session = None
            print("🔄 ONNX Runtime not available - will generate demo data")
    
    def load_onnx_session(self):
        """Try to load ONNX Runtime session"""
        if not Path(self.model_path).exists():
            print(f"❌ ONNX model not found: {self.model_path}")
            return None
        
        try:
            print(f"📥 Loading ONNX model: {self.model_path}")
            
            # Create session with CPU provider only (more stable)
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            
            print(f"✅ ONNX Runtime session created successfully")
            print(f"  Providers: {session.get_providers()}")
            
            # Get input/output names
            self.input_name = session.get_inputs()[0].name
            self.output_name = session.get_outputs()[0].name
            
            print(f"  Input: {self.input_name}")
            print(f"  Output: {self.output_name}")
            
            return session
            
        except Exception as e:
            print(f"❌ Failed to load ONNX session: {e}")
            print("🔄 Will generate demo data instead")
            return None
    
    def load_scaler(self, scaler_path, scaler_type):
        """Load feature scaler"""
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"✅ Loaded {scaler_type} scaler")
            return scaler
        else:
            print(f"⚠️  {scaler_type} scaler not found")
            return None
    
    def generate_synthetic_audio(self, duration=3.0):
        """Generate synthetic speech-like audio"""
        print(f"🎵 Generating synthetic audio ({duration}s)...")
        
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Speech-like audio generation
        audio = np.zeros_like(t)
        f0_base = 120
        f0_variation = 50 * np.sin(2 * np.pi * 0.5 * t)
        f0 = f0_base + f0_variation
        
        # Add harmonics
        for harmonic in range(1, 6):
            amplitude = 0.3 / harmonic
            audio += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
        
        # Add formants
        formants = [800, 1200, 2400]
        for formant in formants:
            formant_signal = 0.1 * np.sin(2 * np.pi * formant * t)
            audio += formant_signal * (1 + 0.5 * np.sin(2 * np.pi * f0 * t))
        
        # Add envelope and pauses
        envelope = np.ones_like(t)
        pause_times = [1.2, 2.2]
        for pause_time in pause_times:
            pause_start = int(pause_time * self.sample_rate)
            pause_end = int((pause_time + 0.15) * self.sample_rate)
            if pause_end < len(envelope):
                envelope[pause_start:pause_end] *= 0.1
        
        audio *= envelope
        audio += 0.02 * np.random.randn(len(t))
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio.astype(np.float32)
    
    def extract_mel_features(self, audio):
        """Extract mel features"""
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
        return log_mel_spec.T  # (time, mels)
    
    def create_context_windows(self, mel_features):
        """Create context windows"""
        num_frames = mel_features.shape[0]
        
        if num_frames < self.context_frames:
            padding = np.zeros((self.context_frames - num_frames, self.n_mels))
            mel_features = np.vstack([padding, mel_features])
            num_frames = self.context_frames
        
        windows = []
        for i in range(num_frames - self.context_frames + 1):
            window = mel_features[i:i + self.context_frames]
            windows.append(window)
        
        return np.array(windows)
    
    def predict_with_onnx(self, context_windows):
        """Predict using ONNX Runtime"""
        if self.session is None:
            print("⚠️  ONNX session not available, generating demo data...")
            return self.generate_demo_predictions(len(context_windows))
        
        print(f"🧠 Running ONNX inference on {len(context_windows)} windows...")
        
        all_predictions = []
        
        try:
            for i, window in enumerate(context_windows):
                # Normalize input
                if self.audio_scaler is not None:
                    window_flat = window.reshape(-1, window.shape[-1])
                    window_normalized = self.audio_scaler.transform(window_flat)
                    window_normalized = window_normalized.reshape(window.shape)
                else:
                    window_normalized = window
                
                # Convert to float32 and add batch dimension
                input_array = window_normalized.astype(np.float32)[np.newaxis, ...]
                
                # ONNX Runtime inference
                ort_inputs = {self.input_name: input_array}
                ort_outputs = self.session.run([self.output_name], ort_inputs)
                output = ort_outputs[0]  # (1, seq_len, 59)
                
                # Get latest prediction
                output = output[0]  # (seq_len, 59)
                latest_prediction = output[-1]  # (59,)
                
                # Denormalize
                if self.target_scaler is not None:
                    latest_prediction = self.target_scaler.inverse_transform(
                        latest_prediction.reshape(1, -1)
                    )[0]
                
                all_predictions.append(latest_prediction)
            
            predictions = np.array(all_predictions)
            blendshapes = predictions[:, :52]
            head_pose = predictions[:, 52:]
            
            print(f"✅ ONNX inference completed successfully")
            return blendshapes, head_pose
            
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            print("🔄 Generating demo data instead...")
            return self.generate_demo_predictions(len(context_windows))
    
    def generate_demo_predictions(self, num_frames):
        """Generate realistic demo predictions when ONNX fails"""
        print(f"🎭 Generating demo blendshape data for {num_frames} frames...")
        
        # Generate realistic blendshape values
        blendshapes = np.zeros((num_frames, 52))
        head_pose = np.zeros((num_frames, 7))
        
        for i in range(num_frames):
            t = i / num_frames
            
            # Simulate speech patterns
            mouth_activity = 0.3 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz mouth movement
            jaw_open = 0.05 + 0.15 * mouth_activity
            
            # Key blendshapes with speech-like patterns
            blendshapes[i, 25] = jaw_open  # jawOpen (approximate index)
            blendshapes[i, 19] = mouth_activity * 0.2  # mouth activity
            blendshapes[i, 27] = 0.02 + 0.1 * mouth_activity  # mouth close
            
            # Eye blinks (occasional)
            if i % 120 == 0:  # Blink every ~1.2 seconds at 100fps
                blendshapes[i, 9] = 0.8  # eyeBlinkLeft
                blendshapes[i, 10] = 0.8  # eyeBlinkRight
            
            # Eyebrow movement
            blendshapes[i, 3] = 0.1 + 0.05 * np.sin(2 * np.pi * 0.5 * t)  # browInnerUp
            blendshapes[i, 4] = 0.15 + 0.08 * np.sin(2 * np.pi * 0.3 * t)  # browOuterUpLeft
            blendshapes[i, 5] = 0.15 + 0.08 * np.sin(2 * np.pi * 0.3 * t)  # browOuterUpRight
            
            # Smile variation
            smile_amount = 0.1 + 0.05 * np.sin(2 * np.pi * 0.2 * t)
            blendshapes[i, 43] = smile_amount  # mouthSmileLeft
            blendshapes[i, 44] = smile_amount  # mouthSmileRight
            
            # Head pose (subtle movement)
            head_pose[i, 0] = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # x translation
            head_pose[i, 1] = 0.05 * np.sin(2 * np.pi * 0.15 * t)  # y translation
            head_pose[i, 2] = -30.0  # z translation (distance)
            head_pose[i, 3] = 0.98 + 0.02 * np.sin(2 * np.pi * 0.05 * t)  # qw
            head_pose[i, 4] = 0.05 * np.sin(2 * np.pi * 0.08 * t)  # qx
            head_pose[i, 5] = 0.02 * np.sin(2 * np.pi * 0.12 * t)  # qy
            head_pose[i, 6] = 0.03 * np.sin(2 * np.pi * 0.07 * t)  # qz
        
        return blendshapes, head_pose
    
    def format_output(self, blendshapes, head_pose, audio_duration):
        """Format output in specified JSON structure"""
        session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
        start_time = int(time.time() * 1000)
        
        num_frames = len(blendshapes)
        audio_chunk_count = max(1, int(audio_duration / 0.5))
        
        frames = []
        
        for i in range(num_frames):
            timestamp = start_time + int(i * self.frame_interval_ms)
            
            # Create blendshapes dict
            blendshapes_dict = {}
            for j, name in enumerate(self.blendshape_names):
                if j < len(blendshapes[i]):
                    value = float(blendshapes[i][j])
                    if name != "_neutral":
                        value = max(0.0, min(1.0, abs(value)))
                    blendshapes_dict[name] = value
                else:
                    blendshapes_dict[name] = 0.0
            
            # Head pose
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
    
    def run_inference(self, audio_path=None, output_path="onnx_result.json"):
        """Run ONNX inference"""
        print("\n" + "="*60)
        print("🌐 ONNX INFERENCE DEMO")
        print("="*60)
        
        # Generate audio
        audio = self.generate_synthetic_audio(duration=3.0)
        audio_duration = len(audio) / self.sample_rate
        
        # Extract features
        print(f"🎯 Extracting mel features...")
        mel_features = self.extract_mel_features(audio)
        
        # Create context windows
        context_windows = self.create_context_windows(mel_features)
        print(f"  Context windows: {context_windows.shape}")
        
        # Run inference
        blendshapes, head_pose = self.predict_with_onnx(context_windows)
        
        # Format output
        formatted_output = self.format_output(blendshapes, head_pose, audio_duration)
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        
        print(f"\n✅ ONNX inference completed!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Summary:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Frame count: {formatted_output['frameCount']}")
        print(f"  Session ID: {formatted_output['sessionInfo']['sessionId']}")
        print(f"  ONNX Runtime used: {'✅' if self.session else '❌ (demo data)'}")
        
        return formatted_output

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple ONNX inference")
    parser.add_argument("--model", type=str, default="best_full_model.onnx",
                       help="ONNX model file")
    parser.add_argument("--output", type=str, default="onnx_inference_result.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        demo = SimpleONNXInference(onnx_model_path=args.model)
        result = demo.run_inference(output_path=args.output)
        
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

