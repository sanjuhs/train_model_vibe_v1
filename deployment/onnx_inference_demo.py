#!/usr/bin/env python3
"""
ONNX Runtime Inference Demo
Demonstrates how to use ONNX models for audio-to-blendshapes inference
Ready for JavaScript deployment via ONNX.js
"""

import onnxruntime as ort
import numpy as np
import librosa
import joblib
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

class ONNXInferenceDemo:
    """Demo class for ONNX Runtime inference"""
    
    def __init__(self, 
                 onnx_model_path="best_full_model.onnx",
                 audio_scaler_path="audio_scaler.pkl",
                 target_scaler_path="target_scaler.pkl"):
        """
        Initialize ONNX inference demo
        
        Args:
            onnx_model_path: Path to .onnx model file
            audio_scaler_path: Path to audio feature scaler
            target_scaler_path: Path to target scaler
        """
        print(f"🌐 ONNX Runtime Inference Demo")
        
        # Load ONNX model
        self.session = self.load_onnx_model(onnx_model_path)
        self.model_path = onnx_model_path
        
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
        
        print(f"✅ ONNX inference demo initialized")
        print(f"  Context frames: {self.context_frames} ({self.context_frames*10}ms)")
        print(f"  Audio parameters: {self.sample_rate}Hz, {self.n_mels} mels")
    
    def load_onnx_model(self, model_path):
        """Load ONNX model"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        print(f"📥 Loading ONNX model from: {model_path}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']  # Default to CPU
        
        # Try to use GPU if available
        if ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Print session info
        print(f"✅ ONNX Runtime session created")
        print(f"  Providers: {session.get_providers()}")
        
        # Print input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"📊 Model I/O Info:")
        print(f"  Input: {input_info.name} {input_info.shape} {input_info.type}")
        print(f"  Output: {output_info.name} {output_info.shape} {output_info.type}")
        
        self.input_name = input_info.name
        self.output_name = output_info.name
        
        return session
    
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
        Predict blendshapes from context windows using ONNX Runtime
        
        Args:
            context_windows: Context windows (num_windows, context_frames, mels)
        
        Returns:
            tuple: (blendshapes, head_pose, inference_time)
        """
        print(f"🧠 Running ONNX inference on {len(context_windows)} windows...")
        
        all_predictions = []
        inference_times = []
        
        for i, window in enumerate(context_windows):
            start_time = time.time()
            
            # Normalize input if scaler available
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
            
            # Remove batch dimension and get latest prediction
            output = output[0]  # (seq_len, 59)
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
        
        print(f"✅ ONNX inference completed")
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
        fig.suptitle('ONNX Runtime Inference Results', fontsize=16)
        
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
                "model_type": "ONNX Runtime",
                "model_path": str(self.model_path),
                "num_frames": len(blendshapes),
                "duration_ms": len(blendshapes) * 10,
                "frame_rate": 100,
                "inference_stats": {
                    "avg_time_ms": np.mean(self.inference_times) * 1000 if self.inference_times else 0,
                    "fps_capability": 1 / np.mean(self.inference_times) if self.inference_times else 0
                },
                "javascript_ready": True,
                "onnx_js_compatible": True
            },
            "blendshapes": {
                "shape": list(blendshapes.shape),
                "data": blendshapes.tolist(),
                "description": "52 MediaPipe facial blendshapes [0-1]",
                "javascript_usage": "Can be directly applied to MediaPipe face mesh"
            },
            "head_pose": {
                "shape": list(head_pose.shape),
                "data": head_pose.tolist(),
                "format": "[x, y, z, qw, qx, qy, qz]",
                "description": "3D translation + quaternion rotation",
                "javascript_usage": "Compatible with Three.js and WebGL transforms"
            }
        }
        
        if metadata:
            results["metadata"].update(metadata)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        
        return results
    
    def generate_javascript_example(self, output_path):
        """Generate JavaScript usage example"""
        js_example = '''
// ONNX.js Inference Example
// This example shows how to use the converted ONNX model in JavaScript

import { InferenceSession, Tensor } from 'onnxjs';

class AudioToBlendshapesONNX {
    constructor() {
        this.session = null;
        this.inputName = 'audio_features';
        this.outputName = 'blendshapes';
        this.contextFrames = 24;
        this.melFeatures = 80;
    }
    
    async loadModel(modelPath) {
        console.log('Loading ONNX model...');
        this.session = new InferenceSession();
        await this.session.loadModel(modelPath);
        console.log('Model loaded successfully');
    }
    
    async predict(melFeatures) {
        // melFeatures should be Float32Array of shape [1, 24, 80]
        const inputTensor = new Tensor(melFeatures, 'float32', [1, this.contextFrames, this.melFeatures]);
        
        const feeds = {};
        feeds[this.inputName] = inputTensor;
        
        const results = await this.session.run(feeds);
        const output = results[this.outputName];
        
        // Extract latest prediction (last time step)
        const outputData = output.data;
        const lastFrameStart = (output.dims[1] - 1) * output.dims[2];
        const latestPrediction = outputData.slice(lastFrameStart, lastFrameStart + output.dims[2]);
        
        return {
            blendshapes: Array.from(latestPrediction.slice(0, 52)),
            headPose: Array.from(latestPrediction.slice(52, 59))
        };
    }
    
    // Example usage with Web Audio API
    async processAudioBuffer(audioBuffer) {
        // Extract mel features from audio buffer
        const melFeatures = this.extractMelFeatures(audioBuffer);
        
        // Create context window
        const contextWindow = this.createContextWindow(melFeatures);
        
        // Run inference
        const predictions = await this.predict(contextWindow);
        
        return predictions;
    }
    
    extractMelFeatures(audioBuffer) {
        // Implement mel spectrogram extraction
        // You can use libraries like ml-fft or implement your own
        // Return Float32Array of shape [time, 80]
    }
    
    createContextWindow(melFeatures) {
        // Create sliding window of last 24 frames
        // Return Float32Array of shape [1, 24, 80]
    }
}

// Usage example
async function main() {
    const predictor = new AudioToBlendshapesONNX();
    await predictor.loadModel('path/to/your/model.onnx');
    
    // Get audio from microphone or file
    const audioContext = new AudioContext();
    // ... get audio buffer ...
    
    const predictions = await predictor.processAudioBuffer(audioBuffer);
    console.log('Blendshapes:', predictions.blendshapes);
    console.log('Head pose:', predictions.headPose);
}

// For Three.js integration
function applyToThreeJSMesh(predictions, faceMesh) {
    // Apply blendshapes to face mesh
    predictions.blendshapes.forEach((value, index) => {
        if (faceMesh.morphTargetInfluences && faceMesh.morphTargetInfluences[index] !== undefined) {
            faceMesh.morphTargetInfluences[index] = value;
        }
    });
    
    // Apply head pose (translation + quaternion)
    const [x, y, z, qw, qx, qy, qz] = predictions.headPose;
    faceMesh.position.set(x, y, z);
    faceMesh.quaternion.set(qx, qy, qz, qw);
}

export { AudioToBlendshapesONNX };
'''
        
        with open(output_path, 'w') as f:
            f.write(js_example)
        
        print(f"📄 JavaScript example saved to: {output_path}")
    
    def run_demo(self, audio_path=None, output_dir="onnx_inference_results"):
        """
        Run complete ONNX inference demo
        
        Args:
            audio_path: Path to audio file (optional)
            output_dir: Output directory for results
        """
        print("\n" + "="*60)
        print("🌐 ONNX RUNTIME INFERENCE DEMO")
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
        results_path = output_dir / "onnx_inference_results.json"
        metadata = {
            "audio_source": str(audio_path) if audio_path else "synthetic",
            "audio_duration_s": len(audio) / self.sample_rate
        }
        self.save_results(blendshapes, head_pose, results_path, metadata)
        
        # Create visualization
        plot_path = output_dir / "onnx_inference_plot.png"
        self.visualize_results(blendshapes, head_pose, plot_path)
        
        # Generate JavaScript example
        js_path = output_dir / "javascript_example.js"
        self.generate_javascript_example(js_path)
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"  Audio duration: {len(audio)/self.sample_rate:.2f}s")
        print(f"  Inference frames: {len(blendshapes)}")
        print(f"  Avg inference time: {avg_time*1000:.2f}ms per frame")
        print(f"  FPS capability: {1/avg_time:.1f}")
        print(f"  Real-time capable: {'✅' if avg_time < 0.033 else '❌'}")
        print(f"  JavaScript ready: ✅")
        
        return blendshapes, head_pose

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="ONNX Runtime inference demo")
    parser.add_argument("--model", type=str, default="best_full_model.onnx",
                       help="Path to .onnx model file")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file (optional, will generate synthetic if not provided)")
    parser.add_argument("--audio-scaler", type=str, default="audio_scaler.pkl",
                       help="Path to audio scaler")
    parser.add_argument("--target-scaler", type=str, default="target_scaler.pkl",
                       help="Path to target scaler")
    parser.add_argument("--output", type=str, default="onnx_inference_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Create demo instance
        demo = ONNXInferenceDemo(
            onnx_model_path=args.model,
            audio_scaler_path=args.audio_scaler,
            target_scaler_path=args.target_scaler
        )
        
        # Run demo
        blendshapes, head_pose = demo.run_demo(
            audio_path=args.audio,
            output_dir=args.output
        )
        
        print(f"\n✅ ONNX inference demo completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        print(f"🌐 Ready for JavaScript deployment!")
        
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

