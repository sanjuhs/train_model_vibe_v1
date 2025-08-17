#!/usr/bin/env python3
"""
Real-time Audio-to-Blendshapes Inference Pipeline
MediaPipe-like real-time processing with audio ring buffer
"""

import torch
import numpy as np
import sounddevice as sd
import librosa
import threading
import queue
import time
import json
import joblib
from collections import deque
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class AudioRingBuffer:
    """
    Ring buffer for real-time audio processing
    """
    def __init__(self, sample_rate=16000, buffer_duration=2.0, hop_length=160):
        """
        Initialize audio ring buffer
        
        Args:
            sample_rate: Audio sample rate
            buffer_duration: Buffer duration in seconds
            hop_length: Hop length for mel computation
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Ring buffer for raw audio
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Mel spectrogram parameters (must match training)
        self.n_mels = 80
        self.win_length = 400  # 25ms window
        self.n_fft = 512
        
        # Frame rate and timing
        self.mel_frame_rate = sample_rate / hop_length  # 100 Hz
        self.mel_frame_duration = hop_length / sample_rate  # 0.01 seconds
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        print(f"Audio Ring Buffer initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Buffer duration: {buffer_duration}s ({self.buffer_size} samples)")
        print(f"  Mel frame rate: {self.mel_frame_rate} Hz")
        print(f"  Frame duration: {self.mel_frame_duration*1000:.1f}ms")
    
    def append_audio(self, audio_chunk):
        """Add new audio chunk to ring buffer"""
        with self.lock:
            self.audio_buffer.extend(audio_chunk)
    
    def get_latest_mel_frame(self):
        """
        Get the latest mel spectrogram frame
        
        Returns:
            np.ndarray: Latest mel frame (n_mels,) or None if insufficient data
        """
        with self.lock:
            if len(self.audio_buffer) < self.win_length:
                return None
            
            # Get latest audio for one mel frame
            # We need enough samples for the window
            required_samples = self.win_length + self.hop_length
            if len(self.audio_buffer) < required_samples:
                return None
            
            # Extract latest audio segment
            audio_segment = np.array(list(self.audio_buffer)[-required_samples:])
            
            # Compute mel spectrogram for just the latest frame
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log mel and get the latest frame
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            latest_frame = log_mel_spec[:, -1]  # Last time frame
            
            return latest_frame
    
    def get_mel_context_window(self, context_frames=24):
        """
        Get mel context window for TCN inference
        
        Args:
            context_frames: Number of context frames needed
        
        Returns:
            np.ndarray: Mel context (context_frames, n_mels) or None
        """
        with self.lock:
            # Calculate required audio samples for context
            context_duration = context_frames * self.hop_length / self.sample_rate
            required_samples = int(context_duration * self.sample_rate) + self.win_length
            
            if len(self.audio_buffer) < required_samples:
                return None
            
            # Extract audio for context window
            audio_segment = np.array(list(self.audio_buffer)[-required_samples:])
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
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
            
            # Get the latest context_frames
            if log_mel_spec.shape[1] >= context_frames:
                context_window = log_mel_spec[:, -context_frames:].T  # (time, mels)
                return context_window
            else:
                return None

class RealTimeBlendshapePredictor:
    """
    Real-time blendshape predictor using trained TCN model
    """
    def __init__(self, 
                 model_path="models/best_tcn_model.pth",
                 scaler_path="extracted_features/audio_scaler.pkl",
                 target_scaler_path="extracted_features/target_scaler.pkl",
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize real-time predictor
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to audio feature scaler
            target_scaler_path: Path to target scaler
            device: Device for inference
        """
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        
        # Load scalers
        self.audio_scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None
        self.target_scaler = joblib.load(target_scaler_path) if Path(target_scaler_path).exists() else None
        
        # Model parameters
        self.context_frames = 24  # 240ms context
        self.output_dim = 59      # 52 blendshapes + 7 pose
        
        # EMA smoothing for stability
        self.ema_alpha = 0.85     # Smoothing factor
        self.previous_output = None
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
        print(f"Real-time Predictor initialized:")
        print(f"  Device: {device}")
        print(f"  Context frames: {self.context_frames} ({self.context_frames*10}ms)")
        print(f"  EMA smoothing: α={self.ema_alpha}")
        print(f"  Output features: {self.output_dim}")
    
    def _load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Create and load model
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from: {self.model_path}")
        return model
    
    def predict(self, mel_context):
        """
        Predict blendshapes from mel context
        
        Args:
            mel_context: Mel context window (context_frames, n_mels)
        
        Returns:
            np.ndarray: Predicted blendshapes + pose (59,)
        """
        if mel_context is None or mel_context.shape[0] < self.context_frames:
            # Return previous output or zeros if no context
            if self.previous_output is not None:
                return self.previous_output
            else:
                return np.zeros(self.output_dim)
        
        start_time = time.time()
        
        # Normalize input if scaler available
        if self.audio_scaler is not None:
            mel_flat = mel_context.reshape(-1, mel_context.shape[-1])
            mel_normalized_flat = self.audio_scaler.transform(mel_flat)
            mel_normalized = mel_normalized_flat.reshape(mel_context.shape)
        else:
            mel_normalized = mel_context
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(mel_normalized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            output = output_tensor.squeeze(0).cpu().numpy()  # (seq_len, 59)
        
        # Get the latest prediction (last time step)
        latest_prediction = output[-1]  # (59,)
        
        # Denormalize output if scaler available
        if self.target_scaler is not None:
            latest_prediction = self.target_scaler.inverse_transform(latest_prediction.reshape(1, -1))[0]
        
        # Apply EMA smoothing for stability
        if self.previous_output is not None:
            smoothed_output = (self.ema_alpha * self.previous_output + 
                             (1 - self.ema_alpha) * latest_prediction)
        else:
            smoothed_output = latest_prediction
        
        self.previous_output = smoothed_output
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return smoothed_output
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.inference_times) == 0:
            return None
        
        times_ms = np.array(self.inference_times) * 1000
        return {
            'avg_inference_time_ms': np.mean(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'fps_capability': 1000.0 / np.mean(times_ms),
            'real_time_capable': np.mean(times_ms) < 33.33  # 30 FPS
        }

class RealTimeAudioProcessor:
    """
    Complete real-time audio processing pipeline
    """
    def __init__(self, 
                 sample_rate=16000,
                 chunk_size=160,  # 10ms chunks at 16kHz
                 buffer_duration=2.0):
        """
        Initialize real-time audio processor
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size (samples per callback)
            buffer_duration: Ring buffer duration
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        
        # Initialize components
        self.ring_buffer = AudioRingBuffer(sample_rate, buffer_duration)
        self.predictor = RealTimeBlendshapePredictor()
        
        # Output queue for blendshapes
        self.output_queue = queue.Queue(maxsize=100)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        
        print(f"Real-time Audio Processor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Chunk size: {chunk_size} samples ({chunk_size/sample_rate*1000:.1f}ms)")
        print(f"  Buffer duration: {buffer_duration}s")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio to ring buffer
        audio_chunk = indata[:, 0]  # Take first channel if stereo
        self.ring_buffer.append_audio(audio_chunk)
        
        # Process every few frames to match target frame rate
        self.frame_count += 1
        
        # Process at ~30 Hz (every 5-6 audio chunks at 100Hz chunk rate)
        if self.frame_count % 3 == 0:  # ~33 Hz processing
            self._process_frame()
    
    def _process_frame(self):
        """Process one frame of blendshape prediction"""
        try:
            # Get mel context window
            mel_context = self.ring_buffer.get_mel_context_window(self.predictor.context_frames)
            
            if mel_context is not None:
                # Predict blendshapes
                blendshapes = self.predictor.predict(mel_context)
                
                # Add timestamp and put in output queue
                timestamp = time.time()
                output_frame = {
                    'timestamp': timestamp,
                    'blendshapes': blendshapes[:52].tolist(),  # First 52 are blendshapes
                    'head_pose': blendshapes[52:].tolist(),    # Last 7 are head pose
                    'frame_id': self.frame_count
                }
                
                # Non-blocking put to avoid audio dropouts
                try:
                    self.output_queue.put_nowait(output_frame)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(output_frame)
                    except queue.Empty:
                        pass
        
        except Exception as e:
            print(f"Error in frame processing: {e}")
    
    def start_processing(self, duration=None):
        """
        Start real-time processing
        
        Args:
            duration: Processing duration in seconds (None for infinite)
        """
        print("\\nStarting real-time audio processing...")
        print("Speak into your microphone!")
        print("Press Ctrl+C to stop\\n")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                dtype=np.float32
            ):
                if duration:
                    time.sleep(duration)
                else:
                    while self.running:
                        # Print periodic statistics
                        if self.frame_count % 300 == 0:  # Every ~10 seconds
                            self._print_stats()
                        time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\\nStopping...")
        
        finally:
            self.running = False
            self._print_final_stats()
    
    def _print_stats(self):
        """Print current statistics"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        perf_stats = self.predictor.get_performance_stats()
        if perf_stats:
            print(f"[{elapsed:.1f}s] Frames: {self.frame_count}, "
                  f"FPS: {fps:.1f}, "
                  f"Inference: {perf_stats['avg_inference_time_ms']:.2f}ms, "
                  f"Queue: {self.output_queue.qsize()}")
    
    def _print_final_stats(self):
        """Print final statistics"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        perf_stats = self.predictor.get_performance_stats()
        
        print(f"\\nFinal Statistics:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Output queue final size: {self.output_queue.qsize()}")
        
        if perf_stats:
            print(f"  Average inference time: {perf_stats['avg_inference_time_ms']:.2f}ms")
            print(f"  Max inference time: {perf_stats['max_inference_time_ms']:.2f}ms")
            print(f"  Real-time capable: {perf_stats['real_time_capable']}")
    
    def get_latest_output(self):
        """Get latest blendshape output (non-blocking)"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

def main():
    """
    Main function for real-time processing demo
    """
    print("=== REAL-TIME AUDIO-TO-BLENDSHAPES DEMO ===")
    
    try:
        # Create processor
        processor = RealTimeAudioProcessor(
            sample_rate=16000,
            chunk_size=160,     # 10ms chunks
            buffer_duration=2.0  # 2 second buffer
        )
        
        # Start processing for 30 seconds (or until Ctrl+C)
        processor.start_processing(duration=30)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("  1. A microphone connected")
        print("  2. Trained model at models/best_tcn_model.pth")
        print("  3. Audio scalers at extracted_features/")

if __name__ == "__main__":
    main()