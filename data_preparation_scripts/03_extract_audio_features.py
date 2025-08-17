#!/usr/bin/env python3
"""
Audio Feature Extraction Script
Extracts audio from video and computes mel spectrograms for the TCN model
"""

import librosa
import numpy as np
import soundfile as sf
import json
import os
from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80, hop_length=160, win_length=400, n_fft=512):
        """
        Initialize audio feature extractor
        
        Args:
            sample_rate: Target sample rate (16kHz for efficiency)
            n_mels: Number of mel filter banks (80 is standard)
            hop_length: Hop length in samples (10ms at 16kHz)
            win_length: Window length in samples (25ms at 16kHz)
            n_fft: FFT size
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        
        # Calculate frame rate for mel spectrograms
        # At 16kHz with hop_length=160, we get 100 mel frames per second
        self.mel_frame_rate = sample_rate / hop_length
        
        print(f"Audio Feature Extractor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Mel features: {n_mels}")
        print(f"  Mel frame rate: {self.mel_frame_rate} Hz")
        print(f"  Hop length: {hop_length} samples ({hop_length/sample_rate*1000:.1f}ms)")
        print(f"  Window length: {win_length} samples ({win_length/sample_rate*1000:.1f}ms)")
    
    def extract_audio_from_video(self, video_path, output_dir="extracted_features"):
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save audio
        
        Returns:
            str: Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting audio from: {video_path}")
        
        # Load video with moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save audio as WAV file
        audio_path = Path(output_dir) / "extracted_audio.wav"
        audio.write_audiofile(str(audio_path), 
                             fps=self.sample_rate)
        
        # Clean up
        audio.close()
        video.close()
        
        print(f"Audio extracted to: {audio_path}")
        return str(audio_path)
    
    def extract_mel_features(self, audio_path, output_dir="extracted_features", max_duration=None):
        """
        Extract mel spectrogram features from audio
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save features
            max_duration: Maximum duration to process (seconds, for testing)
        
        Returns:
            dict: Extracted audio features
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Loading audio from: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Limit duration for testing
        if max_duration:
            max_samples = int(max_duration * self.sample_rate)
            audio = audio[:max_samples]
            print(f"Limited audio to {max_duration} seconds ({len(audio)} samples)")
        
        print(f"Audio loaded: {len(audio)} samples, {len(audio)/sr:.2f} seconds")
        
        # Extract mel spectrogram
        print("Computing mel spectrogram...")
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
        
        # Convert to log mel spectrogram (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, features) format
        mel_features = log_mel_spec.T  # Shape: (time_frames, n_mels)
        
        # Compute additional features
        print("Computing additional audio features...")
        
        # Voice Activity Detection (VAD) using RMS energy
        rms_energy = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
            frame_length=self.win_length
        )[0]
        
        # Simple VAD threshold (adjust based on your data)
        vad_threshold = np.percentile(rms_energy, 30)  # Bottom 30% is likely silence
        voice_activity = (rms_energy > vad_threshold).astype(float)
        
        # Zero Crossing Rate (useful for voiced/unvoiced detection)
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length,
            frame_length=self.win_length
        )[0]
        
        # Ensure all features have the same length
        min_length = min(len(mel_features), len(voice_activity), len(zcr))
        mel_features = mel_features[:min_length]
        voice_activity = voice_activity[:min_length]
        zcr = zcr[:min_length]
        
        # Create time stamps for each frame
        timestamps = librosa.frames_to_time(
            range(min_length),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Compile features
        audio_features = {
            'audio_path': audio_path,
            'sample_rate': self.sample_rate,
            'duration_seconds': len(audio) / self.sample_rate,
            'n_mels': self.n_mels,
            'hop_length': self.hop_length,
            'mel_frame_rate': self.mel_frame_rate,
            'n_frames': min_length,
            'timestamps': timestamps.tolist(),
            'mel_features': mel_features.tolist(),  # Shape: (time, n_mels)
            'voice_activity': voice_activity.tolist(),
            'zero_crossing_rate': zcr.tolist(),
            'rms_energy': rms_energy[:min_length].tolist()
        }
        
        # Save features
        output_file = Path(output_dir) / "audio_features.json"
        with open(output_file, 'w') as f:
            json.dump(audio_features, f, indent=2)
        
        print(f"\\nAudio feature extraction complete!")
        print(f"Mel features shape: {mel_features.shape}")
        print(f"Features saved to: {output_file}")
        
        return audio_features
    
    def extract_from_video(self, video_path, output_dir="extracted_features", max_duration=None):
        """
        Complete pipeline: extract audio from video and compute features
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save features
            max_duration: Maximum duration to process (seconds, for testing)
        
        Returns:
            dict: Extracted audio features
        """
        # Step 1: Extract audio from video
        audio_path = self.extract_audio_from_video(video_path, output_dir)
        
        # Step 2: Extract mel features from audio
        features = self.extract_mel_features(audio_path, output_dir, max_duration)
        
        return features

def main():
    """
    Main function to extract audio features from video
    """
    video_path = "videodata/ml_sanjay_assortmentSounds55_15min_dataset.mp4"
    
    print("Initializing Audio Feature Extractor...")
    extractor = AudioFeatureExtractor(
        sample_rate=16000,  # 16kHz for efficiency
        n_mels=80,          # 80 mel features
        hop_length=160,     # 10ms hop (160 samples at 16kHz)
        win_length=400,     # 25ms window (400 samples at 16kHz)
        n_fft=512          # 512-point FFT
    )
    
    print("\\nStarting audio feature extraction...")
    
    # For testing, limit to same duration as blendshapes (~33 seconds)
    # Remove max_duration=33.33 to process the entire video
    features = extractor.extract_from_video(
        video_path,
        output_dir="extracted_features",
        max_duration=33.33  # Match the 1000 frames we extracted earlier
    )
    
    # Print summary
    print("\\n" + "="*60)
    print("AUDIO FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Audio file: {features['audio_path']}")
    print(f"Duration: {features['duration_seconds']:.2f} seconds")
    print(f"Sample rate: {features['sample_rate']} Hz")
    print(f"Mel features: {features['n_mels']}")
    print(f"Number of frames: {features['n_frames']}")
    print(f"Mel frame rate: {features['mel_frame_rate']:.1f} Hz")
    print(f"Feature dimensions: {len(features['mel_features'])} x {len(features['mel_features'][0])}")
    
    # Voice activity statistics
    vad_ratio = np.mean(features['voice_activity'])
    print(f"Voice activity: {vad_ratio:.1%} of frames")
    
    print("="*60)

if __name__ == "__main__":
    main()