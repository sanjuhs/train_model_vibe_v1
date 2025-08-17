#!/usr/bin/env python3
"""
Dataset Creation Script
Synchronizes audio features with blendshape/pose targets for training
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

class DatasetCreator:
    def __init__(self, sequence_length_ms=240, overlap_ms=120):
        """
        Initialize dataset creator
        
        Args:
            sequence_length_ms: Length of input sequences in milliseconds (240-320ms recommended)
            overlap_ms: Overlap between sequences in milliseconds
        """
        self.sequence_length_ms = sequence_length_ms
        self.overlap_ms = overlap_ms
        
        print(f"Dataset Creator initialized:")
        print(f"  Sequence length: {sequence_length_ms}ms")
        print(f"  Overlap: {overlap_ms}ms")
        print(f"  Step size: {sequence_length_ms - overlap_ms}ms")
    
    def load_features(self, features_dir="extracted_features"):
        """
        Load both audio and visual features
        
        Args:
            features_dir: Directory containing extracted features
        
        Returns:
            tuple: (audio_data, visual_data)
        """
        features_path = Path(features_dir)
        
        # Load audio features
        audio_file = features_path / "audio_features.json"
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio features not found: {audio_file}")
        
        with open(audio_file, 'r') as f:
            audio_data = json.load(f)
        
        # Load visual features (blendshapes + pose)
        visual_file = features_path / "blendshapes_and_pose.json"
        if not visual_file.exists():
            raise FileNotFoundError(f"Visual features not found: {visual_file}")
        
        with open(visual_file, 'r') as f:
            visual_data = json.load(f)
        
        print(f"Loaded audio features: {len(audio_data['mel_features'])} frames")
        print(f"Loaded visual features: {len(visual_data['frames'])} frames")
        
        return audio_data, visual_data
    
    def synchronize_features(self, audio_data, visual_data):
        """
        Synchronize audio and visual features using timestamps
        
        Args:
            audio_data: Audio features dict
            visual_data: Visual features dict
        
        Returns:
            tuple: (synchronized_audio, synchronized_targets, metadata)
        """
        
        # Extract data arrays
        audio_features = np.array(audio_data['mel_features'])  # Shape: (time, 80)
        audio_timestamps = np.array(audio_data['timestamps'])
        audio_vad = np.array(audio_data['voice_activity'])
        
        # Extract visual features and timestamps
        visual_frames = visual_data['frames']
        visual_timestamps = np.array([frame['timestamp'] for frame in visual_frames])
        
        # Combine blendshapes and pose into targets (59 values per frame)
        targets = []
        has_face_flags = []
        
        for frame in visual_frames:
            # Combine 52 blendshapes + 7 head pose values = 59 total
            target = frame['blendshapes'] + frame['head_pose']
            targets.append(target)
            has_face_flags.append(frame['has_face'])
        
        targets = np.array(targets)  # Shape: (time, 59)
        has_face_flags = np.array(has_face_flags)
        
        print(f"\\nSynchronization details:")
        print(f"  Audio: {len(audio_features)} frames, {audio_timestamps[0]:.3f}s to {audio_timestamps[-1]:.3f}s")
        print(f"  Visual: {len(targets)} frames, {visual_timestamps[0]:.3f}s to {visual_timestamps[-1]:.3f}s")
        
        # Find common time range
        start_time = max(audio_timestamps[0], visual_timestamps[0])
        end_time = min(audio_timestamps[-1], visual_timestamps[-1])
        
        print(f"  Common range: {start_time:.3f}s to {end_time:.3f}s ({end_time-start_time:.3f}s)")
        
        # Interpolate visual features to audio timestamps
        # This upsamples visual features from 30fps to 100fps (audio frame rate)
        
        # Filter audio to common time range
        audio_mask = (audio_timestamps >= start_time) & (audio_timestamps <= end_time)
        sync_audio_timestamps = audio_timestamps[audio_mask]
        sync_audio_features = audio_features[audio_mask]
        sync_audio_vad = audio_vad[audio_mask]
        
        # Interpolate visual targets to audio timestamps
        sync_targets = np.zeros((len(sync_audio_timestamps), 59))
        sync_has_face = np.zeros(len(sync_audio_timestamps), dtype=bool)
        
        for i, target_dim in enumerate(range(59)):
            target_values = targets[:, target_dim]
            sync_targets[:, i] = np.interp(sync_audio_timestamps, visual_timestamps, target_values)
        
        # Interpolate face detection flags
        face_values = has_face_flags.astype(float)
        interpolated_face = np.interp(sync_audio_timestamps, visual_timestamps, face_values)
        sync_has_face = interpolated_face > 0.5  # Threshold for face presence
        
        print(f"\\nSynchronized dataset:")
        print(f"  Duration: {sync_audio_timestamps[-1] - sync_audio_timestamps[0]:.2f} seconds")
        print(f"  Audio features: {sync_audio_features.shape}")
        print(f"  Target features: {sync_targets.shape}")
        print(f"  Face detection rate: {np.mean(sync_has_face):.1%}")
        
        # Create metadata
        metadata = {
            'duration_seconds': float(sync_audio_timestamps[-1] - sync_audio_timestamps[0]),
            'sample_rate_hz': float(len(sync_audio_timestamps) / (sync_audio_timestamps[-1] - sync_audio_timestamps[0])),
            'num_frames': len(sync_audio_timestamps),
            'audio_features_dim': sync_audio_features.shape[1],
            'target_features_dim': sync_targets.shape[1],
            'face_detection_rate': float(np.mean(sync_has_face)),
            'voice_activity_rate': float(np.mean(sync_audio_vad))
        }
        
        return {
            'audio_features': sync_audio_features,
            'targets': sync_targets,
            'timestamps': sync_audio_timestamps,
            'voice_activity': sync_audio_vad,
            'has_face': sync_has_face,
            'metadata': metadata
        }
    
    def create_sequences(self, synchronized_data):
        """
        Create training sequences from synchronized data
        
        Args:
            synchronized_data: Output from synchronize_features()
        
        Returns:
            dict: Training sequences
        """
        audio_features = synchronized_data['audio_features']
        targets = synchronized_data['targets']
        voice_activity = synchronized_data['voice_activity']
        has_face = synchronized_data['has_face']
        timestamps = synchronized_data['timestamps']
        
        # Calculate sequence parameters
        sample_rate = synchronized_data['metadata']['sample_rate_hz']
        seq_length_frames = int(self.sequence_length_ms * sample_rate / 1000)
        step_size_frames = int((self.sequence_length_ms - self.overlap_ms) * sample_rate / 1000)
        
        print(f"\\nCreating sequences:")
        print(f"  Sequence length: {seq_length_frames} frames ({self.sequence_length_ms}ms)")
        print(f"  Step size: {step_size_frames} frames ({self.sequence_length_ms - self.overlap_ms}ms)")
        
        # Generate sequences
        sequences_audio = []
        sequences_targets = []
        sequences_vad = []
        sequences_face = []
        sequences_timestamps = []
        
        for start_idx in range(0, len(audio_features) - seq_length_frames + 1, step_size_frames):
            end_idx = start_idx + seq_length_frames
            
            # Extract sequence
            seq_audio = audio_features[start_idx:end_idx]  # Shape: (seq_len, 80)
            seq_targets = targets[start_idx:end_idx]       # Shape: (seq_len, 59)
            seq_vad = voice_activity[start_idx:end_idx]
            seq_face = has_face[start_idx:end_idx]
            seq_time = timestamps[start_idx:end_idx]
            
            # Quality checks
            face_ratio = np.mean(seq_face)
            vad_ratio = np.mean(seq_vad)
            
            # Only include sequences with reasonable face detection
            if face_ratio >= 0.5:  # At least 50% of frames have face detected
                sequences_audio.append(seq_audio)
                sequences_targets.append(seq_targets)
                sequences_vad.append(seq_vad)
                sequences_face.append(seq_face)
                sequences_timestamps.append(seq_time)
        
        sequences_audio = np.array(sequences_audio)      # Shape: (num_seq, seq_len, 80)
        sequences_targets = np.array(sequences_targets)  # Shape: (num_seq, seq_len, 59)
        sequences_vad = np.array(sequences_vad)          # Shape: (num_seq, seq_len)
        sequences_face = np.array(sequences_face)        # Shape: (num_seq, seq_len)
        
        print(f"  Generated {len(sequences_audio)} sequences")
        print(f"  Audio sequences shape: {sequences_audio.shape}")
        print(f"  Target sequences shape: {sequences_targets.shape}")
        
        return {
            'audio_sequences': sequences_audio,
            'target_sequences': sequences_targets,
            'vad_sequences': sequences_vad,
            'face_sequences': sequences_face,
            'sequence_timestamps': sequences_timestamps,
            'metadata': {
                'num_sequences': len(sequences_audio),
                'sequence_length_frames': seq_length_frames,
                'sequence_length_ms': self.sequence_length_ms,
                'step_size_frames': step_size_frames,
                'overlap_ms': self.overlap_ms,
                'audio_feature_dim': sequences_audio.shape[2],
                'target_feature_dim': sequences_targets.shape[2]
            }
        }
    
    def normalize_features(self, sequences_data, output_dir="extracted_features"):
        """
        Normalize audio features and targets
        
        Args:
            sequences_data: Output from create_sequences()
            output_dir: Directory to save normalization parameters
        
        Returns:
            dict: Normalized sequences with scalers
        """
        audio_sequences = sequences_data['audio_sequences']
        target_sequences = sequences_data['target_sequences']
        
        # Reshape for normalization (flatten time dimension)
        audio_flat = audio_sequences.reshape(-1, audio_sequences.shape[-1])
        targets_flat = target_sequences.reshape(-1, target_sequences.shape[-1])
        
        print(f"\\nNormalizing features:")
        print(f"  Audio features: {audio_flat.shape}")
        print(f"  Target features: {targets_flat.shape}")
        
        # Fit scalers
        audio_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        audio_normalized_flat = audio_scaler.fit_transform(audio_flat)
        targets_normalized_flat = target_scaler.fit_transform(targets_flat)
        
        # Reshape back to sequences
        audio_normalized = audio_normalized_flat.reshape(audio_sequences.shape)
        targets_normalized = targets_normalized_flat.reshape(target_sequences.shape)
        
        # Save scalers
        output_path = Path(output_dir)
        joblib.dump(audio_scaler, output_path / "audio_scaler.pkl")
        joblib.dump(target_scaler, output_path / "target_scaler.pkl")
        
        print(f"  Scalers saved to {output_path}")
        
        # Update sequences data
        normalized_data = sequences_data.copy()
        normalized_data['audio_sequences'] = audio_normalized
        normalized_data['target_sequences'] = targets_normalized
        normalized_data['audio_scaler'] = audio_scaler
        normalized_data['target_scaler'] = target_scaler
        
        # Add normalization stats
        normalized_data['normalization_stats'] = {
            'audio_mean': audio_scaler.mean_.tolist(),
            'audio_std': audio_scaler.scale_.tolist(),
            'target_mean': target_scaler.mean_.tolist(),
            'target_std': target_scaler.scale_.tolist()
        }
        
        return normalized_data
    
    def save_dataset(self, dataset, output_dir="extracted_features"):
        """
        Save the final dataset
        
        Args:
            dataset: Final dataset from normalize_features()
            output_dir: Directory to save dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save sequences as numpy arrays (more efficient for training)
        np.save(output_path / "audio_sequences.npy", dataset['audio_sequences'])
        np.save(output_path / "target_sequences.npy", dataset['target_sequences'])
        np.save(output_path / "vad_sequences.npy", dataset['vad_sequences'])
        np.save(output_path / "face_sequences.npy", dataset['face_sequences'])
        
        # Save metadata as JSON
        metadata = {
            'dataset_info': dataset['metadata'],
            'normalization_stats': dataset['normalization_stats']
        }
        
        with open(output_path / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\\nDataset saved to {output_path}")
        print(f"  Files: audio_sequences.npy, target_sequences.npy, dataset_metadata.json")
        
        return output_path

def main():
    """
    Main function to create the training dataset
    """
    print("Creating synchronized training dataset...")
    
    # Initialize dataset creator
    # 240ms sequences with 120ms overlap = 120ms step size
    creator = DatasetCreator(sequence_length_ms=240, overlap_ms=120)
    
    # Load features
    print("\\nLoading extracted features...")
    audio_data, visual_data = creator.load_features("extracted_features")
    
    # Synchronize features
    print("\\nSynchronizing audio and visual features...")
    synchronized_data = creator.synchronize_features(audio_data, visual_data)
    
    # Create training sequences
    print("\\nCreating training sequences...")
    sequences_data = creator.create_sequences(synchronized_data)
    
    # Normalize features
    print("\\nNormalizing features...")
    final_dataset = creator.normalize_features(sequences_data, "extracted_features")
    
    # Save dataset
    print("\\nSaving dataset...")
    output_path = creator.save_dataset(final_dataset, "extracted_features")
    
    # Print final summary
    print("\\n" + "="*60)
    print("DATASET CREATION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Number of sequences: {final_dataset['metadata']['num_sequences']}")
    print(f"Sequence length: {final_dataset['metadata']['sequence_length_ms']}ms")
    print(f"Audio features per sequence: {final_dataset['metadata']['sequence_length_frames']} x {final_dataset['metadata']['audio_feature_dim']}")
    print(f"Target features per sequence: {final_dataset['metadata']['sequence_length_frames']} x {final_dataset['metadata']['target_feature_dim']}")
    print(f"\\nReady for TCN training!")
    print("="*60)

if __name__ == "__main__":
    main()