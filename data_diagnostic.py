#!/usr/bin/env python3
"""
Comprehensive Data Quality Diagnostic Script for Audio-to-Blendshapes Training Data
Analyzes audio_sequences.npy, target_sequences.npy, and vad_sequences.npy for training viability
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

class DataQualityDiagnostic:
    def __init__(self, data_dir="multi_video_features_fixed/combined_dataset"):
        self.data_dir = Path(data_dir)
        self.report = []
        self.verdict = "UNKNOWN"
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
        # Expected ranges for different components
        self.expected_ranges = {
            'blendshapes': (0.0, 0.8),  # MediaPipe blendshapes natural range
            'jaw_open': (0.0, 0.6),    # Jaw open during speech
            'mouth_region': (0.0, 0.5), # Other mouth movements
            'pose': (-0.2, 0.2),       # Head pose movements
            'audio_mel': (-80, 10),    # Mel spectrogram range (dB)
            'vad': (0.0, 1.0)          # Voice activity detection
        }
        
        # Mouth region indices (MediaPipe blendshapes)
        self.mouth_indices = list(range(23, 52))  # Mouth-related blendshapes
        self.critical_mouth_indices = [25, 26, 27, 28]  # jawOpen, mouthClose, mouthFunnel, mouthPucker
        self.jaw_open_idx = 25  # Most important for speech
        
    def log(self, message, level="INFO"):
        """Log message with timestamp and level"""
        timestamp = f"[{level}]"
        full_message = f"{timestamp} {message}"
        print(full_message)
        self.report.append(full_message)
        
        if level == "CRITICAL":
            self.critical_issues.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
    
    def load_data(self):
        """Load and validate data files"""
        self.log("="*80)
        self.log("LOADING DATA FILES")
        self.log("="*80)
        
        try:
            # Load files
            audio_file = self.data_dir / "audio_sequences.npy"
            target_file = self.data_dir / "target_sequences.npy"
            vad_file = self.data_dir / "vad_sequences.npy"
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            if not target_file.exists():
                raise FileNotFoundError(f"Target file not found: {target_file}")
            if not vad_file.exists():
                raise FileNotFoundError(f"VAD file not found: {vad_file}")
            
            self.audio_sequences = np.load(audio_file)
            self.target_sequences = np.load(target_file)
            self.vad_sequences = np.load(vad_file)
            
            self.log(f"✅ Successfully loaded all data files")
            self.log(f"   Audio sequences: {audio_file} -> {self.audio_sequences.shape}")
            self.log(f"   Target sequences: {target_file} -> {self.target_sequences.shape}")
            self.log(f"   VAD sequences: {vad_file} -> {self.vad_sequences.shape}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to load data files: {str(e)}", "CRITICAL")
            return False
    
    def validate_shapes(self):
        """Validate data shapes and dimensions"""
        self.log("\n" + "="*80)
        self.log("VALIDATING DATA SHAPES AND DIMENSIONS")
        self.log("="*80)
        
        # Check basic shape compatibility
        if len(self.audio_sequences.shape) != 3:
            self.log(f"❌ Audio sequences should be 3D (batch, time, features), got {self.audio_sequences.shape}", "CRITICAL")
            return False
        
        if len(self.target_sequences.shape) != 3:
            self.log(f"❌ Target sequences should be 3D (batch, time, features), got {self.target_sequences.shape}", "CRITICAL")
            return False
        
        if len(self.vad_sequences.shape) != 2:
            self.log(f"❌ VAD sequences should be 2D (batch, time), got {self.vad_sequences.shape}", "CRITICAL")
            return False
        
        # Extract dimensions
        audio_batch, audio_time, audio_features = self.audio_sequences.shape
        target_batch, target_time, target_features = self.target_sequences.shape
        vad_batch, vad_time = self.vad_sequences.shape
        
        self.log(f"Audio sequences: {audio_batch} samples, {audio_time} timesteps, {audio_features} features")
        self.log(f"Target sequences: {target_batch} samples, {target_time} timesteps, {target_features} features")
        self.log(f"VAD sequences: {vad_batch} samples, {vad_time} timesteps")
        
        # Check batch size consistency
        if not (audio_batch == target_batch == vad_batch):
            self.log(f"❌ Batch sizes don't match: audio={audio_batch}, target={target_batch}, vad={vad_batch}", "CRITICAL")
            return False
        
        # Check time dimension consistency
        if not (audio_time == target_time == vad_time):
            self.log(f"❌ Time dimensions don't match: audio={audio_time}, target={target_time}, vad={vad_time}", "CRITICAL")
            return False
        
        # Check expected feature dimensions
        if audio_features != 80:
            self.log(f"⚠️  Expected 80 audio features (mel), got {audio_features}", "WARNING")
        
        if target_features != 59:
            self.log(f"⚠️  Expected 59 target features (52 blendshapes + 7 pose), got {target_features}", "WARNING")
        
        # Store dimensions
        self.batch_size = audio_batch
        self.sequence_length = audio_time
        self.audio_features = audio_features
        self.target_features = target_features
        
        self.log("✅ All shape validations passed")
        return True
    
    def analyze_audio_sequences(self):
        """Comprehensive analysis of audio sequences"""
        self.log("\n" + "="*80)
        self.log("ANALYZING AUDIO SEQUENCES")
        self.log("="*80)
        
        audio = self.audio_sequences
        
        # Basic statistics
        self.log(f"Audio data type: {audio.dtype}")
        self.log(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        self.log(f"Audio mean: {audio.mean():.3f}")
        self.log(f"Audio std: {audio.std():.3f}")
        
        # Check for expected mel spectrogram range
        expected_min, expected_max = self.expected_ranges['audio_mel']
        if audio.min() < expected_min or audio.max() > expected_max:
            self.log(f"⚠️  Audio range [{audio.min():.3f}, {audio.max():.3f}] outside expected mel range [{expected_min}, {expected_max}]", "WARNING")
        
        # Check for problematic patterns
        if audio.std() < 1.0:
            self.log(f"❌ Audio has very low variance (std={audio.std():.3f}). Features may be over-normalized or corrupted.", "CRITICAL")
        
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            self.log(f"❌ Audio contains NaN or infinite values", "CRITICAL")
        
        # Analyze dynamic range per feature
        feature_stds = np.std(audio, axis=(0, 1))
        dead_features = np.sum(feature_stds < 0.01)
        if dead_features > 0:
            self.log(f"⚠️  Found {dead_features} audio features with very low variance (< 0.01)", "WARNING")
        
        # Check for speech-like patterns
        # Calculate energy (sum across frequency bins)
        audio_energy = np.sum(audio, axis=2)  # Shape: (batch, time)
        energy_std = np.std(audio_energy)
        
        self.log(f"Audio energy std: {energy_std:.3f}")
        if energy_std < 1.0:
            self.log(f"⚠️  Low audio energy variation. May indicate poor audio extraction or silent audio.", "WARNING")
        
        # Store for later correlation analysis
        self.audio_energy = audio_energy
        
        self.log("✅ Audio sequence analysis completed")
    
    def analyze_target_sequences(self):
        """Comprehensive analysis of target sequences (blendshapes + pose)"""
        self.log("\n" + "="*80)
        self.log("ANALYZING TARGET SEQUENCES (BLENDSHAPES + POSE)")
        self.log("="*80)
        
        targets = self.target_sequences
        
        # Split blendshapes and pose
        if self.target_features >= 52:
            self.blendshapes = targets[:, :, :52]
            if self.target_features > 52:
                self.pose = targets[:, :, 52:]
            else:
                self.pose = None
        else:
            self.log(f"❌ Target sequences only have {self.target_features} features, expected at least 52 for blendshapes", "CRITICAL")
            return False
        
        self.log(f"Blendshapes shape: {self.blendshapes.shape}")
        if self.pose is not None:
            self.log(f"Pose shape: {self.pose.shape}")
        
        # Analyze blendshapes
        self._analyze_blendshapes()
        
        # Analyze pose if available
        if self.pose is not None:
            self._analyze_pose()
        
        return True
    
    def _analyze_blendshapes(self):
        """Detailed blendshape analysis"""
        self.log("\n--- BLENDSHAPE ANALYSIS ---")
        
        bs = self.blendshapes
        
        # Basic statistics
        self.log(f"Blendshapes range: [{bs.min():.4f}, {bs.max():.4f}]")
        self.log(f"Blendshapes mean: {bs.mean():.4f}")
        self.log(f"Blendshapes std: {bs.std():.4f}")
        
        # Check expected range
        expected_min, expected_max = self.expected_ranges['blendshapes']
        if bs.max() < 0.1:
            self.log(f"❌ CRITICAL: Blendshape maximum ({bs.max():.4f}) is too small! Expected range: [{expected_min}, {expected_max}]", "CRITICAL")
            self.log("   This means facial movements will be barely visible. Model will never learn proper animation.", "CRITICAL")
        elif bs.max() < 0.3:
            self.log(f"⚠️  Blendshape maximum ({bs.max():.4f}) is quite small. May result in subtle movements.", "WARNING")
        
        # Analyze individual blendshape channels
        channel_stats = []
        dead_channels = []
        
        for i in range(52):
            channel = bs[:, :, i]
            channel_std = np.std(channel)
            channel_max = np.max(channel)
            
            if channel_std < 0.001:
                dead_channels.append(i)
            
            channel_stats.append({
                'index': i,
                'min': np.min(channel),
                'max': channel_max,
                'mean': np.mean(channel),
                'std': channel_std
            })
        
        if dead_channels:
            self.log(f"⚠️  Found {len(dead_channels)} inactive blendshape channels: {dead_channels}", "WARNING")
        
        # Focus on mouth region
        self._analyze_mouth_region()
        
        # Analyze critical mouth movements
        self._analyze_critical_mouth_movements()
    
    def _analyze_mouth_region(self):
        """Analyze mouth region blendshapes specifically"""
        self.log("\n--- MOUTH REGION ANALYSIS ---")
        
        mouth_bs = self.blendshapes[:, :, self.mouth_indices]  # Indices 23-51
        
        self.log(f"Mouth blendshapes range: [{mouth_bs.min():.4f}, {mouth_bs.max():.4f}]")
        self.log(f"Mouth blendshapes std: {mouth_bs.std():.4f}")
        
        expected_min, expected_max = self.expected_ranges['mouth_region']
        if mouth_bs.max() < 0.2:
            self.log(f"❌ CRITICAL: Mouth movements are too small (max={mouth_bs.max():.4f}). Expected > 0.2 for visible speech.", "CRITICAL")
        
        # Calculate mouth movement variance per sequence
        mouth_variance_per_seq = np.var(mouth_bs, axis=1).mean(axis=1)  # Average variance across mouth channels per sequence
        low_variance_sequences = np.sum(mouth_variance_per_seq < 0.001)
        
        if low_variance_sequences > self.batch_size * 0.5:
            self.log(f"❌ {low_variance_sequences}/{self.batch_size} sequences have very low mouth variance. Training data lacks diversity.", "CRITICAL")
        
        self.mouth_blendshapes = mouth_bs
    
    def _analyze_critical_mouth_movements(self):
        """Analyze the most important mouth movements for speech"""
        self.log("\n--- CRITICAL MOUTH MOVEMENTS ANALYSIS ---")
        
        # Jaw open (most important for speech)
        jaw_open = self.blendshapes[:, :, self.jaw_open_idx]
        self.log(f"Jaw open (index {self.jaw_open_idx}):")
        self.log(f"  Range: [{jaw_open.min():.4f}, {jaw_open.max():.4f}]")
        self.log(f"  Std: {jaw_open.std():.4f}")
        
        expected_min, expected_max = self.expected_ranges['jaw_open']
        if jaw_open.max() < 0.2:
            self.log(f"❌ CRITICAL: Jaw opening is too small (max={jaw_open.max():.4f}). Expected > 0.2 for clear speech.", "CRITICAL")
        
        if jaw_open.std() < 0.02:
            self.log(f"❌ CRITICAL: Jaw movement variation is too low (std={jaw_open.std():.4f}). Need more diverse speech data.", "CRITICAL")
        
        # Other critical mouth movements
        for idx in self.critical_mouth_indices:
            if idx < self.blendshapes.shape[2]:
                movement = self.blendshapes[:, :, idx]
                movement_name = f"blendshape_{idx}"
                self.log(f"{movement_name}: range=[{movement.min():.4f}, {movement.max():.4f}], std={movement.std():.4f}")
        
        self.jaw_open_data = jaw_open
    
    def _analyze_pose(self):
        """Analyze head pose data"""
        self.log("\n--- HEAD POSE ANALYSIS ---")
        
        pose = self.pose
        self.log(f"Pose range: [{pose.min():.4f}, {pose.max():.4f}]")
        self.log(f"Pose std: {pose.std():.4f}")
        
        expected_min, expected_max = self.expected_ranges['pose']
        if pose.min() < expected_min or pose.max() > expected_max:
            self.log(f"⚠️  Pose range [{pose.min():.4f}, {pose.max():.4f}] outside expected range [{expected_min}, {expected_max}]", "WARNING")
    
    def analyze_vad_sequences(self):
        """Analyze Voice Activity Detection sequences"""
        self.log("\n" + "="*80)
        self.log("ANALYZING VAD SEQUENCES")
        self.log("="*80)
        
        vad = self.vad_sequences
        
        self.log(f"VAD range: [{vad.min():.3f}, {vad.max():.3f}]")
        self.log(f"VAD mean: {vad.mean():.3f}")
        self.log(f"VAD std: {vad.std():.3f}")
        
        # Check VAD range
        expected_min, expected_max = self.expected_ranges['vad']
        if vad.min() < expected_min or vad.max() > expected_max:
            self.log(f"⚠️  VAD values outside expected range [{expected_min}, {expected_max}]", "WARNING")
        
        # Analyze speech activity
        speech_ratio = np.mean(vad > 0.5)
        self.log(f"Speech activity ratio: {speech_ratio:.3f} ({speech_ratio*100:.1f}% of time)")
        
        if speech_ratio < 0.2:
            self.log(f"⚠️  Low speech activity ({speech_ratio*100:.1f}%). Most data is silence.", "WARNING")
        elif speech_ratio > 0.8:
            self.log(f"⚠️  Very high speech activity ({speech_ratio*100:.1f}%). May lack silence examples.", "WARNING")
        
        # Check for binary vs continuous VAD
        unique_values = len(np.unique(vad))
        if unique_values == 2:
            self.log("VAD appears to be binary (0/1)")
        elif unique_values < 10:
            self.log(f"VAD has few unique values ({unique_values}), may be quantized")
        else:
            self.log(f"VAD appears continuous with {unique_values} unique values")
        
        self.vad_data = vad
        self.speech_ratio = speech_ratio
        
        self.log("✅ VAD sequence analysis completed")
    
    def analyze_correlations(self):
        """Analyze correlations between audio, VAD, and mouth movements"""
        self.log("\n" + "="*80)
        self.log("ANALYZING CROSS-MODAL CORRELATIONS")
        self.log("="*80)
        
        # We need to detect alignment issues between audio and visual features
        self._analyze_audio_visual_correlation()
        self._analyze_vad_mouth_correlation()
        self._analyze_temporal_alignment()
    
    def _analyze_audio_visual_correlation(self):
        """Check correlation between audio energy and mouth movements"""
        self.log("\n--- AUDIO-VISUAL CORRELATION ---")
        
        # Flatten data for correlation analysis
        audio_energy_flat = self.audio_energy.flatten()
        jaw_open_flat = self.jaw_open_data.flatten()
        mouth_movement_flat = np.mean(self.mouth_blendshapes, axis=2).flatten()
        
        # Calculate correlations
        audio_jaw_corr = np.corrcoef(audio_energy_flat, jaw_open_flat)[0, 1]
        audio_mouth_corr = np.corrcoef(audio_energy_flat, mouth_movement_flat)[0, 1]
        
        self.log(f"Audio energy vs Jaw opening correlation: {audio_jaw_corr:.3f}")
        self.log(f"Audio energy vs Mouth movement correlation: {audio_mouth_corr:.3f}")
        
        # Check correlation quality
        if abs(audio_jaw_corr) < 0.1:
            self.log(f"❌ CRITICAL: Very low audio-jaw correlation ({audio_jaw_corr:.3f}). Possible alignment issues or poor data quality.", "CRITICAL")
        elif abs(audio_jaw_corr) < 0.3:
            self.log(f"⚠️  Low audio-jaw correlation ({audio_jaw_corr:.3f}). May indicate alignment or quality issues.", "WARNING")
        else:
            self.log(f"✅ Good audio-jaw correlation ({audio_jaw_corr:.3f})")
        
        if abs(audio_mouth_corr) < 0.1:
            self.log(f"❌ CRITICAL: Very low audio-mouth correlation ({audio_mouth_corr:.3f}). Likely alignment or extraction issues.", "CRITICAL")
        elif abs(audio_mouth_corr) < 0.3:
            self.log(f"⚠️  Low audio-mouth correlation ({audio_mouth_corr:.3f}). May indicate issues.", "WARNING")
        else:
            self.log(f"✅ Good audio-mouth correlation ({audio_mouth_corr:.3f})")
        
        self.audio_jaw_correlation = audio_jaw_corr
        self.audio_mouth_correlation = audio_mouth_corr
    
    def _analyze_vad_mouth_correlation(self):
        """Check correlation between VAD and mouth movements"""
        self.log("\n--- VAD-MOUTH CORRELATION ---")
        
        # Flatten data
        vad_flat = self.vad_data.flatten()
        jaw_open_flat = self.jaw_open_data.flatten()
        mouth_movement_flat = np.mean(self.mouth_blendshapes, axis=2).flatten()
        
        # Calculate correlations
        vad_jaw_corr = np.corrcoef(vad_flat, jaw_open_flat)[0, 1]
        vad_mouth_corr = np.corrcoef(vad_flat, mouth_movement_flat)[0, 1]
        
        self.log(f"VAD vs Jaw opening correlation: {vad_jaw_corr:.3f}")
        self.log(f"VAD vs Mouth movement correlation: {vad_mouth_corr:.3f}")
        
        if abs(vad_jaw_corr) < 0.2:
            self.log(f"⚠️  Low VAD-jaw correlation ({vad_jaw_corr:.3f}). VAD may not accurately reflect speech activity.", "WARNING")
        
        if abs(vad_mouth_corr) < 0.2:
            self.log(f"⚠️  Low VAD-mouth correlation ({vad_mouth_corr:.3f}). Possible VAD accuracy issues.", "WARNING")
    
    def _analyze_temporal_alignment(self):
        """Check for temporal alignment issues"""
        self.log("\n--- TEMPORAL ALIGNMENT ANALYSIS ---")
        
        # We cannot fully detect alignment issues without ground truth,
        # but we can look for suspicious patterns
        
        # Check if mouth movements lag behind audio (common in extraction)
        # Calculate cross-correlation at different lags
        audio_energy_sample = self.audio_energy[0]  # Use first sequence
        jaw_open_sample = self.jaw_open_data[0]
        
        # Normalize for cross-correlation
        audio_norm = (audio_energy_sample - np.mean(audio_energy_sample)) / np.std(audio_energy_sample)
        jaw_norm = (jaw_open_sample - np.mean(jaw_open_sample)) / np.std(jaw_open_sample)
        
        # Cross-correlation
        cross_corr = correlate(audio_norm, jaw_norm, mode='full')
        lags = np.arange(-len(jaw_norm) + 1, len(audio_norm))
        
        # Find best lag
        best_lag_idx = np.argmax(cross_corr)
        best_lag = lags[best_lag_idx]
        best_correlation = cross_corr[best_lag_idx] / len(jaw_norm)  # Normalize
        
        self.log(f"Cross-correlation analysis (sample sequence):")
        self.log(f"  Best lag: {best_lag} frames")
        self.log(f"  Best correlation: {best_correlation:.3f}")
        
        if abs(best_lag) > 5:
            self.log(f"⚠️  Significant lag detected ({best_lag} frames). Possible alignment issues during extraction.", "WARNING")
            self.log(f"     Recommendation: Check audio-visual synchronization in original videos.")
        
        # Check for consistent timing patterns across sequences
        lag_variations = []
        for i in range(min(5, self.batch_size)):  # Check first 5 sequences
            audio_seq = self.audio_energy[i]
            jaw_seq = self.jaw_open_data[i]
            
            # Normalize
            if np.std(audio_seq) > 0 and np.std(jaw_seq) > 0:
                audio_norm = (audio_seq - np.mean(audio_seq)) / np.std(audio_seq)
                jaw_norm = (jaw_seq - np.mean(jaw_seq)) / np.std(jaw_seq)
                
                cross_corr = correlate(audio_norm, jaw_norm, mode='full')
                lags = np.arange(-len(jaw_norm) + 1, len(audio_norm))
                best_lag = lags[np.argmax(cross_corr)]
                lag_variations.append(best_lag)
        
        if lag_variations:
            lag_std = np.std(lag_variations)
            self.log(f"Lag variation across sequences: {lag_std:.1f} frames std")
            if lag_std > 3:
                self.log(f"⚠️  High lag variation ({lag_std:.1f} frames). Inconsistent alignment across sequences.", "WARNING")
    
    def detect_normalization_issues(self):
        """Detect over-normalization or improper scaling"""
        self.log("\n" + "="*80)
        self.log("DETECTING NORMALIZATION ISSUES")
        self.log("="*80)
        
        # Check blendshapes for over-normalization
        bs = self.blendshapes
        
        # Check if all values are in a very narrow range (over-normalization)
        bs_range = bs.max() - bs.min()
        if bs_range < 0.1:
            self.log(f"❌ CRITICAL: Blendshapes have very narrow range ({bs_range:.4f}). Likely over-normalized.", "CRITICAL")
            self.log("   This will prevent the model from learning meaningful facial movements.", "CRITICAL")
        
        # Check for quantization artifacts
        unique_values = len(np.unique(bs))
        total_values = bs.size
        if unique_values < total_values * 0.01:  # Less than 1% unique values
            self.log(f"⚠️  Blendshapes appear heavily quantized ({unique_values} unique values out of {total_values})", "WARNING")
        
        # Check for unnatural distributions
        # Natural blendshapes should have exponential-like distribution (most values near 0)
        hist, bins = np.histogram(bs.flatten(), bins=50)
        peak_bin = np.argmax(hist)
        
        if peak_bin > 25:  # Peak is not near zero
            self.log(f"⚠️  Blendshape distribution peak is not near zero (bin {peak_bin}/50). May indicate improper normalization.", "WARNING")
        
        # Check audio normalization
        audio = self.audio_sequences
        audio_range = audio.max() - audio.min()
        if audio_range < 1.0:
            self.log(f"⚠️  Audio has very small range ({audio_range:.3f}). May be over-normalized.", "WARNING")
        
        self.log("✅ Normalization analysis completed")
    
    def generate_final_verdict(self):
        """Generate final verdict on data quality"""
        self.log("\n" + "="*80)
        self.log("FINAL VERDICT AND RECOMMENDATIONS")
        self.log("="*80)
        
        # Count issues
        num_critical = len(self.critical_issues)
        num_warnings = len(self.warnings)
        
        self.log(f"Issues found: {num_critical} critical, {num_warnings} warnings")
        
        if num_critical > 0:
            self.verdict = "POOR - NOT SUITABLE FOR TRAINING"
            self.log(f"❌ VERDICT: {self.verdict}", "CRITICAL")
            self.log("❌ This data will NOT produce a working mouth animation model.", "CRITICAL")
        elif num_warnings > 3:
            self.verdict = "MARGINAL - MAY HAVE ISSUES"
            self.log(f"⚠️  VERDICT: {self.verdict}", "WARNING")
            self.log("⚠️  This data may produce poor quality mouth animation.")
        else:
            self.verdict = "GOOD - SUITABLE FOR TRAINING"
            self.log(f"✅ VERDICT: {self.verdict}")
            self.log("✅ This data should work for training a mouth animation model.")
        
        # Generate specific recommendations
        self._generate_recommendations()
        
        # Print summary
        self.log(f"\nSUMMARY:")
        self.log(f"  Data quality: {self.verdict}")
        self.log(f"  Critical issues: {num_critical}")
        self.log(f"  Warnings: {num_warnings}")
        self.log(f"  Recommendations: {len(self.recommendations)}")
    
    def _generate_recommendations(self):
        """Generate specific recommendations based on issues found"""
        self.log("\n--- RECOMMENDATIONS ---")
        
        if len(self.critical_issues) > 0:
            self.recommendations.append("CRITICAL: Fix all critical issues before attempting to train")
            
        # Specific recommendations based on common issues
        if hasattr(self, 'blendshapes') and self.blendshapes.max() < 0.1:
            self.recommendations.append("Re-extract blendshapes with proper scaling. Current values are too small.")
            self.recommendations.append("Check MediaPipe extraction parameters and video quality.")
        
        if hasattr(self, 'audio_jaw_correlation') and abs(self.audio_jaw_correlation) < 0.2:
            self.recommendations.append("Check audio-visual synchronization in original videos.")
            self.recommendations.append("Consider re-extracting features with better alignment.")
        
        if hasattr(self, 'speech_ratio') and self.speech_ratio < 0.3:
            self.recommendations.append("Add more speech data. Current dataset has too much silence.")
        
        # Print all recommendations
        for i, rec in enumerate(self.recommendations, 1):
            self.log(f"{i}. {rec}")
        
        if not self.recommendations:
            self.log("No specific recommendations - data appears to be in good shape!")
    
    def save_detailed_report(self):
        """Save detailed report to file"""
        report_file = self.data_dir / "data_quality_report.txt"
        
        self.log(f"\n💾 Saving detailed report to: {report_file}")
        
        try:
            with open(report_file, 'w') as f:
                f.write("COMPREHENSIVE DATA QUALITY DIAGNOSTIC REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated for: {self.data_dir}\n")
                f.write(f"Final Verdict: {self.verdict}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write full log
                for line in self.report:
                    f.write(line + "\n")
                
                # Write summary section
                f.write("\n" + "=" * 80 + "\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Overall Assessment: {self.verdict}\n")
                f.write(f"Critical Issues: {len(self.critical_issues)}\n")
                f.write(f"Warnings: {len(self.warnings)}\n\n")
                
                if self.critical_issues:
                    f.write("CRITICAL ISSUES:\n")
                    for i, issue in enumerate(self.critical_issues, 1):
                        f.write(f"{i}. {issue}\n")
                    f.write("\n")
                
                if self.warnings:
                    f.write("WARNINGS:\n")
                    for i, warning in enumerate(self.warnings, 1):
                        f.write(f"{i}. {warning}\n")
                    f.write("\n")
                
                if self.recommendations:
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(self.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                # Technical details
                f.write("TECHNICAL DETAILS:\n")
                f.write(f"  Dataset shape: {self.batch_size} sequences, {self.sequence_length} timesteps\n")
                f.write(f"  Audio features: {self.audio_features}\n")
                f.write(f"  Target features: {self.target_features}\n")
                
                if hasattr(self, 'blendshapes'):
                    f.write(f"  Blendshape range: [{self.blendshapes.min():.4f}, {self.blendshapes.max():.4f}]\n")
                
                if hasattr(self, 'jaw_open_data'):
                    f.write(f"  Jaw open range: [{self.jaw_open_data.min():.4f}, {self.jaw_open_data.max():.4f}]\n")
                    f.write(f"  Jaw open std: {self.jaw_open_data.std():.4f}\n")
                
                if hasattr(self, 'audio_jaw_correlation'):
                    f.write(f"  Audio-jaw correlation: {self.audio_jaw_correlation:.3f}\n")
                
                if hasattr(self, 'speech_ratio'):
                    f.write(f"  Speech ratio: {self.speech_ratio:.3f}\n")
                
            self.log(f"✅ Report saved successfully")
            
        except Exception as e:
            self.log(f"❌ Failed to save report: {str(e)}", "CRITICAL")
    
    def create_visualization_plots(self):
        """Create visualization plots for key metrics"""
        self.log(f"\n📊 Creating visualization plots...")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Data Quality Diagnostic Visualizations', fontsize=16)
            
            # Plot 1: Blendshape distribution
            if hasattr(self, 'blendshapes'):
                axes[0, 0].hist(self.blendshapes.flatten(), bins=50, alpha=0.7, color='blue')
                axes[0, 0].set_title('Blendshape Value Distribution')
                axes[0, 0].set_xlabel('Blendshape Value')
                axes[0, 0].set_ylabel('Frequency')
            
            # Plot 2: Jaw open over time (sample sequence)
            if hasattr(self, 'jaw_open_data'):
                sample_seq = min(0, self.batch_size - 1)
                axes[0, 1].plot(self.jaw_open_data[sample_seq])
                axes[0, 1].set_title('Jaw Open Movement (Sample Sequence)')
                axes[0, 1].set_xlabel('Time Frame')
                axes[0, 1].set_ylabel('Jaw Open Value')
            
            # Plot 3: Audio energy over time (sample sequence)
            if hasattr(self, 'audio_energy'):
                sample_seq = min(0, self.batch_size - 1)
                axes[0, 2].plot(self.audio_energy[sample_seq])
                axes[0, 2].set_title('Audio Energy (Sample Sequence)')
                axes[0, 2].set_xlabel('Time Frame')
                axes[0, 2].set_ylabel('Audio Energy')
            
            # Plot 4: VAD over time (sample sequence)
            if hasattr(self, 'vad_data'):
                sample_seq = min(0, self.batch_size - 1)
                axes[1, 0].plot(self.vad_data[sample_seq])
                axes[1, 0].set_title('Voice Activity Detection (Sample)')
                axes[1, 0].set_xlabel('Time Frame')
                axes[1, 0].set_ylabel('VAD Value')
            
            # Plot 5: Audio-Visual correlation scatter
            if hasattr(self, 'audio_energy') and hasattr(self, 'jaw_open_data'):
                sample_seq = min(0, self.batch_size - 1)
                axes[1, 1].scatter(self.audio_energy[sample_seq], self.jaw_open_data[sample_seq], alpha=0.5)
                axes[1, 1].set_title('Audio Energy vs Jaw Open')
                axes[1, 1].set_xlabel('Audio Energy')
                axes[1, 1].set_ylabel('Jaw Open')
            
            # Plot 6: Blendshape channel variance
            if hasattr(self, 'blendshapes'):
                channel_vars = np.var(self.blendshapes, axis=(0, 1))
                axes[1, 2].bar(range(len(channel_vars)), channel_vars)
                axes[1, 2].set_title('Blendshape Channel Variance')
                axes[1, 2].set_xlabel('Blendshape Index')
                axes[1, 2].set_ylabel('Variance')
                # Highlight mouth region
                axes[1, 2].axvline(x=23, color='red', linestyle='--', alpha=0.7, label='Mouth Start')
                axes[1, 2].axvline(x=51, color='red', linestyle='--', alpha=0.7, label='Mouth End')
                axes[1, 2].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.data_dir / "data_quality_plots.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"✅ Plots saved to: {plot_file}")
            
        except Exception as e:
            self.log(f"⚠️  Could not create plots: {str(e)}", "WARNING")
    
    def run_full_diagnostic(self):
        """Run the complete diagnostic pipeline"""
        self.log("🚀 STARTING COMPREHENSIVE DATA QUALITY DIAGNOSTIC")
        self.log("=" * 80)
        
        # Step 1: Load data
        if not self.load_data():
            self.log("❌ Failed to load data. Stopping diagnostic.", "CRITICAL")
            return False
        
        # Step 2: Validate shapes
        if not self.validate_shapes():
            self.log("❌ Shape validation failed. Stopping diagnostic.", "CRITICAL")
            return False
        
        # Step 3: Analyze each component
        self.analyze_audio_sequences()
        
        if not self.analyze_target_sequences():
            self.log("❌ Target sequence analysis failed. Stopping diagnostic.", "CRITICAL")
            return False
        
        self.analyze_vad_sequences()
        
        # Step 4: Cross-modal analysis
        self.analyze_correlations()
        
        # Step 5: Detect normalization issues
        self.detect_normalization_issues()
        
        # Step 6: Generate verdict and recommendations
        self.generate_final_verdict()
        
        # Step 7: Save results
        self.save_detailed_report()
        self.create_visualization_plots()
        
        self.log("\n🎉 DIAGNOSTIC COMPLETED!")
        self.log("=" * 80)
        self.log(f"📋 Detailed report saved to: {self.data_dir / 'data_quality_report.txt'}")
        self.log(f"📊 Visualization plots saved to: {self.data_dir / 'data_quality_plots.png'}")
        
        return True


def main():
    """Main function to run the diagnostic"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Data Quality Diagnostic for Audio-to-Blendshapes Training Data')
    parser.add_argument('--data_dir', type=str, default='multi_video_features_fixed/combined_dataset',
                        help='Directory containing the .npy data files')
    
    args = parser.parse_args()
    
    # Create diagnostic instance
    diagnostic = DataQualityDiagnostic(args.data_dir)
    
    # Run full diagnostic
    success = diagnostic.run_full_diagnostic()
    
    if success:
        print(f"\n{'='*80}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'='*80}")
        print(f"Final Verdict: {diagnostic.verdict}")
        print(f"Critical Issues: {len(diagnostic.critical_issues)}")
        print(f"Warnings: {len(diagnostic.warnings)}")
        
        if diagnostic.verdict.startswith("POOR"):
            print("\n❌ TRAINING NOT RECOMMENDED with this data")
            print("Fix critical issues before attempting to train your model.")
            return 1
        elif diagnostic.verdict.startswith("MARGINAL"):
            print("\n⚠️  TRAINING MAY HAVE ISSUES with this data")
            print("Consider addressing warnings for better results.")
            return 0
        else:
            print("\n✅ DATA LOOKS GOOD FOR TRAINING")
            print("You should be able to train a working model with this data.")
            return 0
    else:
        print("\n❌ DIAGNOSTIC FAILED")
        print("Could not complete data quality analysis.")
        return 1


if __name__ == "__main__":
    exit(main())