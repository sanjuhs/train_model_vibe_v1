#!/usr/bin/env python3
"""
Video Analysis Script - Analyze video properties and quality
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def analyze_video(video_path):
    """
    Analyze video file properties and quality
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Get file size
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    
    # Sample frames to check quality
    sample_frames = []
    frame_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
    
    cap.release()
    
    # Analyze frame quality
    avg_brightness = 0
    avg_contrast = 0
    if sample_frames:
        brightnesses = []
        contrasts = []
        for frame in sample_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(np.mean(gray))
            contrasts.append(np.std(gray))
        
        avg_brightness = np.mean(brightnesses)
        avg_contrast = np.mean(contrasts)
    
    # Compile results
    analysis = {
        'file_path': video_path,
        'file_size_mb': file_size,
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration_seconds': duration,
        'duration_minutes': duration / 60,
        'resolution': f"{width}x{height}",
        'avg_brightness': avg_brightness,
        'avg_contrast': avg_contrast,
        'bitrate_mbps': (file_size * 8) / duration if duration > 0 else 0,
    }
    
    return analysis

def print_analysis_report(analysis):
    """
    Print a formatted analysis report
    """
    if not analysis:
        return
    
    print("="*60)
    print("VIDEO ANALYSIS REPORT")
    print("="*60)
    print(f"File: {analysis['file_path']}")
    print(f"File Size: {analysis['file_size_mb']:.2f} MB")
    print(f"Duration: {analysis['duration_minutes']:.2f} minutes ({analysis['duration_seconds']:.1f} seconds)")
    print(f"Resolution: {analysis['resolution']}")
    print(f"Frame Rate: {analysis['fps']:.2f} FPS")
    print(f"Total Frames: {analysis['frame_count']}")
    print(f"Estimated Bitrate: {analysis['bitrate_mbps']:.2f} Mbps")
    print()
    print("QUALITY METRICS:")
    print(f"Average Brightness: {analysis['avg_brightness']:.1f} (0-255)")
    print(f"Average Contrast: {analysis['avg_contrast']:.1f}")
    print()
    
    # Quality recommendations
    print("RECOMMENDATIONS:")
    
    # Check FPS consistency
    if analysis['fps'] < 25:
        print("⚠️  Low frame rate detected. Consider using 25-30 FPS for better model training.")
    elif analysis['fps'] > 35:
        print("ℹ️  High frame rate detected. You may downsample to 25-30 FPS to reduce computational load.")
    else:
        print("✅ Frame rate is suitable for training (25-30 FPS range).")
    
    # Check resolution
    if analysis['width'] < 640 or analysis['height'] < 480:
        print("⚠️  Low resolution detected. Higher resolution may improve face tracking accuracy.")
    else:
        print("✅ Resolution is adequate for face tracking.")
    
    # Check brightness
    if analysis['avg_brightness'] < 80:
        print("⚠️  Video appears dark. Consider brightness adjustment for better face detection.")
    elif analysis['avg_brightness'] > 200:
        print("⚠️  Video appears overexposed. Consider brightness adjustment.")
    else:
        print("✅ Brightness levels appear good.")
    
    # Check contrast
    if analysis['avg_contrast'] < 20:
        print("⚠️  Low contrast detected. May affect feature extraction quality.")
    else:
        print("✅ Contrast levels appear adequate.")
    
    print("="*60)

def main():
    """
    Main function to analyze the video
    """
    # Video file path
    video_path = "videodata/ml_sanjay_assortmentSounds55_15min_dataset.mp4"
    
    print("Starting video analysis...")
    analysis = analyze_video(video_path)
    
    if analysis:
        print_analysis_report(analysis)
        
        # Save analysis to file
        import json
        output_file = "data_preparation_scripts/video_analysis_report.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {output_file}")
    else:
        print("Failed to analyze video.")
        sys.exit(1)

if __name__ == "__main__":
    main()