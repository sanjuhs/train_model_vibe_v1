#!/usr/bin/env python3
"""
Multi-Video Data Processor
Processes multiple videos for larger training datasets with moviepy and enhanced error handling
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp_core
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import traceback
import time

# Try to import moviepy, use fallback if not available
try:
    from moviepy.editor import VideoFileClip
    HAS_MOVIEPY = True
    print("✅ MoviePy available for video validation")
except ImportError:
    HAS_MOVIEPY = False
    print("⚠️ MoviePy not available, using basic validation")

# Import our existing processors
import sys
from pathlib import Path

# Add both current directory and parent to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.extend([str(current_dir), str(parent_dir)])

# Import processors with proper error handling
try:
    exec(open(current_dir / "02_extract_blendshapes.py").read())
    exec(open(current_dir / "03_extract_audio_features.py").read())
    exec(open(current_dir / "04_create_dataset.py").read())
    print("✅ Successfully imported all processing modules")
except Exception as e:
    print(f"❌ Error importing modules: {e}")
    raise

class MultiVideoProcessor:
    """
    Process multiple videos efficiently with parallel processing
    """
    
    def __init__(self, 
                 video_directory="videodata",
                 output_directory="multi_video_features",
                 max_workers=None):
        """
        Initialize multi-video processor
        
        Args:
            video_directory: Directory containing video files
            output_directory: Output directory for processed features
            max_workers: Number of parallel workers (None = auto)
        """
        self.video_directory = Path(video_directory)
        self.output_directory = Path(output_directory)
        self.max_workers = max_workers or min(mp_core.cpu_count(), 4)
        
        print(f"Multi-Video Processor initialized:")
        print(f"  Video directory: {self.video_directory}")
        print(f"  Output directory: {self.output_directory}")
        print(f"  Max workers: {self.max_workers}")
    
    def validate_video(self, video_path):
        """
        Validate video file using moviepy
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict with validation results
        """
        try:
            clip = VideoFileClip(str(video_path))
            info = {
                'valid': True,
                'duration': clip.duration,
                'fps': clip.fps,
                'size': clip.size,
                'has_audio': clip.audio is not None
            }
            clip.close()
            return info
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def find_videos(self, video_extensions=None):
        """
        Find all video files in the directory with validation
        
        Args:
            video_extensions: List of video extensions to look for
            
        Returns:
            List of validated video file paths
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if not self.video_directory.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_directory}")
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.video_directory.glob(f"*{ext}")))
            video_files.extend(list(self.video_directory.glob(f"*{ext.upper()}")))
        
        video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
        
        print(f"Found {len(video_files)} potential video files. Validating...")
        
        valid_videos = []
        for i, video_file in enumerate(video_files, 1):
            size_gb = video_file.stat().st_size / (1024**3)
            print(f"  {i}. {video_file.name} ({size_gb:.1f} GB) - ", end="")
            
            validation = self.validate_video(video_file)
            if validation['valid']:
                print(f"✅ Valid ({validation['duration']:.1f}s, {validation['fps']:.1f}fps)")
                valid_videos.append(video_file)
            else:
                print(f"❌ Invalid: {validation['error']}")
        
        print(f"\n✅ {len(valid_videos)} valid video files found")
        return valid_videos
    
    def process_single_video(self, video_path, video_index=0):
        """
        Process a single video file with enhanced error handling and moviepy validation
        
        Args:
            video_path: Path to video file
            video_index: Index of video for naming
            
        Returns:
            Dict with processing results
        """
        video_name = video_path.stem
        video_output_dir = self.output_directory / f"video_{video_index:02d}_{video_name}"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Processing video {video_index + 1}: {video_name}")
        print(f"{'='*60}")
        
        # Clean up any temp files from previous runs
        temp_files = list(video_output_dir.glob("temp_*.py"))
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                print(f"  🧹 Cleaned up {temp_file.name}")
            except:
                pass
        
        try:
            # Pre-validate video with moviepy
            print("  📹 Validating video file...")
            validation = self.validate_video(video_path)
            if not validation['valid']:
                raise ValueError(f"Invalid video file: {validation['error']}")
            
            print(f"    Duration: {validation['duration']:.1f}s")
            print(f"    FPS: {validation['fps']:.1f}")
            print(f"    Has audio: {validation['has_audio']}")
            
            # Step 1: Extract blendshapes and pose
            print("  🎭 Step 1: Extracting blendshapes and head pose...")
            blendshape_extractor = FaceBlendshapeExtractor()
            
            blendshape_data = blendshape_extractor.extract_from_video(
                str(video_path),
                str(video_output_dir),
                max_frames=None  # Process entire video
            )
            
            print(f"    ✅ Extracted {len(blendshape_data['frames'])} frames")
            print(f"    🎯 Face detection rate: {(1-blendshape_data['failure_rate'])*100:.1f}%")
            
            # Verify blendshapes file exists
            blendshape_file = video_output_dir / "blendshapes_and_pose.json"
            if not blendshape_file.exists():
                raise FileNotFoundError("Blendshapes extraction failed - no output file created")
            
            # Step 2: Extract audio features
            print("  🔊 Step 2: Extracting audio features...")
            audio_extractor = AudioFeatureExtractor()
            
            audio_data = audio_extractor.extract_from_video(
                str(video_path),
                str(video_output_dir),
                max_duration=None  # Process entire audio
            )
            
            print(f"    ✅ Extracted {audio_data['n_frames']} audio frames")
            print(f"    ⏱️ Duration: {audio_data['duration_seconds']:.1f} seconds")
            
            # Verify audio file exists
            audio_file = video_output_dir / "audio_features.json"
            if not audio_file.exists():
                raise FileNotFoundError("Audio extraction failed - no output file created")
            
            # Step 3: Create synchronized dataset
            print("  🔄 Step 3: Creating synchronized dataset...")
            dataset_creator = DatasetCreator(
                sequence_length_ms=240,  # 240ms sequences
                overlap_ms=120          # 120ms overlap
            )
            
            # Load the extracted features
            with open(audio_file, 'r') as f:
                audio_features = json.load(f)
            
            with open(blendshape_file, 'r') as f:
                visual_features = json.load(f)
            
            # Synchronize features
            synchronized_data = dataset_creator.synchronize_features(audio_features, visual_features)
            
            # Create sequences
            sequences_data = dataset_creator.create_sequences(synchronized_data)
            
            # Normalize and save
            final_dataset = dataset_creator.normalize_features(sequences_data, str(video_output_dir))
            dataset_creator.save_dataset(final_dataset, str(video_output_dir))
            
            print(f"    ✅ Created {sequences_data['metadata']['num_sequences']} sequences")
            
            # Verify dataset files exist
            required_files = ["audio_sequences.npy", "target_sequences.npy", "vad_sequences.npy"]
            for file_name in required_files:
                if not (video_output_dir / file_name).exists():
                    raise FileNotFoundError(f"Dataset creation failed - {file_name} not created")
            
            processing_time = time.time() - start_time
            
            # Save video processing summary
            processing_summary = {
                'video_path': str(video_path),
                'video_name': video_name,
                'video_index': video_index,
                'duration_seconds': validation['duration'],
                'fps': validation['fps'],
                'total_video_frames': len(blendshape_data['frames']),
                'total_audio_frames': audio_data['n_frames'],
                'face_detection_rate': 1 - blendshape_data['failure_rate'],
                'voice_activity_rate': np.mean(audio_data['voice_activity']),
                'num_sequences': sequences_data['metadata']['num_sequences'],
                'sequence_length_ms': sequences_data['metadata']['sequence_length_ms'],
                'processing_time_seconds': processing_time,
                'processing_status': 'success',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(video_output_dir / "processing_summary.json", 'w') as f:
                json.dump(processing_summary, f, indent=2)
            
            print(f"  🎉 Video {video_index + 1} processed successfully in {processing_time:.1f}s!")
            return processing_summary
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(f"  ❌ Error processing video {video_index + 1}: {e}")
            print(f"  📝 Full error trace saved to processing_summary.json")
            
            error_summary = {
                'video_path': str(video_path),
                'video_name': video_name,
                'video_index': video_index,
                'processing_status': 'failed',
                'error': str(e),
                'error_trace': error_trace,
                'processing_time_seconds': processing_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(video_output_dir / "processing_summary.json", 'w') as f:
                json.dump(error_summary, f, indent=2)
            
            return error_summary
    
    def process_all_videos(self, parallel=False):  # Default to sequential for stability
        """
        Process all videos in the directory with enhanced progress tracking
        
        Args:
            parallel: Whether to use parallel processing (default False for stability)
            
        Returns:
            List of processing results
        """
        # Find all video files
        video_files = self.find_videos()
        
        if not video_files:
            print("❌ No valid video files found!")
            return []
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting processing of {len(video_files)} videos...")
        print(f"📁 Output directory: {self.output_directory}")
        
        start_time = time.time()
        
        if parallel and len(video_files) > 1:
            print(f"⚡ Using parallel processing with {self.max_workers} workers")
            
            # Process videos in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_video = {
                    executor.submit(self.process_single_video, video_path, i): (video_path, i)
                    for i, video_path in enumerate(video_files)
                }
                
                results = []
                
                # Process completed futures as they finish
                for future in tqdm(as_completed(future_to_video), 
                                 total=len(video_files), 
                                 desc="🎬 Processing videos"):
                    video_path, video_index = future_to_video[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Show progress
                        if result['processing_status'] == 'success':
                            print(f"✅ Completed {video_index + 1}/{len(video_files)}: {video_path.name}")
                        else:
                            print(f"❌ Failed {video_index + 1}/{len(video_files)}: {video_path.name}")
                            
                    except Exception as e:
                        print(f"❌ Parallel processing error for {video_path.name}: {e}")
                        error_result = {
                            'video_path': str(video_path),
                            'video_name': video_path.stem,
                            'video_index': video_index,
                            'processing_status': 'failed',
                            'error': f"Parallel processing error: {str(e)}"
                        }
                        results.append(error_result)
        else:
            print("🔄 Using sequential processing for maximum stability")
            results = []
            
            for i, video_path in enumerate(video_files):
                print(f"\n📹 Processing {i + 1}/{len(video_files)}: {video_path.name}")
                result = self.process_single_video(video_path, i)
                results.append(result)
                
                # Show intermediate progress
                successful = len([r for r in results if r['processing_status'] == 'success'])
                failed = len([r for r in results if r['processing_status'] == 'failed'])
                print(f"📊 Progress: {i + 1}/{len(video_files)} completed ({successful} ✅, {failed} ❌)")
        
        processing_time = time.time() - start_time
        
        # Save overall processing summary
        successful_results = [r for r in results if r['processing_status'] == 'success']
        failed_results = [r for r in results if r['processing_status'] == 'failed']
        
        overall_summary = {
            'total_videos': len(video_files),
            'successful_videos': len(successful_results),
            'failed_videos': len(failed_results),
            'total_sequences': sum(r.get('num_sequences', 0) for r in successful_results),
            'total_duration_seconds': sum(r.get('duration_seconds', 0) for r in successful_results),
            'total_processing_time_seconds': processing_time,
            'processing_mode': 'parallel' if parallel else 'sequential',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_results': results
        }
        
        with open(self.output_directory / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        # Print comprehensive summary
        print(f"\n{'='*70}")
        print("🎉 MULTI-VIDEO PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total videos: {overall_summary['total_videos']}")
        print(f"✅ Successful: {overall_summary['successful_videos']}")
        print(f"❌ Failed: {overall_summary['failed_videos']}")
        print(f"🎯 Success rate: {(overall_summary['successful_videos']/overall_summary['total_videos']*100):.1f}%")
        print(f"📈 Total sequences: {overall_summary['total_sequences']:,}")
        print(f"⏱️ Total duration: {overall_summary['total_duration_seconds']/60:.1f} minutes")
        print(f"🚀 Processing time: {processing_time/60:.1f} minutes")
        print(f"📁 Output directory: {self.output_directory}")
        
        if failed_results:
            print(f"\n❌ Failed videos:")
            for result in failed_results:
                print(f"  - {result['video_name']}: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*70}")
        
        return results
    
    def combine_datasets(self):
        """
        Combine all processed datasets into a single training dataset
        
        Returns:
            Path to combined dataset directory
        """
        print("\\nCombining datasets from all videos...")
        
        # Find all video output directories
        video_dirs = [d for d in self.output_directory.iterdir() 
                     if d.is_dir() and d.name.startswith("video_")]
        
        if not video_dirs:
            raise ValueError("No processed video directories found!")
        
        print(f"Found {len(video_dirs)} processed video directories")
        
        # Combine audio and target sequences
        all_audio_sequences = []
        all_target_sequences = []
        all_vad_sequences = []
        
        combined_metadata = {
            'num_videos': len(video_dirs),
            'video_sources': [],
            'total_sequences': 0,
            'total_duration_minutes': 0
        }
        
        for video_dir in sorted(video_dirs):
            try:
                # Load sequences
                audio_seq = np.load(video_dir / "audio_sequences.npy")
                target_seq = np.load(video_dir / "target_sequences.npy") 
                vad_seq = np.load(video_dir / "vad_sequences.npy")
                
                # Load metadata
                with open(video_dir / "dataset_metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                # Load processing summary
                with open(video_dir / "processing_summary.json", 'r') as f:
                    processing_info = json.load(f)
                
                print(f"  {video_dir.name}: {len(audio_seq)} sequences")
                
                all_audio_sequences.append(audio_seq)
                all_target_sequences.append(target_seq)
                all_vad_sequences.append(vad_seq)
                
                combined_metadata['video_sources'].append({
                    'directory': video_dir.name,
                    'video_name': processing_info.get('video_name', 'unknown'),
                    'num_sequences': len(audio_seq),
                    'duration_seconds': processing_info.get('duration_seconds', 0)
                })
                
                combined_metadata['total_sequences'] += len(audio_seq)
                combined_metadata['total_duration_minutes'] += processing_info.get('duration_seconds', 0) / 60
                
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load {video_dir.name}: {e}")
        
        if not all_audio_sequences:
            raise ValueError("No valid sequences found!")
        
        # Concatenate all sequences
        combined_audio = np.concatenate(all_audio_sequences, axis=0)
        combined_targets = np.concatenate(all_target_sequences, axis=0)
        combined_vad = np.concatenate(all_vad_sequences, axis=0)
        
        print(f"\\nCombined dataset:")
        print(f"  Audio sequences: {combined_audio.shape}")
        print(f"  Target sequences: {combined_targets.shape}")
        print(f"  VAD sequences: {combined_vad.shape}")
        print(f"  Total duration: {combined_metadata['total_duration_minutes']:.1f} minutes")
        
        # Save combined dataset
        combined_dir = self.output_directory / "combined_dataset"
        combined_dir.mkdir(exist_ok=True)
        
        np.save(combined_dir / "audio_sequences.npy", combined_audio)
        np.save(combined_dir / "target_sequences.npy", combined_targets)
        np.save(combined_dir / "vad_sequences.npy", combined_vad)
        
        # Copy scalers from first video (they should all be similar)
        first_video_dir = video_dirs[0]
        import shutil
        if (first_video_dir / "audio_scaler.pkl").exists():
            shutil.copy(first_video_dir / "audio_scaler.pkl", combined_dir / "audio_scaler.pkl")
        if (first_video_dir / "target_scaler.pkl").exists():
            shutil.copy(first_video_dir / "target_scaler.pkl", combined_dir / "target_scaler.pkl")
        
        # Save combined metadata
        combined_metadata['dataset_shape'] = {
            'audio_sequences': list(combined_audio.shape),
            'target_sequences': list(combined_targets.shape),
            'vad_sequences': list(combined_vad.shape)
        }
        
        with open(combined_dir / "dataset_metadata.json", 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        print(f"\\n✅ Combined dataset saved to: {combined_dir}")
        return combined_dir

def main():
    """
    Main function with command line arguments
    """
    parser = argparse.ArgumentParser(description="Process multiple videos for training")
    parser.add_argument("--video-dir", type=str, default="videodata", 
                       help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="multi_video_features",
                       help="Output directory for processed features")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential processing instead of parallel")
    
    args = parser.parse_args()
    
    print("=== MULTI-VIDEO PROCESSOR ===")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if video directory exists, if not provide helpful message
    if not Path(args.video_dir).exists():
        print(f"\\n❌ Video directory not found: {args.video_dir}")
        print("\\nTo use a custom path, run:")
        print("python data_preparation_scripts/05_multi_video_processor.py --video-dir C:\\Users\\USER\\Videos\\mlVideoData")
        print("\\nOr create the directory and add video files:")
        print(f"mkdir {args.video_dir}")
        return
    
    try:
        # Initialize processor
        processor = MultiVideoProcessor(
            video_directory=args.video_dir,
            output_directory=args.output_dir,
            max_workers=args.workers
        )
        
        # Process all videos
        results = processor.process_all_videos(parallel=not args.sequential)
        
        # Combine datasets
        combined_dataset_dir = processor.combine_datasets()
        
        print(f"\\n🎉 Multi-video processing completed!")
        print(f"Combined dataset ready for training at: {combined_dataset_dir}")
        
    except Exception as e:
        print(f"\\n❌ Error in multi-video processing: {e}")

if __name__ == "__main__":
    main()