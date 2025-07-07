#!/usr/bin/env python
"""
Background subtraction processor for DeepLabCut video preprocessing.
Splits video into chunks, processes in parallel, and recombines efficiently.
Run as a standalone script with real-time terminal output and progress bars.
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import json
import logging


# Configure logging for multiprocessing debug
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(processName)s - %(message)s',
    datefmt='%H:%M:%S'
)


def print_progress(message, level="INFO"):
    """
    Print progress message with timestamp.
    
    Args:
        message (str): Message to print
        level (str): Message level (INFO, WARNING, ERROR, SUCCESS, DEBUG)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Colour codes for different levels
    colours = {
        "INFO": "\033[94m",      # Blue
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "SUCCESS": "\033[92m",   # Green
        "DEBUG": "\033[95m",     # Magenta
        "RESET": "\033[0m"       # Reset
    }
    
    colour = colours.get(level, colours["INFO"])
    reset = colours["RESET"]
    
    print(f"{colour}[{timestamp}] {level}: {message}{reset}")
    sys.stdout.flush()


def get_video_info(video_path):
    """
    Extract video properties using ffprobe.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video properties (fps, duration, frame_count, width, height)
    """
    print_progress(f"Getting video information for: {os.path.basename(video_path)}")
    
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
        '-show_streams', str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        
        # Parse frame rate
        fps_parts = video_stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        info = {
            'fps': fps,
            'duration': float(data['format']['duration']),
            'frame_count': int(video_stream['nb_frames']),
            'width': int(video_stream['width']),
            'height': int(video_stream['height'])
        }
        
        print_progress(f"Video info: {info['frame_count']} frames, {info['fps']:.2f} fps, {info['duration']:.1f}s", "SUCCESS")
        print_progress(f"Resolution: {info['width']}x{info['height']}", "SUCCESS")
        
        return info
        
    except Exception as e:
        print_progress(f"Error getting video info: {e}", "ERROR")
        return None


def split_video_ffmpeg(video_path, output_dir, num_chunks):
    """
    Split video into a specific number of chunks using ffmpeg.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save chunks
        num_chunks (int): Number of chunks to create (matches number of processors)
        
    Returns:
        list: Paths to created chunks
    """
    print_progress(f"Splitting video into {num_chunks} chunks (one per processor)...")
    
    video_info = get_video_info(video_path)
    if not video_info:
        raise ValueError("Could not get video information")
    
    total_duration = video_info['duration']
    chunk_duration = total_duration / num_chunks
    
    print_progress(f"Each chunk will be {chunk_duration:.1f} seconds long")
    
    chunk_paths = []
    
    # Use tqdm for chunk creation progress
    with tqdm(total=num_chunks, desc="Creating chunks", unit="chunk") as pbar:
        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * chunk_duration
            end_time = min((chunk_idx + 1) * chunk_duration, total_duration)
            
            # Handle the last chunk edge case
            if chunk_idx == num_chunks - 1:
                end_time = total_duration
            
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.avi")
            
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Seek before input for fast seeking
                '-i', str(video_path),
                '-t', str(end_time - start_time),
                '-c:v', 'copy',          # Stream copy - no re-encoding!
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-v', 'quiet',           # Suppress ffmpeg output
                '-y',
                chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print_progress(f"Error creating chunk {chunk_idx}: {result.stderr.decode()}", "ERROR")
                raise RuntimeError(f"FFmpeg failed for chunk {chunk_idx}")
            
            chunk_paths.append(chunk_path)
            pbar.update(1)
    
    print_progress(f"Successfully created {len(chunk_paths)} chunks", "SUCCESS")
    return chunk_paths


def create_background_model_opencv(video_path, sample_frames=50):
    """
    Create background model using OpenCV by sampling frames.
    
    Args:
        video_path (str): Path to video file
        sample_frames (int): Number of frames to sample
        
    Returns:
        np.ndarray: Background model
    """
    print_progress(f"Creating background model from {sample_frames} sample frames...")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < sample_frames:
        sample_frames = total_frames
        print_progress(f"Video has only {total_frames} frames, using all for background model", "WARNING")
    
    # Sample frame indices evenly throughout video
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    frames = []
    
    # Use tqdm for frame sampling progress
    with tqdm(total=len(frame_indices), desc="Sampling frames", unit="frame") as pbar:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame.astype(np.float32))
            pbar.update(1)
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not read any frames from video")
    
    print_progress("Computing median background model...")
    background = np.median(frames, axis=0).astype(np.uint8)
    print_progress("Background model created successfully", "SUCCESS")
    
    return background


def initialise_worker():
    """
    Initialise worker process for multiprocessing.
    Ensures each process has proper OpenCV configuration.
    """
    # Set OpenCV to use single thread per process to avoid conflicts
    cv2.setNumThreads(1)


def process_chunk_with_background_subtraction(chunk_info):
    """
    Process a single video chunk with background subtraction.
    
    Args:
        chunk_info (tuple): (chunk_path, background_path, output_path, params, chunk_idx, total_chunks)
        
    Returns:
        tuple: (output_path, frame_count, processing_time, chunk_name, chunk_idx)
    """
    chunk_path, background_path, output_path, params, chunk_idx, total_chunks = chunk_info
    
    start_time = time.time()
    chunk_name = os.path.basename(chunk_path)
    process_name = current_process().name
    
    # Debug output
    logging.info(f"Starting processing of chunk {chunk_idx + 1}/{total_chunks}: {chunk_name}")
    
    try:
        # Load background model
        background = cv2.imread(background_path)
        if background is None:
            raise ValueError(f"Failed to load background from {background_path}")
        
        # Open input video
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video {chunk_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_in_chunk = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Chunk {chunk_idx + 1} has {total_frames_in_chunk} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Failed to create output video writer for {output_path}")
        
        frame_count = 0
        # Process frames without internal progress bar to avoid conflicts
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply background subtraction
            processed_frame = subtract_background(
                frame, background, 
                params['threshold'], 
                params['background_alpha']
            )
            
            out.write(processed_frame)
            frame_count += 1
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logging.info(f"Chunk {chunk_idx + 1}: Processed {frame_count}/{total_frames_in_chunk} frames")
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        logging.info(f"Completed chunk {chunk_idx + 1}: {frame_count} frames in {processing_time:.2f}s")
        
        return output_path, frame_count, processing_time, chunk_name, chunk_idx
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
        raise


def subtract_background(frame, background, threshold=30, background_alpha=0.3):
    """
    Apply background subtraction to a single frame.
    
    Args:
        frame (np.ndarray): Current frame
        background (np.ndarray): Background model
        threshold (int): Threshold for foreground detection
        background_alpha (float): How much to darken background (0-1)
        
    Returns:
        np.ndarray: Processed frame
    """
    # Calculate absolute difference
    diff = cv2.absdiff(frame, background)
    
    # Convert to grayscale for thresholding
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Create binary mask of foreground
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to create result
    result = frame.copy()
    background_darkened = (background * background_alpha).astype(np.uint8)
    result[mask == 0] = background_darkened[mask == 0]
    
    return result


def combine_chunks_ffmpeg(chunk_paths, output_path):
    """
    Combine processed chunks back into single video using ffmpeg.
    
    Args:
        chunk_paths (list): Paths to processed chunks
        output_path (str): Path for output video
    """
    print_progress(f"Combining {len(chunk_paths)} processed chunks into final video...")
    
    # Create file list for ffmpeg concat
    concat_file = output_path.replace('.avi', '_concat.txt')
    
    with open(concat_file, 'w') as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{os.path.abspath(chunk_path)}'\n")
    
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-v', 'quiet',  # Suppress ffmpeg output
        '-y', output_path
    ]
    
    print_progress("Running ffmpeg concat command...")
    result = subprocess.run(cmd, capture_output=True)
    
    # Clean up concat file
    os.remove(concat_file)
    
    if result.returncode != 0:
        print_progress(f"Error combining chunks: {result.stderr.decode()}", "ERROR")
        raise RuntimeError("FFmpeg failed to combine chunks")
    
    print_progress(f"Final video saved: {output_path}", "SUCCESS")


def process_video_with_background_subtraction(video_path, 
                                            threshold=10, background_alpha=0,
                                            num_processes=None):
    """
    Main function to process video with background subtraction.
    
    Args:
        video_path (str): Path to input video
        threshold (int): Background subtraction threshold
        background_alpha (float): Background darkening factor
        num_processes (int): Number of parallel processes
    
    Returns:
        str: Path to output video
    """
    # Generate output filename automatically
    input_dir = os.path.dirname(video_path)
    input_basename = os.path.basename(video_path)
    input_name, input_ext = os.path.splitext(input_basename)
    
    # Create descriptive output filename
    alpha_str = str(background_alpha).replace('.', '')
    output_path = os.path.join(input_dir, f"{input_name}_bg_subtracted_t{threshold}_a{alpha_str}{input_ext}")
    
    if num_processes is None:
        num_processes = min(cpu_count(), 64)  # Use all CPUs up to 16
    
    print_progress("=" * 60)
    print_progress("BACKGROUND SUBTRACTION PROCESSOR", "SUCCESS")
    print_progress("=" * 60)
    print_progress(f"Input video: {video_path}")
    print_progress(f"Output video: {output_path}")
    print_progress(f"Parameters: threshold={threshold}, background_alpha={background_alpha}")
    print_progress(f"Parallel processes: {num_processes}")
    print_progress(f"Number of chunks: {num_processes} (one per processor)")
    print_progress("=" * 60)
    
    overall_start_time = time.time()
    
    # Create temp directory in the same parent folder as the video
    video_parent_dir = os.path.dirname(video_path)
    temp_dir_name = f"bg_subtraction_temp_{int(time.time())}"
    temp_dir = os.path.join(video_parent_dir, temp_dir_name)
    
    print_progress(f"Creating temporary directory: {temp_dir}")
    
    chunks_dir = os.path.join(temp_dir, 'chunks')
    processed_dir = os.path.join(temp_dir, 'processed')
    os.makedirs(chunks_dir)
    os.makedirs(processed_dir)
    
    try:
        # Step 1: Create background model
        print_progress("STEP 1: Creating background model...")
        background = create_background_model_opencv(video_path)
        background_path = os.path.join(temp_dir, 'background.png')
        cv2.imwrite(background_path, background)
        print_progress("Background model saved", "SUCCESS")
        
        # Step 2: Split video into chunks (one per processor)
        print_progress(f"STEP 2: Splitting video into {num_processes} chunks...")
        chunk_paths = split_video_ffmpeg(video_path, chunks_dir, num_processes)
        
        # Step 3: Process all chunks in parallel
        print_progress(f"STEP 3: Processing {len(chunk_paths)} chunks in parallel...")
        print_progress("All chunks will process simultaneously (one per CPU core)")
        
        # Prepare chunk processing arguments
        chunk_args = []
        for i, chunk_path in enumerate(chunk_paths):
            output_chunk_path = os.path.join(processed_dir, f"processed_chunk_{i:04d}.avi")
            params = {
                'threshold': threshold,
                'background_alpha': background_alpha
            }
            chunk_args.append((chunk_path, background_path, output_chunk_path, params, i, len(chunk_paths)))
        
        # Process all chunks in parallel
        processed_chunk_paths = []
        total_frames_processed = 0
        
        parallel_start_time = time.time()
        print_progress("Starting parallel processing of chunks...", "DEBUG")
        print_progress(f"Creating multiprocessing pool with {num_processes} processes", "DEBUG")
        
        try:
            # Use imap_unordered for better progress tracking
            with Pool(num_processes, initializer=initialise_worker) as pool:
                print_progress("Pool created successfully", "DEBUG")
                
                # Create progress bar for chunk completion
                with tqdm(total=len(chunk_args), desc="Processing chunks", unit="chunk") as pbar:
                    # Use imap_unordered to get results as they complete
                    for result in pool.imap_unordered(process_chunk_with_background_subtraction, chunk_args):
                        result_path, frame_count, proc_time, chunk_name, chunk_idx = result
                        processed_chunk_paths.append((chunk_idx, result_path))
                        total_frames_processed += frame_count
                        pbar.set_postfix({'frames': total_frames_processed, 'last_time': f'{proc_time:.1f}s'})
                        pbar.update(1)
                        print_progress(f"Completed {chunk_name}: {frame_count} frames in {proc_time:.2f}s", "SUCCESS")
                
                print_progress("All chunks submitted, waiting for completion...", "DEBUG")
        
        except Exception as e:
            print_progress(f"Error during parallel processing: {str(e)}", "ERROR")
            raise
        
        # Sort processed chunks by index to maintain order
        processed_chunk_paths.sort(key=lambda x: x[0])
        processed_chunk_paths = [path for _, path in processed_chunk_paths]
        
        parallel_time = time.time() - parallel_start_time
        print_progress(f"All chunks processed! Total frames: {total_frames_processed}", "SUCCESS")
        print_progress(f"Parallel processing completed in {parallel_time:.2f}s", "SUCCESS")
        
        # Step 4: Combine processed chunks
        print_progress("STEP 4: Combining processed chunks...")
        combine_chunks_ffmpeg(processed_chunk_paths, output_path)
        
        # Calculate final statistics
        total_time = time.time() - overall_start_time
        
        # Get file sizes
        input_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print_progress("=" * 60)
        print_progress("PROCESSING COMPLETED SUCCESSFULLY!", "SUCCESS")
        print_progress("=" * 60)
        print_progress(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print_progress(f"Parallel processing time: {parallel_time:.2f} seconds")
        print_progress(f"Efficiency: {len(chunk_paths)} chunks processed simultaneously")
        print_progress(f"Frames processed: {total_frames_processed}")
        print_progress(f"Processing rate: {total_frames_processed/total_time:.1f} frames/second")
        print_progress(f"Input file size: {input_size:.1f} MB")
        print_progress(f"Output file size: {output_size:.1f} MB")
        print_progress(f"Output saved to: {output_path}")
        print_progress("=" * 60)
        
        return output_path
        
    except Exception as e:
        print_progress(f"Processing failed: {e}", "ERROR")
        raise
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print_progress(f"Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print_progress("Temporary files cleaned up successfully", "SUCCESS")
            except Exception as e:
                print_progress(f"Warning: Could not clean up temp directory: {e}", "WARNING")
                print_progress(f"You may need to manually delete: {temp_dir}", "WARNING")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Background subtraction video processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python background_subtraction_processor.py video.avi
  python background_subtraction_processor.py video.avi --threshold 45 --background_alpha 0.2
  python background_subtraction_processor.py video.avi --num_processes 8

Output file will be automatically created in the same directory with descriptive naming.
Chunks are created in a temporary folder within the video's parent directory.
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--threshold', type=int, default=30, 
                       help='Background subtraction threshold (default: 30)')
    parser.add_argument('--background_alpha', type=float, default=0.3,
                       help='Background darkening factor 0-1 (default: 0.3)')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of parallel processes (default: auto, max 16)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print_progress(f"Input video file not found: {args.input_video}", "ERROR")
        sys.exit(1)
    
    # Check dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_progress("ffmpeg not found. Please install ffmpeg.", "ERROR")
        sys.exit(1)
    
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_progress("ffprobe not found. Please install ffmpeg.", "ERROR")
        sys.exit(1)
    
    # Check for tqdm
    try:
        import tqdm
    except ImportError:
        print_progress("tqdm not found. Install with: pip install tqdm", "ERROR")
        sys.exit(1)
    
    try:
        output_path = process_video_with_background_subtraction(
            args.input_video,
            threshold=args.threshold,
            background_alpha=args.background_alpha,
            num_processes=args.num_processes
        )
        
        print_progress("Ready for DeepLabCut analysis!", "SUCCESS")
        
    except Exception as e:
        print_progress(f"Processing failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()